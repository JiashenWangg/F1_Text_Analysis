##############################################################
#  predict_pipeline.R
#  Stylometric Speaker-Classification Pipeline
#  Predicts whether a transcript was spoken by Max Verstappen
#  using the saved LASSO model, then queries an LLM for an
#  enriched, natural-language interpretation.
##############################################################
#
#  PREREQUISITES
#    install.packages(c("quanteda","quanteda.extras","udpipe",
#                       "glmnet","tidyverse","httr2","jsonlite"))
#
#  USAGE
#    source("predict_pipeline.R")
#    result <- predict_mv(raw_text = "It was a good race today...")
#    print(result)
#
#  REQUIRED FILES (same paths used in project.qmd)
#    english-ewt-ud-2.5-191206.udpipe   – UDPipe language model
#    lasso_model.rds                    – serialised glmnet model
#    feature_scaler.rds                 – list(center=, scale=) from training
#
#  API KEY
#    Set the environment variable ANTHROPIC_API_KEY before running, e.g.:
#      Sys.setenv(ANTHROPIC_API_KEY = "sk-ant-...")
##############################################################

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyverse)
  library(quanteda)
  library(quanteda.extras)   # for additional token helpers
  library(udpipe)
  library(glmnet)
  library(httr2)
  library(jsonlite)
})

# ─────────────────────────────────────────────────────────────
# 0.  Load persistent assets (model, scaler, UDPipe model)
# ─────────────────────────────────────────────────────────────

.load_assets <- function(udpipe_path  = "english-ewt-ud-2.5-191206.udpipe",
                         model_path   = "lasso_model.rds",
                         scaler_path  = "feature_scaler.rds") {
  
  if (!file.exists(udpipe_path))
    stop("UDPipe model not found: ", udpipe_path)
  if (!file.exists(model_path))
    stop("LASSO model not found: ", model_path)
  if (!file.exists(scaler_path))
    stop("Feature scaler not found: ", scaler_path)
  
  list(
    ud_model = udpipe_load_model(udpipe_path),
    lasso    = readRDS(model_path),
    scaler   = readRDS(scaler_path)   # list(center = named_vec, scale = named_vec)
  )
}

# ─────────────────────────────────────────────────────────────
# 1.  Pre-process raw text
#     Returns a quanteda corpus with a single document
# ─────────────────────────────────────────────────────────────

.preprocess <- function(raw_text) {
  # Normalise whitespace, strip stray control characters
  clean <- raw_text |>
    str_squish() |>
    str_replace_all("[[:cntrl:]]", " ")
  
  corpus(clean, docnames = "input_doc")
}

# ─────────────────────────────────────────────────────────────
# 2.  Extract the same 30 features used during training
# ─────────────────────────────────────────────────────────────

# 2a. Keyword & pronoun features (quanteda-based, no POS needed)
.extract_surface_features <- function(corp) {
  
  toks <- tokens(corp,
                 remove_punct   = TRUE,
                 remove_numbers = TRUE,
                 remove_symbols = TRUE,
                 what           = "word") |>
    tokens_tolower()
  
  sent_toks   <- tokens(corp, what = "sentence")
  token_count <- ntoken(toks)
  avg_word    <- vapply(toks, function(x) mean(nchar(x)), numeric(1))
  avg_sentence <- token_count / lengths(sent_toks)
  
  keywords <- c("they", "struggle", "happy", "difficult",
                "tough",  "win",     "fast",  "fight")
  
  # Use lapply so the result is always a named list regardless of doc count,
  # then index with [[ ]] — avoids the sapply vector/matrix ambiguity.
  toks_list <- as.list(toks)
  kw_hits <- lapply(stats::setNames(keywords, keywords), function(k)
    vapply(toks_list, function(x) as.integer(k %in% x), integer(1))
  )
  
  count_i  <- vapply(toks_list, function(x) sum(x == "i"),  numeric(1))
  count_we <- vapply(toks_list, function(x) sum(x == "we"), numeric(1))
  
  tibble(
    doc_id       = docnames(corp),
    token_count  = token_count,
    avg_word     = avg_word,
    avg_sentence = avg_sentence,
    they         = kw_hits[["they"]],
    struggle     = kw_hits[["struggle"]],
    happy        = kw_hits[["happy"]],
    difficult    = kw_hits[["difficult"]],
    tough        = kw_hits[["tough"]],
    win          = kw_hits[["win"]],
    fast         = kw_hits[["fast"]],
    fight        = kw_hits[["fight"]],
    i            = (count_i  / token_count) * 1000,
    we           = (count_we / token_count) * 1000
  )
}

# 2b. POS proportion features (UDPipe-based)
.extract_pos_features <- function(corp, ud_model) {
  
  text_df <- tibble(
    doc_id = docnames(corp),
    text   = as.character(corp)
  )
  
  anno <- udpipe_annotate(ud_model,
                          x      = text_df$text,
                          doc_id = text_df$doc_id) |>
    as.data.frame()
  
  pos_df <- anno |>
    dplyr::count(doc_id, upos, name = "pos_count") |>
    tidyr::complete(doc_id, upos, fill = list(pos_count = 0))
  
  total_tok <- anno |>
    dplyr::count(doc_id, name = "total_tokens")
  
  pos_wide <- pos_df |>
    left_join(total_tok, by = "doc_id") |>
    mutate(pos_percent = pos_count / total_tokens * 100) |>
    select(doc_id, upos, pos_percent) |>
    tidyr::pivot_wider(names_from = upos, values_from = pos_percent,
                       values_fill = 0)
  
  # Ensure all expected POS columns exist (fill 0 if UDPipe did not emit them)
  expected_pos <- c("ADJ","ADP","ADV","AUX","CCONJ","DET","INTJ",
                    "NOUN","NUM","PART","PRON","PROPN","PUNCT",
                    "SCONJ","SYM","VERB","X")
  for (p in expected_pos) {
    if (!p %in% names(pos_wide)) pos_wide[[p]] <- 0
  }
  
  pos_wide |> select(doc_id, all_of(expected_pos))
}

# 2c. Combine all features into the model matrix
.build_feature_matrix <- function(surface, pos) {
  
  predictors <- c(
    "token_count","they","struggle","happy","difficult","tough",
    "win","fast","fight","i","we","avg_word","avg_sentence",
    "ADJ","ADP","ADV","AUX","CCONJ","DET","INTJ","NOUN","NUM",
    "PART","PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X"
  )
  
  full <- surface |>
    left_join(pos, by = "doc_id") |>
    select(all_of(c("doc_id", predictors)))
  
  list(
    feature_df  = full,
    feature_mat = as.matrix(full[, predictors]),
    predictor_names = predictors
  )
}

# ─────────────────────────────────────────────────────────────
# 3.  Scale features with training-set parameters and run model
# ─────────────────────────────────────────────────────────────

.run_model <- function(feature_mat, lasso_model) {
  
  prob <- predict(lasso_model, feature_mat, type = "response")[1, 1]
  
  label <- if (prob >= 0.5) "Max Verstappen" else "Other Driver"
  
  list(
    label = label,
    probability = round(prob, 4)
  )
}

# ─────────────────────────────────────────────────────────────
# 4.  Build a structured summary of the prediction
# ─────────────────────────────────────────────────────────────

.build_summary <- function(feature_df, prediction) {
  
  f <- as.list(feature_df[1, ])   # single-row tibble → named list
  
  list(
    prediction = list(
      label       = prediction$label,
      probability = prediction$probability,
      confidence  = dplyr::case_when(
        prediction$probability >= 0.80 | prediction$probability <= 0.20 ~ "High",
        prediction$probability >= 0.65 | prediction$probability <= 0.35 ~ "Moderate",
        TRUE ~ "Low"
      )
    ),
    linguistic_features = list(
      token_count  = f$token_count,
      avg_word_len = round(f$avg_word, 3),
      avg_sent_len = round(f$avg_sentence, 3),
      pronouns_per_1k = list(i = round(f$i, 3), we = round(f$we, 3)),
      keyword_presence = list(
        they     = as.integer(f$they),
        difficult= as.integer(f$difficult),
        win      = as.integer(f$win),
        happy    = as.integer(f$happy),
        struggle = as.integer(f$struggle),
        tough    = as.integer(f$tough),
        fast     = as.integer(f$fast),
        fight    = as.integer(f$fight)
      ),
      pos_proportions = list(
        NOUN  = round(f$NOUN,  2),
        VERB  = round(f$VERB,  2),
        ADJ   = round(f$ADJ,   2),
        ADV   = round(f$ADV,   2),
        PRON  = round(f$PRON,  2),
        AUX   = round(f$AUX,   2),
        DET   = round(f$DET,   2),
        ADP   = round(f$ADP,   2),
        PART  = round(f$PART,  2),
        PROPN = round(f$PROPN, 2)
      )
    )
  )
}

# ─────────────────────────────────────────────────────────────
# 5.  Query the Anthropic LLM API with the structured summary
# ─────────────────────────────────────────────────────────────

.query_llm <- function(summary_obj,
                       api_key = Sys.getenv("HF_TOKEN"),
                       model = "openai/gpt-oss-120b") {
  
  if (nchar(api_key) == 0) {
    stop("HF_TOKEN is not set. Run: Sys.setenv(HF_TOKEN = '...')")
  }
  
  system_prompt <- paste(
    "You are an expert in computational linguistics and Formula 1.",
    "You will receive a JSON object containing the output of a LASSO logistic regression model.",
    "Write a short explanation of the prediction in 3 to 4 sentences.",
    "Return a single valid JSON object with exactly this field:",
    '"explanation": string.',
    "Do not return predicted_label or probability.",
    "Do not output anything outside the JSON object."
  )
  
  user_content <- paste(
    "Here is the structured prediction summary:\n",
    jsonlite::toJSON(summary_obj, auto_unbox = TRUE, pretty = TRUE)
  )
  
  resp <- httr2::request("https://router.huggingface.co/v1/chat/completions") |>
    httr2::req_headers(
      Authorization = paste("Bearer", api_key),
      `Content-Type` = "application/json"
    ) |>
    httr2::req_body_json(list(
      model = model,
      messages = list(
        list(role = "system", content = system_prompt),
        list(role = "user", content = user_content)
      ),
      temperature = 0.2,
      max_tokens = 400
    )) |>
    httr2::req_error(is_error = function(resp) FALSE) |>
    httr2::req_perform()
  
  status <- httr2::resp_status(resp)
  ctype  <- httr2::resp_header(resp, "content-type")
  txt    <- httr2::resp_body_string(resp)
  
  if (status != 200) {
    stop("Hugging Face API error (HTTP ", status, "):\n", txt)
  }
  
  if (is.null(ctype) || !grepl("json", ctype, ignore.case = TRUE)) {
    stop("Expected JSON but got content type: ", ctype, "\nRaw response:\n", txt)
  }
  
  body <- jsonlite::fromJSON(txt, simplifyVector = FALSE)
  raw_text <- body$choices[[1]]$message$content
  
  parsed <- tryCatch(
    jsonlite::fromJSON(raw_text, simplifyVector = FALSE),
    error = function(e) NULL
  )
  
  if (!is.null(parsed) && !is.null(parsed$explanation)) {
    list(
      predicted_label = as.character(summary_obj$prediction$label),
      probability = as.numeric(summary_obj$prediction$probability),
      explanation = as.character(parsed$explanation)
    )
  } else {
    list(
      predicted_label = as.character(summary_obj$prediction$label),
      probability = as.numeric(summary_obj$prediction$probability),
      explanation = as.character(raw_text)
    )
  }
}

# ─────────────────────────────────────────────────────────────
# 6.  Master pipeline function
# ─────────────────────────────────────────────────────────────

#' Predict whether a transcript was spoken by Max Verstappen
#'
#' @param raw_text   Character string — the press-conference transcript.
#' @param assets     Optional list returned by .load_assets(). If NULL the
#'                   assets are loaded fresh on every call (slower but simpler).
#' @param api_key    Anthropic API key. Defaults to ANTHROPIC_API_KEY env var.
#' @param verbose    Print progress messages if TRUE.
#'
#' @return A named list with:
#'   $predicted_label  — "Max Verstappen" or "Other Driver"
#'   $probability      — model probability of Verstappen (0–1)
#'   $explanation      — LLM natural-language explanation
#'   $features         — full feature data.frame (for debugging)
#'
predict_mv <- function(raw_text,
                       assets  = NULL,
                       api_key = Sys.getenv("HF_TOKEN"),
                       verbose = TRUE) {
  
  .msg <- function(...) if (verbose) message("[predict_mv] ", ...)
  
  # ── Load assets ──────────────────────────────────────────
  if (is.null(assets)) {
    .msg("Loading model assets…")
    assets <- .load_assets()
  }
  
  # ── Step 1: Pre-process ───────────────────────────────────
  .msg("Pre-processing text…")
  corp <- .preprocess(raw_text)
  
  # Guard: require at least 200 tokens (same threshold used in training)
  tok_n <- ntoken(tokens(corp, remove_punct = TRUE,
                         remove_numbers = TRUE, remove_symbols = TRUE))
  if (tok_n < 200)
    warning("Input has only ", tok_n, " tokens; model was trained on docs ",
            "with ≥ 200 tokens. Predictions may be unreliable.")
  
  # ── Step 2: Extract features ──────────────────────────────
  .msg("Extracting surface features…")
  surface <- .extract_surface_features(corp)
  
  .msg("Extracting POS features (UDPipe — may take a moment)…")
  pos <- .extract_pos_features(corp, assets$ud_model)
  
  feat <- .build_feature_matrix(surface, pos)
  
  # ── Step 3: Run model ─────────────────────────────────────
  .msg("Running LASSO model…")
  prediction <- .run_model(feat$feature_mat, assets$lasso)
  .msg(sprintf("  → label: %s  |  P(MV) = %.4f",
               prediction$label, prediction$probability))
  
  # ── Step 4: Build structured summary ─────────────────────
  summary_obj <- .build_summary(feat$feature_df, prediction)
  
  # ── Step 5: Query LLM ────────────────────────────────────
  .msg("Querying Anthropic API for natural-language explanation…")
  llm_result <- .query_llm(summary_obj, api_key = api_key)
  
  # ── Assemble final output ─────────────────────────────────
  output <- list(
    predicted_label = llm_result$predicted_label,
    probability     = llm_result$probability,
    explanation     = llm_result$explanation,
    features        = feat$feature_df   # for inspection / debugging
  )
  
  class(output) <- c("mv_prediction", "list")
  output
}

# ─────────────────────────────────────────────────────────────
# S3 print method for clean console output
# ─────────────────────────────────────────────────────────────

print.mv_prediction <- function(x, ...) {
  cat("╔══════════════════════════════════════════════════╗\n")
  cat("║   F1 Speaker Identification — Prediction Result  ║\n")
  cat("╚══════════════════════════════════════════════════╝\n\n")
  cat(sprintf("  Predicted Speaker : %s\n",  x$predicted_label))
  cat(sprintf("  P(Max Verstappen) : %.4f\n", x$probability))
  cat("\n  ── LLM Explanation ─────────────────────────────\n\n")
  cat(strwrap(x$explanation, width = 72, prefix = "  "), sep = "\n")
  cat("\n")
  invisible(x)
}

# ─────────────────────────────────────────────────────────────
# Optional helper: save the trained model & scaler from the
# project.qmd workspace so this pipeline can load them.
#
#   After running project.qmd in RStudio, call:
#     save_training_artifacts(lasso_model, x_train)
# ─────────────────────────────────────────────────────────────

#' Serialize the trained LASSO model and feature scaler to disk
#'
#' Call this once from the project.qmd environment after the model is fitted.
#'
#' @param lasso_model  The fitted glmnet object (trained on scaled features).
#' @param x_train      The raw (unscaled) training matrix — used to recover
#'                     center and scale vectors.
#' @param model_path   Output path for the model RDS file.
#' @param scaler_path  Output path for the scaler RDS file.
#'
save_training_artifacts <- function(lasso_model,
                                    x_train,
                                    model_path  = "lasso_model.rds",
                                    scaler_path = "feature_scaler.rds") {
  saveRDS(lasso_model, model_path)
  message("Model saved to: ", model_path)
  
  sc <- scale(x_train)
  scaler <- list(
    center = attr(sc, "scaled:center"),
    scale  = attr(sc, "scaled:scale")
  )
  saveRDS(scaler, scaler_path)
  message("Scaler saved to: ", scaler_path)
  
  invisible(list(model_path = model_path, scaler_path = scaler_path))
}