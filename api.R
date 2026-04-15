library(plumber)
library(jsonlite)

source("predict_pipeline.R")
assets <- .load_assets()

#* @post /predict
function(req, res) {
  body <- jsonlite::fromJSON(req$postBody)
  txt <- body$text
  
  result <- predict_mv(
    raw_text = txt,
    assets = assets,
    verbose = FALSE
  )
  
  list(
    predicted_label = result$predicted_label,
    probability = result$probability,
    explanation = result$explanation,
    features = result$features
  )
}