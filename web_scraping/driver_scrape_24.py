import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from collections import defaultdict

IN_CSV = "drivers_2024.csv"         # must have at least: url (title optional)
OUT_CSV = "drivers_2024_aggregated.csv"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# ---------------- Driver/team list (2024) ----------------
DRIVER_TEAMS_2024 = {
    "Max Verstappen": "Red Bull",
    "Sergio Pérez": "Red Bull",
    "Charles Leclerc": "Ferrari",
    "Carlos Sainz": "Ferrari",
    "Lewis Hamilton": "Mercedes",
    "George Russell": "Mercedes",
    "Lando Norris": "McLaren",
    "Oscar Piastri": "McLaren",
    "Fernando Alonso": "Aston Martin",
    "Lance Stroll": "Aston Martin",
    "Esteban Ocon": "Alpine",
    "Pierre Gasly": "Alpine",
    "Daniel Ricciardo": "RB",
    "Yuki Tsunoda": "RB",
    "Alex Albon": "Williams",
    "Logan Sargeant": "Williams",
    "Valtteri Bottas": "Sauber",
    "Zhou Guanyu": "Sauber",
    "Kevin Magnussen": "Haas",
    "Nico Hülkenberg": "Haas"
}

DRIVER_NAMES_2024 = list(DRIVER_TEAMS_2024.keys())

# Build initials map from 2024 names (e.g., "LN" -> ["Lando Norris"])
INITIALS_TO_NAMES = defaultdict(list)
for name in DRIVER_NAMES_2024:
    parts = name.split()
    if len(parts) >= 2:
        ini = (parts[0][0] + parts[-1][0]).upper()
        INITIALS_TO_NAMES[ini].append(name)

# ---------------- Title parsing (to fill missing GP/type if needed) ----------------
TITLE_RX = re.compile(
    r"""^\s*FIA\s+(?P<ctype>.+?)\s+press\b.*?[–—-]\s*(?P<gp>.+?)\s*(?:\d{4})?\s*$""",
    re.IGNORECASE
)

def parse_title_for_meta(title: str):
    if not isinstance(title, str):
        return "", ""
    m = TITLE_RX.match(title.strip())
    if not m:
        return "", ""
    ctype = m.group("ctype").strip().lower()
    gp = re.sub(r"[ \u2013\u2014-]+$", "", m.group("gp").strip())
    return ctype, gp

# ---------------- Speaker line parsing ----------------
SPEAKER_LINE = re.compile(
    r"""^\s*(?P<speaker>[^:]{1,80}?)\s*:\s*(?P<text>.+)$""",
    re.IGNORECASE | re.UNICODE | re.DOTALL,
)
NAME_WITH_TEAM = re.compile(r"^\s*(?P<name>.+?)\s*\((?P<team>[^)]+)\)\s*$")

def get_soup(url: str) -> BeautifulSoup:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

def clean_initials(token: str) -> str:
    return re.sub(r"[^A-Za-z]", "", token).upper()

def normalize_name(token: str) -> str:
    t = token.strip()
    for name in DRIVER_NAMES_2024:
        if t.lower() == name.lower():
            return name
    return ""

def resolve_initials(ini: str, present_names: set) -> str:
    candidates = INITIALS_TO_NAMES.get(ini, [])
    if len(candidates) == 1:
        return candidates[0]
    if present_names:
        narrowed = [c for c in candidates if c in present_names]
        if len(narrowed) == 1:
            return narrowed[0]
    return ""

def scrape_article(url: str):
    """
    Return dict: driver -> {'team': str, 'chunks': [str, ...]}
    Only keeps paragraphs that start with a driver full name or initials.
    """
    soup = get_soup(url)
    body = soup.select_one("[data-component='article-body'], .f1-article--rich-text, article") or soup

    # Pass 1: find full-name speakers to help disambiguate initials
    present_names = set()
    for p in body.find_all("p"):
        txt = p.get_text(" ", strip=True)
        if not txt:
            continue
        m = SPEAKER_LINE.match(txt)
        if not m:
            continue
        raw_speaker = m.group("speaker").strip()
        mw = NAME_WITH_TEAM.match(raw_speaker)
        speaker_core = mw.group("name").strip() if mw else raw_speaker
        name_full = normalize_name(speaker_core)
        if name_full:
            present_names.add(name_full)

    # Pass 2: collect only driver lines
    per_driver = {}
    for p in body.find_all("p"):
        txt = p.get_text(" ", strip=True)
        if not txt:
            continue
        m = SPEAKER_LINE.match(txt)
        if not m:
            continue

        raw_speaker = m.group("speaker").strip()
        speech = m.group("text").strip()

        # Optional "(Team)" in speaker token
        mw = NAME_WITH_TEAM.match(raw_speaker)
        if mw:
            speaker_token = mw.group("name").strip()
            team = mw.group("team").strip()
        else:
            speaker_token = raw_speaker
            team = ""

        # Resolve to driver full name
        name_full = normalize_name(speaker_token)
        if not name_full:
            ini = clean_initials(speaker_token)
            if 2 <= len(ini) <= 3:
                name_full = resolve_initials(ini, present_names)

        if not name_full:
            continue  # not a recognized 2024 driver

        team_final = team or DRIVER_TEAMS_2024.get(name_full, "")
        slot = per_driver.setdefault(name_full, {"team": team_final, "chunks": []})
        if not slot["team"] and team_final:
            slot["team"] = team_final
        slot["chunks"].append(speech)

    return per_driver

def main():
    in_df = pd.read_csv(IN_CSV)
    if "url" not in in_df.columns:
        raise ValueError(f"{IN_CSV} must contain a 'url' column.")

    cols = set(in_df.columns)
    need_title = "title" not in cols
    need_gp = "grand_prix" not in cols
    need_conf = "conference_type" not in cols

    titles, gps, ctypes = [], [], []
    for _, r in in_df.iterrows():
        title = r["title"] if not need_title else ""
        url = r["url"]

        if need_gp or need_conf:
            if not title and "title" in r and isinstance(r["title"], str):
                title = r["title"]
            ctype, gp = parse_title_for_meta(title) if title else ("", "")
            if need_gp:
                gps.append(gp)
            if need_conf:
                ctypes.append(ctype)
        if need_title:
            titles.append(title)

    if need_title:
        in_df["title"] = titles
    if need_gp:
        in_df["grand_prix"] = gps
    if need_conf:
        in_df["conference_type"] = ctypes

    rows = []
    for _, r in in_df.iterrows():
        url = r["url"]
        gp = r.get("grand_prix", "")
        conf = r.get("conference_type", "")

        print(f"Scraping: {gp or '[GP?]'} - {conf or '[type?]'} -> {url}")
        per_driver = scrape_article(url)

        for driver, info in per_driver.items():
            text_concat = " ".join(info["chunks"]).strip()
            if not text_concat:
                continue
            rows.append({
                "driver": driver,
                "text": text_concat,
                "grand_prix": gp,
                "conference_type": conf,
                "team": info.get("team", "")
            })

    out_df = pd.DataFrame(rows, columns=["driver", "text", "grand_prix", "conference_type", "team"])
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"Saved {len(out_df)} rows to {OUT_CSV}")

if __name__ == "__main__":
    main()
