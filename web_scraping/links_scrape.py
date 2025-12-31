import re
import csv
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# ---------- Config ----------
BASE_URL = "https://www.formula1.com/en/latest/tags/press-conferences.55FMj3vhksoIIIQmuKwSuq?page={}"
URLS = [BASE_URL.format(i) for i in range(1, 26)]
HEADERS = {"User-Agent": "Mozilla/5.0"}

OUTFILES = {
    ("drivers", "2022"): "drivers_2022.csv",
    ("drivers", "2023"): "drivers_2023.csv",
    ("drivers", "2024"): "drivers_2024.csv",
    ("drivers", "2025"): "drivers_2025.csv",
    ("team_principals", "2022"): "team_principals_2022.csv",
    ("team_principals", "2023"): "team_principals_2023.csv",
    ("team_principals", "2024"): "team_principals_2024.csv",
    ("team_principals", "2025"): "team_principals_2025.csv",
}

# ---------- Parsing helpers ----------
TITLE_RX = re.compile(
    r"""^\s*FIA\s+(?P<ctype>.+?)\s+press\b.*?[–—-]\s*(?P<gp>.+?)\s*(?:\d{4})?\s*$""",
    re.IGNORECASE,
)


def parse_title(title: str):
    """
    Parse 'conference_type' (lowercase; between 'FIA ' and ' press') and 'grand_prix'
    (location after the dash) from the visible title.
    """
    m = TITLE_RX.match(title.strip())
    if not m:
        return "", ""
    ctype = (
        m.group("ctype").strip().lower()
    )  # keep hyphens/spaces as-is, just lowercase
    gp = re.sub(r"[ \u2013\u2014-]+$", "", m.group("gp").strip())
    return ctype, gp


def year_from_url(url: str) -> str:
    """Extract trailing -YYYY from the article slug only."""
    slug = urlparse(url).path.rstrip("/").split("/")[-1].split(".")[0]
    m = re.search(r"-(20\d{2})$", slug)
    return m.group(1) if m else ""


def is_team_principals(title: str) -> bool:
    """Classify Team Principals vs Drivers from the title (case/quote tolerant)."""
    t = title.lower().replace("’", "'")
    return "team principal" in t  # matches 'team principals’ press conference' etc.


# ---------- Scrape ----------
def scrape_page(url: str):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    rows = []
    for a in soup.select("a[href*='/en/latest/article/']"):
        title = a.get_text(strip=True)
        href = a.get("href")
        if not title or not href:
            continue
        if not title.lower().startswith("fia"):
            continue

        full_url = urljoin(url, href)
        ctype, gp = parse_title(title)
        year = year_from_url(full_url)
        role = "team_principals" if is_team_principals(title) else "drivers"

        rows.append(
            {
                "title": title,
                "url": full_url,
                "grand_prix": gp,
                "conference_type": ctype,
                "year": year,
                "role": role,
            }
        )
    return rows


if __name__ == "__main__":
    seen = set()
    collected = []

    for page in URLS:
        print(f"Scraping {page} ...")
        for row in scrape_page(page):
            if row["url"] not in seen:
                seen.add(row["url"])
                collected.append(row)

    collected = [r for r in collected if r["year"] in {"2022", "2023", "2024", "2025"}]

    # Bucket and write
    header = ["title", "url", "grand_prix", "conference_type", "year"]
    buckets = {k: [] for k in OUTFILES}

    for r in collected:
        key = (r["role"], r["year"])
        if key in buckets:
            buckets[key].append(r)

    for key, outfile in OUTFILES.items():
        rows = buckets[key]
        if not rows:
            print(f"No rows for {key}; skipping.")
            continue
        with open(outfile, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in rows:
                w.writerow([r[h] for h in header])
        print(f"Wrote {len(rows)} rows to {outfile}")
