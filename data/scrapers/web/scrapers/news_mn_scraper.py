import re
import time
import requests
from bs4 import BeautifulSoup
from utils import make_row

SOURCE_NAME = "news.mn"
REQUEST_DELAY = 1.5

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "mn,en;q=0.9",
    "Referer": "https://news.mn/",
}

RE_CYRILLIC = re.compile(r'[\u0400-\u04FF]{3,}')

_session = requests.Session()
_session.headers.update(HEADERS)

def fetch(url, retries=2):
    for attempt in range(retries + 1):
        try:
            r = _session.get(url, timeout=(5, 10))
            r.raise_for_status()
            r.encoding = "utf-8"
            return BeautifulSoup(r.text, "lxml")
        except Exception as e:
            if attempt < retries:
                _session.close()
                time.sleep(2)
                continue
            print(f"  [ERROR] {e}")
            return None

def get_depth(div):
    m = re.search(r"depth-(\d+)", " ".join(div.get("class", [])))
    return int(m.group(1)) if m else 1

def get_int(el, selector):
    found = el.select_one(selector)
    if found:
        try:
            return int(found.get_text(strip=True))
        except ValueError:
            pass
    return 0

def extract(div):
    ct = div.select_one("div.comment-text")
    if not ct:
        return None, 0, 0

    text = ""
    for p in ct.select("p"):
        if not p.select("p"):
            t = p.get_text(separator=" ", strip=True)
            if t and len(t) > text.__len__():
                text = t

    likes    = get_int(ct, "span.it-vote-item.it-vote-up i span")
    dislikes = get_int(ct, "span.it-vote-item.it-vote-down i span")

    return text, likes, dislikes

def scrape_news_mn(article_url, category=""):
    rows = []
    row_id = 1

    soup = fetch(article_url)
    if not soup:
        return rows

    comment_divs = [
        d for d in soup.find_all("div", class_="comment")
        if "depth-" in " ".join(d.get("class", []))
    ]

    for div in comment_divs:
        text, likes, dislikes = extract(div)
        if not text:
            continue
        if not RE_CYRILLIC.search(text):
            continue
        depth = get_depth(div)
        rows.append(make_row(row_id, text, likes, dislikes,
                             SOURCE_NAME, 0 if depth == 1 else 1, category=category))
        row_id += 1

    if rows:
        print(f"  news.mn: {len(rows)} comments from {article_url.split('/')[-2]}")

    time.sleep(REQUEST_DELAY)
    return rows
