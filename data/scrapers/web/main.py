import os
import pandas as pd
from url_collector import get_news_mn_urls, get_gogo_mn_urls
from scrapers.news_mn_scraper import scrape_news_mn
from scrapers.gogo_mn_scraper import scrape_gogo_mn
from driver_setup import build_driver
from utils import save_to_excel, clean_dataframe

TARGET_PER_SITE = 1000
OUTPUT_FILE     = "output/mongolian_comments.xlsx"
SAVE_EVERY      = 10

DONE_FILE = OUTPUT_FILE.replace(".xlsx", "_done_urls.txt")

def load_done():
    if os.path.exists(DONE_FILE):
        return set(open(DONE_FILE, encoding="utf-8").read().splitlines())
    return set()

def mark_done(url):
    with open(DONE_FILE, "a", encoding="utf-8") as f:
        f.write(url + "\n")

def load_rows():
    if os.path.exists(OUTPUT_FILE):
        try:
            return pd.read_excel(OUTPUT_FILE).to_dict("records")
        except Exception:
            pass
    return []

def save_all(rows):
    os.makedirs("output", exist_ok=True)
    df = clean_dataframe(pd.DataFrame(rows)) if rows else pd.DataFrame()
    if not df.empty:
        df["id"] = range(1, len(df) + 1)
    save_to_excel(df, OUTPUT_FILE)
    return df

def run(name, url_cat_pairs, scrape_fn, all_rows, done, **scrape_kwargs):
    new = 0
    for i, (url, category) in enumerate(url_cat_pairs, 1):
        if url in done:
            continue
        print(f"  [{name}] {i}/{len(url_cat_pairs)}  {url}  [{category}]")
        try:
            rows = scrape_fn(url, category=category, **scrape_kwargs)
        except Exception as e:
            print(f"    [ERROR] {e}")
            rows = []

        all_rows.extend(rows)
        done.add(url)
        mark_done(url)
        new += 1
        print(f"    +{len(rows)} rows  |  total: {len(all_rows)}")

        if new % SAVE_EVERY == 0:
            save_all(all_rows)
            print(f"    ✓ checkpoint saved")

    return all_rows

def check_file_not_locked(filepath):
    if not os.path.exists(filepath):
        return
    try:
        with open(filepath, "a"):
            pass
    except PermissionError:
        print(f"\n  ⚠ '{filepath}' is open in Excel. Close it and press ENTER to continue.")
        input()
        check_file_not_locked(filepath)

def main():
    os.makedirs("output", exist_ok=True)
    check_file_not_locked(OUTPUT_FILE)
    done = load_done()
    all_rows = load_rows()
    if all_rows:
        print(f"Resuming — {len(all_rows)} existing rows, {len(done)} URLs done\n")

    print("=" * 55)
    print("STEP 1: Collecting URLs with comments...")
    print("=" * 55)
    news_urls = get_news_mn_urls(TARGET_PER_SITE, skip_urls=done)
    gogo_urls = get_gogo_mn_urls(TARGET_PER_SITE, skip_urls=done)
    print(f"\n  {len(news_urls) + len(gogo_urls)} new articles to scrape\n")

    print("=" * 55)
    print("STEP 2: Scraping comments...")
    print("=" * 55)
    all_rows = run("news.mn", news_urls, scrape_news_mn, all_rows, done)

    RESTART_EVERY = 50
    gogo_driver = None
    scraped = 0
    try:
        gogo_driver = build_driver(headless=True)
        for i, (url, category) in enumerate(gogo_urls, 1):
            if url in done:
                continue
            if scraped > 0 and scraped % RESTART_EVERY == 0:
                gogo_driver.quit()
                gogo_driver = build_driver(headless=True)
            print(f"  [gogo.mn] {i}/{len(gogo_urls)}  {url}  [{category}]")
            try:
                rows = scrape_gogo_mn(url, category=category, driver=gogo_driver)
            except Exception as e:
                print(f"    [ERROR] {e}")
                rows = []
            all_rows.extend(rows)
            done.add(url)
            mark_done(url)
            scraped += 1
            print(f"    +{len(rows)} rows  |  total: {len(all_rows)}")
            if scraped % SAVE_EVERY == 0:
                save_all(all_rows)
                print(f"    ✓ checkpoint saved")
    finally:
        if gogo_driver:
            gogo_driver.quit()

    df = save_all(all_rows)
    print("\n" + "=" * 55)
    print(f"✓ DONE — {len(df)} rows → {OUTPUT_FILE}")
    for src, cnt in df["SOURCE"].value_counts().items():
        r0 = (df[df.SOURCE == src]["RELATION"] == 0).sum()
        r1 = (df[df.SOURCE == src]["RELATION"] == 1).sum()
        print(f"  {src}: {cnt} rows  ({r0} comments, {r1} replies)")

if __name__ == "__main__":
    main()
