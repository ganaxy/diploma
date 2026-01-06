import os
import logging
import pandas as pd

from fb_compat import save_to_excel, normalize_whitespace, is_valid_text, COLUMNS

logger = logging.getLogger("fb_scraper.checkpoint")

FB_OUTPUT_FILE   = os.path.join("output", "fb_comments.xlsx")
FB_DONE_POSTS    = os.path.join("output", "fb_done_posts.txt")

def load_done_posts() -> set:
    if os.path.exists(FB_DONE_POSTS):
        with open(FB_DONE_POSTS, encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def mark_post_done(post_url: str) -> None:
    os.makedirs(os.path.dirname(FB_DONE_POSTS), exist_ok=True)
    with open(FB_DONE_POSTS, "a", encoding="utf-8") as f:
        f.write(post_url.strip() + "\n")

def load_existing_rows() -> list:
    if os.path.exists(FB_OUTPUT_FILE):
        try:
            df = pd.read_excel(FB_OUTPUT_FILE)
            rows = df.to_dict("records")
            logger.info("Loaded %d existing rows from %s", len(rows), FB_OUTPUT_FILE)
            return rows
        except Exception as e:
            logger.warning("Could not load existing output: %s", e)
    return []

def save_checkpoint(rows: list) -> None:
    if not rows:
        logger.info("No rows to save yet.")
        return
    os.makedirs("output", exist_ok=True)
    df = pd.DataFrame(rows)
    df = clean_dataframe_fb(df)
    if not df.empty:
        df["id"] = range(1, len(df) + 1)
    save_to_excel(df, FB_OUTPUT_FILE)
    logger.info("Checkpoint saved: %d rows → %s", len(df), FB_OUTPUT_FILE)

def clean_dataframe_fb(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        logger.warning("No data to clean.")
        return pd.DataFrame(columns=COLUMNS)

    for col in COLUMNS:
        if col not in df.columns:
            df[col] = "" if col in ("CATEGORY", "SOURCE") else 0

    df["text"] = df["text"].apply(
        lambda x: normalize_whitespace(str(x)) if pd.notna(x) else ""
    )
    df = df[df["text"].apply(is_valid_text)].copy()

    df = df.drop_duplicates(subset=["text", "SOURCE"], keep="first").copy()

    df["likes"]    = pd.to_numeric(df["likes"],    errors="coerce").fillna(0).astype(int)
    df["dislikes"] = pd.to_numeric(df["dislikes"], errors="coerce").fillna(0).astype(int)
    df["RELATION"] = pd.to_numeric(df["RELATION"], errors="coerce").fillna(0).astype(int)
    df["CATEGORY"] = df["CATEGORY"].fillna("").astype(str)
    df["SOURCE"]   = df["SOURCE"].fillna("").astype(str)

    return df.reset_index(drop=True)[COLUMNS]

def print_summary(rows: list, failed_posts: int) -> None:
    if not rows:
        print("\n  No data collected.")
        return
    df = pd.DataFrame(rows)
    total_comments = (df["RELATION"] == 0).sum() if "RELATION" in df.columns else 0
    total_replies  = (df["RELATION"] == 1).sum() if "RELATION" in df.columns else 0
    sources = df["SOURCE"].value_counts() if "SOURCE" in df.columns else {}

    print("\n" + "=" * 55)
    print("  FACEBOOK SCRAPING SUMMARY")
    print("=" * 55)
    print(f"  Total rows        : {len(df)}")
    print(f"  Top-level comments: {total_comments}")
    print(f"  Replies           : {total_replies}")
    print(f"  Failed posts      : {failed_posts}")
    print()
    for src, cnt in sources.items():
        r0 = ((df["SOURCE"] == src) & (df["RELATION"] == 0)).sum()
        r1 = ((df["SOURCE"] == src) & (df["RELATION"] == 1)).sum()
        print(f"  {src}: {cnt} rows  ({r0} comments, {r1} replies)")
    print(f"\n  Output: {FB_OUTPUT_FILE}")
    print("=" * 55)
