import re
import sys
from pathlib import Path

import pandas as pd

INPUT_FILES = ["mongolian_comments.xlsx", "fb_comments.xlsx"]

TEMPLATE_COLUMNS = ["id", "text", "likes", "dislikes", "SOURCE", "RELATION", "CATEGORY"]

OUTPUT_CLEANED_XLSX = "merged_cleaned_comments.xlsx"
OUTPUT_CLEANED_CSV = "merged_cleaned_comments.csv"
OUTPUT_REMOVED_CSV = "removed_comments.csv"

CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
LATIN_RE = re.compile(r"[a-zA-Z]")

URL_RE = re.compile(
    r"https?://[^\s<>\"')\]]+|www\.[^\s<>\"')\]]+",
    re.IGNORECASE,
)

def remove_urls(text: str) -> str:
    return URL_RE.sub("", text)

def normalize_punctuation(text: str) -> str:
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r"-{3,}", "-", text)
    text = re.sub(r"_{3,}", "_", text)
    text = re.sub(r"={3,}", "=", text)
    text = re.sub(r"~{3,}", "~", text)
    return text

def normalize_whitespace(text: str) -> str:
    text = re.sub(r"[\t\n\r\x0b\x0c]+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()

def is_empty(text) -> bool:
    if pd.isna(text):
        return True
    if not isinstance(text, str):
        return True
    s = text.strip()
    return s == "" or s.lower() == "nan"

def is_latin_dominant(text: str) -> bool:
    cyrillic_count = len(CYRILLIC_RE.findall(text))
    latin_count = len(LATIN_RE.findall(text))
    total_alpha = cyrillic_count + latin_count
    if total_alpha == 0:
        return False
    return (cyrillic_count / total_alpha) < CYRILLIC_THRESHOLD

def is_symbol_only(text: str) -> bool:
    return len(CYRILLIC_RE.findall(text)) == 0

def is_too_short(text: str) -> bool:
    cyrillic_count = len(CYRILLIC_RE.findall(text))
    return cyrillic_count < MIN_TEXT_LENGTH

def load_excel_file(filepath: str) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        print(f"ERROR: File not found -- {filepath}")
        sys.exit(1)

    try:
        df = pd.read_excel(filepath, engine="openpyxl")
    except Exception as exc:
        print(f"ERROR: Failed to read {filepath} -- {exc}")
        sys.exit(1)

    print(f"  Loaded {filepath}: {len(df)} rows, columns={list(df.columns)}")
    return df

def standardize_schema(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    col_map = {c.strip().lower(): c.strip() for c in df.columns}
    rename_map = {}
    for expected in TEMPLATE_COLUMNS:
        key = expected.lower()
        if key in col_map and col_map[key] != expected:
            rename_map[col_map[key]] = expected

    if rename_map:
        df = df.rename(columns=rename_map)

    for col in TEMPLATE_COLUMNS:
        if col not in df.columns:
            if col in ("likes", "dislikes", "RELATION"):
                df[col] = 0
            else:
                df[col] = ""
            print(f"  Warning: column '{col}' missing in {source_file}, filled with default")

    return df

def remove_by_mask(
    df: pd.DataFrame,
    mask: pd.Series,
    reason: str,
    removed_rows: list,
    stats: dict,
) -> pd.DataFrame:
    to_remove = df[mask].copy()
    if len(to_remove) > 0:
        to_remove["reason_removed"] = reason
        removed_rows.append(to_remove)
    stats[reason] = int(mask.sum())
    return df[~mask].copy()

def build_removed_log(removed_rows: list) -> pd.DataFrame:
    if not removed_rows:
        return pd.DataFrame(
            columns=["origin_file", "raw_text", "SOURCE", "CATEGORY", "reason_removed"]
        )

    log = pd.concat(removed_rows, ignore_index=True)
    log_columns = ["origin_file", "raw_text", "SOURCE", "CATEGORY", "reason_removed"]
    available = [c for c in log_columns if c in log.columns]
    return log[available]

def finalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.reset_index(drop=True)
    df["id"] = range(1, len(df) + 1)

    for col in ("likes", "dislikes", "RELATION"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df[TEMPLATE_COLUMNS].copy()

def export_outputs(final_df: pd.DataFrame, removed_log: pd.DataFrame) -> None:
    final_df.to_excel(OUTPUT_CLEANED_XLSX, index=False, engine="openpyxl")
    print(f"  Saved: {OUTPUT_CLEANED_XLSX} ({len(final_df)} rows)")

    final_df.to_csv(OUTPUT_CLEANED_CSV, index=False, encoding="utf-8-sig")
    print(f"  Saved: {OUTPUT_CLEANED_CSV}")

    removed_log.to_csv(OUTPUT_REMOVED_CSV, index=False, encoding="utf-8-sig")
    print(f"  Saved: {OUTPUT_REMOVED_CSV} ({len(removed_log)} rows)")

def print_summary(
    total_rows: int,
    file_counts: dict,
    stats: dict,
    final_df_len: int,
    final_source_counts: dict,
) -> None:
    removal_reasons = [
        "empty",
        "url_only",
        "latin_dominant",
        "symbol_only",
        "too_short",
        "duplicate",
    ]

    print("\n" + "=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)
    print(f"  Original total rows:        {total_rows}")
    for fname, count in file_counts.items():
        print(f"    Rows from {fname}:  {count}")
    print()

    for reason in removal_reasons:
        count = stats.get(reason, 0)
        print(f"  Removed ({reason}):  {count:>6}")
    total_removed = sum(stats.get(r, 0) for r in removal_reasons)
    print("  " + "-" * 40)
    print(f"  Total removed:              {total_removed:>6}")
    print(f"  Final cleaned rows:         {final_df_len:>6}")
    print()

    print("  Final rows by source:")
    for fname, count in final_source_counts.items():
        print(f"    {fname}:  {count}")
    print()

    print(f"  Output file: {OUTPUT_CLEANED_XLSX}")
    print(f"  Output file: {OUTPUT_CLEANED_CSV}")
    print(f"  Removed log: {OUTPUT_REMOVED_CSV}")
    print()
    print(f"  Expected range: 33,000 - 35,000 cleaned rows")
    print(f"  Actual result:  {final_df_len} rows")
    print("=" * 60)

def run_pipeline() -> None:
    print("=" * 60)
    print("Mongolian Comments Preprocessing Pipeline")
    print("=" * 60)

    print("\n[Step 1] Loading Excel files...")
    dfs = {}
    for filepath in INPUT_FILES:
        df = load_excel_file(filepath)
        df = standardize_schema(df, filepath)
        df["origin_file"] = filepath
        df["raw_text"] = df["text"].copy()
        dfs[filepath] = df

    print("\n[Step 2] Merging datasets...")
    combined = pd.concat(dfs.values(), ignore_index=True)
    total_rows = len(combined)
    file_counts = combined["origin_file"].value_counts().to_dict()
    print(f"  Combined: {total_rows} rows")
    for fname, count in file_counts.items():
        print(f"    {fname}: {count}")

    removed_rows: list[pd.DataFrame] = []
    stats: dict[str, int] = {}

    print("\n[Step 3] Removing empty rows...")
    mask_empty = combined["text"].apply(is_empty)
    combined = remove_by_mask(combined, mask_empty, "empty", removed_rows, stats)
    print(f"  Removed: {stats['empty']}")

    combined["text"] = combined["text"].astype(str)

    print("\n[Step 4] Removing URLs...")
    combined["text"] = combined["text"].apply(remove_urls)
    mask_url_empty = combined["text"].apply(lambda t: t.strip() == "")
    combined = remove_by_mask(combined, mask_url_empty, "url_only", removed_rows, stats)
    print(f"  Rows that were URL-only (removed): {stats['url_only']}")

    print("\n[Step 5] Removing Latin-dominant comments...")
    mask_latin = combined["text"].apply(is_latin_dominant)
    combined = remove_by_mask(combined, mask_latin, "latin_dominant", removed_rows, stats)
    print(f"  Removed: {stats['latin_dominant']}")

    print("\n[Step 6] Normalizing punctuation...")
    combined["text"] = combined["text"].apply(normalize_punctuation)

    print("\n[Step 7] Normalizing whitespace...")
    combined["text"] = combined["text"].apply(normalize_whitespace)

    print("\n[Step 8] Removing symbol-only / emoji-only rows...")
    mask_symbol = combined["text"].apply(is_symbol_only)
    combined = remove_by_mask(combined, mask_symbol, "symbol_only", removed_rows, stats)
    print(f"  Removed: {stats['symbol_only']}")

    print(f"\n[Step 9] Removing short comments (< {MIN_TEXT_LENGTH} chars)...")
    mask_short = combined["text"].apply(is_too_short)
    combined = remove_by_mask(combined, mask_short, "too_short", removed_rows, stats)
    print(f"  Removed: {stats['too_short']}")

    print("\n[Step 10] Removing duplicates...")
    combined["_dedup_key"] = combined["text"].str.lower().str.strip()
    dup_mask = combined.duplicated(subset="_dedup_key", keep="first")
    combined = remove_by_mask(combined, dup_mask, "duplicate", removed_rows, stats)
    combined = combined.drop(columns=["_dedup_key"])
    print(f"  Removed: {stats['duplicate']}")

    print("\n[Step 11] Building removed-rows audit log...")
    removed_log = build_removed_log(removed_rows)
    print(f"  Total removed rows logged: {len(removed_log)}")

    print("\n[Step 12] Finalizing dataset and exporting...")
    final_source_counts = combined["origin_file"].value_counts().to_dict()

    final_df = finalize_dataset(combined)
    export_outputs(final_df, removed_log)

    print_summary(total_rows, file_counts, stats, len(final_df), final_source_counts)

if __name__ == "__main__":
    run_pipeline()
