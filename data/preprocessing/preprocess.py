import re
import os
import json
import logging
import argparse
import unicodedata
from datetime import datetime
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split

CONFIG = {
    "input_csv": "merged_cleaned_comments.csv",
    "output_dir": "output/preprocessed",

    "csv_encoding": "utf-8-sig",

    "near_dedup_enabled": False,

    "split_train": 0.70,
    "split_val":   0.15,
    "split_test":  0.15,

    "random_seed": 42,
}

LABEL_MAP = {
    "positive": "positive",
    "pos": "positive",
    "эерэг": "positive",

    "neutral": "neutral",
    "neu": "neutral",
    "тэнцвэртэй": "neutral",

    "negative": "negative",
    "neg": "negative",
    "сөрөг": "negative",

    "toxic": "toxic_negative",
    "toxic_negative": "toxic_negative",
    "toxic negative": "toxic_negative",
    "hate": "toxic_negative",
    "хорлонтой": "toxic_negative",

    "constructive": "constructive_criticism",
    "constructive criticism": "constructive_criticism",
    "constructive_criticism": "constructive_criticism",
    "шүүмж": "constructive_criticism",
    "шүүмжлэл": "constructive_criticism",
}

RE_EMOJI = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F926-\U0001FA9F"
    "\U00010000-\U0010FFFF"
    "\u2600-\u2B55"
    "\u200d"
    "\ufe0f"
    "\u3030"
    "]+",
    flags=re.UNICODE,
)

RE_URL         = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
RE_MENTION     = re.compile(r"@\w+")
RE_HTML_TAG    = re.compile(r"<[^>]+>")
RE_HTML_ENTITY = re.compile(r"&[a-zA-Z]{2,6};|&#\d+;")
RE_WHITESPACE  = re.compile(r"\s+")
RE_INVISIBLE   = re.compile(
)

RE_LATIN_WORD  = re.compile(r"\b[a-zA-Z]{2,}\b")

SPAM_PATTERNS = [
    re.compile(r"(линк|link|click here|дарна уу)", re.IGNORECASE),
    re.compile(r"(viber|wechat|telegram)\s*[\d\-]+", re.IGNORECASE),
]

RE_CYRILLIC = re.compile(r"[\u0400-\u04FF]")

FILTER_SUPPORT_REPLIES = True

_SUPPORT_REPLY_RE = re.compile(
    re.IGNORECASE | re.UNICODE
)

_NAME_PREFIX_RE = re.compile(
    r"^(?:[A-Za-z][a-zA-Z]+ ){1,3}(?=(?:Сайн байна уу|сайн байна уу|Танд))",
    re.UNICODE
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("preprocess")

COLUMN_ALIASES = {
    "text":        ["text", "content", "raw_text", "comment", "body"],
    "id":          ["id", "row_id", "comment_id"],
    "likes":       ["likes", "like_count", "reactions"],
    "dislikes":    ["dislikes", "dislike_count"],
    "source":      ["SOURCE", "source", "site", "domain"],
    "relation":    ["RELATION", "relation", "is_reply", "reply"],
    "category":    ["CATEGORY", "category", "label", "class", "sentiment"],
}

def resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for canonical, variants in COLUMN_ALIASES.items():
        for variant in variants:
            if variant in df.columns and variant != canonical:
                rename_map[variant] = canonical
                break
    if rename_map:
        log.info(f"Column rename map applied: {rename_map}")
        df = df.rename(columns=rename_map)
    for canonical in COLUMN_ALIASES:
        if canonical not in df.columns:
            df[canonical] = "" if canonical in ("category", "source") else 0
    return df

def load_data(filepath: str) -> pd.DataFrame:
    log.info(f"Loading data from: {filepath}")
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(filepath, encoding="utf-8", low_memory=False)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    log.info(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
    log.info(f"  Columns: {list(df.columns)}")
    df = resolve_columns(df)
    return df

def normalize_source_names(df: pd.DataFrame) -> pd.DataFrame:
    SOURCE_CANONICAL = {
        "news.mn":   "news.mn",
        "gogo.mn":   "gogo.mn",
        "ikon.mn":   "IKON.mn",
        "medee.mn":  "Medee.mn",
        "e-mongolia": "E-Mongolia",
        "zaluusinfo": "Zaluusinfo",
        "eagle news": "Eagle News",
        "мэдэхгүй зүйлээ асуу": "Мэдэхгүй зүйлээ асуу",
    }
    def _canonical(s):
        return SOURCE_CANONICAL.get(str(s).strip().lower(), str(s).strip())

    before = df["source"].nunique()
    df["source"] = df["source"].apply(_canonical)
    after = df["source"].nunique()
    if before != after:
        log.info(f"  Source names normalised: {before} variants → {after} unique sources")
    return df

def extract_feature_flags(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Extracting feature flags from raw text...")
    raw = df["text"].fillna("").astype(str)

    df["has_url"]   = raw.str.contains(RE_URL).astype(int)
    df["has_emoji"] = raw.apply(lambda t: 1 if RE_EMOJI.search(t) else 0)
    df["has_latin"] = raw.apply(
        lambda t: 1 if RE_LATIN_WORD.search(t) else 0
    )
    df["has_mention"]  = raw.str.contains(RE_MENTION).astype(int)
    df["has_hashtag"]  = raw.apply(lambda t: 1 if RE_HASHTAG.search(t) else 0)
    df["is_reply"]     = (
        pd.to_numeric(df.get("relation", 0), errors="coerce").fillna(0).astype(int)
    )
    df["raw_char_len"] = raw.apply(len)
    df["raw_word_len"] = raw.apply(lambda t: len(t.split()))

    df["maybe_is_spam"] = raw.apply(
        lambda t: int(any(p.search(t) for p in SPAM_PATTERNS))
    )

    return df

def _remove_html(text: str) -> str:
    text = RE_HTML_TAG.sub(" ", text)
    text = RE_HTML_ENTITY.sub(" ", text)
    return text

def _remove_invisible(text: str) -> str:
    return RE_INVISIBLE.sub("", text)

def _normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)

def _collapse_repeated_chars(text: str, max_repeat: int = 3) -> str:
    pattern = re.compile(r"([^\d])\1{" + str(max_repeat) + r",}")
    return pattern.sub(lambda m: m.group(1) * max_repeat, text)

def _normalize_quotes(text: str) -> str:
    text = re.sub(r"[\u2018\u2019\u02bc]", "'", text)
    text = re.sub(r"[\u201c\u201d\u00ab\u00bb]", '"', text)
    return text

_HOMOGLYPH_MAP = str.maketrans({
    'a': 'а', 'e': 'е', 'o': 'о', 'p': 'р', 'c': 'с',
    'x': 'х', 'y': 'у', 'k': 'к', 'm': 'м', 't': 'т',
    'b': 'б', 'v': 'в',
    'n': 'н', 'u': 'у', 'z': 'з', 'i': 'и',
    's': 'с', 'h': 'х', 'S': 'С', 'H': 'Х',
    'A': 'А', 'E': 'Е', 'O': 'О', 'P': 'Р', 'C': 'С',
    'X': 'Х', 'Y': 'У', 'K': 'К', 'M': 'М', 'T': 'Т',
    'B': 'Б', 'N': 'Н', 'Z': 'З', 'I': 'И',
    'V': 'В', 'U': 'У',
})

_RE_MIXED_WORD = re.compile(
    r'\b(?=[\w]*[\u0400-\u04FF][\w]*)(?=[\w]*[a-zA-Z][\w]*)\w+\b'
)

def _normalize_homoglyphs(text: str) -> str:
    def _fix_token(m: re.Match) -> str:
        token = m.group(0)
        return token.translate(_HOMOGLYPH_MAP)

    return _RE_MIXED_WORD.sub(_fix_token, text)

def clean_light(text: str, max_repeat: int = 3) -> str:
    if not text or not isinstance(text, str):
        return ""

    text = _normalize_unicode(text)
    text = _remove_invisible(text)
    text = _remove_html(text)
    text = _normalize_quotes(text)
    text = RE_URL.sub("[URL]", text)
    text = RE_MENTION.sub("[USER]", text)
    text = RE_EMOJI.sub(" [EMOJI] ", text)
    text = _collapse_repeated_chars(text, max_repeat)
    text = RE_WHITESPACE.sub(" ", text).strip()

    return text

def clean_heavy(text: str, max_repeat: int = 3) -> str:
    if not text or not isinstance(text, str):
        return ""

    text = _normalize_unicode(text)
    text = _remove_invisible(text)
    text = _remove_html(text)
    text = _normalize_quotes(text)
    text = RE_URL.sub(" ", text)
    text = RE_MENTION.sub(" ", text)
    text = RE_EMOJI.sub(" ", text)
    text = _collapse_repeated_chars(text, max_repeat=2)
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    text = text.lower()
    text = RE_WHITESPACE.sub(" ", text).strip()

    return text

def apply_cleaning(df: pd.DataFrame, max_repeat: int = 3) -> pd.DataFrame:
    log.info("Applying text cleaning (light + heavy)...")
    df["text_raw"] = df["text"].fillna("").astype(str)

    stripped = df["text_raw"].apply(lambda t: _NAME_PREFIX_RE.sub("", t).strip())
    changed = (stripped != df["text_raw"]).sum()
    if changed:
        log.info(f"  Stripped name-prefix from {changed:,} rows")
        df["text_raw"] = stripped
    df["text_light_clean"] = df["text_raw"].apply(
        lambda t: clean_light(t, max_repeat)
    )
    df["text_ml_clean"]   = df["text_raw"].apply(
        lambda t: clean_heavy(t, max_repeat)
    )
    df["light_char_len"] = df["text_light_clean"].apply(len)
    df["light_word_len"] = df["text_light_clean"].apply(lambda t: len(t.split()))
    df["ml_char_len"]    = df["text_ml_clean"].apply(len)
    df["ml_word_len"]    = df["text_ml_clean"].apply(lambda t: len(t.split()))
    df["char_len"] = df["light_char_len"]
    df["word_len"] = df["light_word_len"]
    return df

def filter_bad_rows(
    df: pd.DataFrame,
    min_chars: int = 5,
    min_words: int = 5,
    max_words: int = 300,
) -> tuple[pd.DataFrame, list[dict]]:
    log.info("Filtering bad rows...")
    removed_log = []

    def _log_removal(rows: pd.DataFrame, reason: str):
        for _, row in rows.iterrows():
            removed_log.append({
                "id": row.get("id", ""),
                "reason": reason,
                "text_preview": str(row.get("text_raw", ""))[:80],
            })

    mask_empty = df["text_light_clean"].apply(lambda t: not isinstance(t, str) or len(t.strip()) == 0)
    _log_removal(df[mask_empty], "empty_text")
    df = df[~mask_empty].copy()

    mask_short_chars = df["char_len"] < min_chars
    _log_removal(df[mask_short_chars], f"too_short_chars (<{min_chars})")
    df = df[~mask_short_chars].copy()

    mask_short_words = df["ml_word_len"] < min_words
    _log_removal(df[mask_short_words], f"too_few_words_ml (<{min_words})")
    df = df[~mask_short_words].copy()

    if max_words and max_words > 0:
        mask_too_long = df["word_len"] > max_words
        _log_removal(df[mask_too_long], f"too_many_words (>{max_words})")
        df = df[~mask_too_long].copy()

    def has_meaningful_text(t: str) -> bool:
        return bool(RE_CYRILLIC.search(t) or RE_LATIN_WORD.search(t))

    mask_symbol_only = ~df["text_light_clean"].apply(has_meaningful_text)
    _log_removal(df[mask_symbol_only], "symbol_or_number_only")
    df = df[~mask_symbol_only].copy()

    def is_single_repeated_word(t: str) -> bool:
        words = str(t).lower().split()
        return len(words) >= 2 and len(set(words)) == 1

    mask_repeated = df["text_light_clean"].apply(is_single_repeated_word)
    _log_removal(df[mask_repeated], "single_word_repeated")
    df = df[~mask_repeated].copy()

    _ARTIFACT_RE = re.compile(r"^#[A-Z]+[?!]?$")
    mask_artifact = df["text_light_clean"].apply(lambda t: bool(_ARTIFACT_RE.match(str(t).strip())))
    _log_removal(df[mask_artifact], "spreadsheet_artifact")
    df = df[~mask_artifact].copy()

    _UI_ARTIFACTS = re.compile(
        re.UNICODE
    )
    _PM_DM = re.compile(
        r"(?:^[^\w]*(?:pm|dm|inbox|хувийн\s+чат)[^\w]*$"
        r"|\b(?:pm|dm)\s+(?:явуул|бич|me|pls|please|send|бичээрэй)\b"
        r"|\b(?:надруу|над\s+луу|надад)\s+(?:pm|dm)\b"
        r"|\bcheck\s+inbox\b"
        r"|\binbox\s+(?:pls|please|руу)\b)",
        re.IGNORECASE | re.UNICODE
    )
    mask_ui = df["text_light_clean"].apply(lambda t: bool(_UI_ARTIFACTS.search(str(t))))
    _log_removal(df[mask_ui], "ui_artifact")
    df = df[~mask_ui].copy()

    mask_pmdm = df["text_light_clean"].apply(lambda t: bool(_PM_DM.search(str(t))))
    _log_removal(df[mask_pmdm], "pm_dm_interaction")
    df = df[~mask_pmdm].copy()

    if FILTER_SUPPORT_REPLIES:
        mask_support = df["text_light_clean"].apply(
            lambda t: bool(_SUPPORT_REPLY_RE.search(str(t)))
        )
        _log_removal(df[mask_support], "support_reply_emongolia")
        df = df[~mask_support].copy()

    log.info(f"  Rows after filtering: {len(df):,}  |  Removed: {len(removed_log):,}")
    return df.reset_index(drop=True), removed_log

def deduplicate_exact(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    log.info("Running exact deduplication (2-pass)...")
    before = len(df)

    df = df.drop_duplicates(subset=["text_light_clean"], keep="first").copy()
    after_pass1 = len(df)
    log.info(f"  Pass 1 (light-clean dedup): removed {before - after_pass1:,}")

    df = df.drop_duplicates(subset=["text_ml_clean"], keep="first").copy()
    after_pass2 = len(df)
    log.info(f"  Pass 2 (ml-clean dedup):    removed {after_pass1 - after_pass2:,}")

    removed = before - after_pass2
    log.info(f"  Total duplicates removed: {removed:,}")
    return df.reset_index(drop=True), removed

def _text_to_char_ngrams(text: str, n: int = 3) -> set:
    text = text.replace(" ", "")
    return {text[i:i+n] for i in range(len(text) - n + 1)} if len(text) >= n else set()

def flag_near_duplicates(
    df: pd.DataFrame,
    ngram_size: int = 3,
    threshold: float = 0.85,
) -> pd.DataFrame:
    log.info("Running near-duplicate detection (this may take a while)...")
    texts   = df["text_light_clean"].tolist()
    ngrams  = [_text_to_char_ngrams(t, ngram_size) for t in texts]
    near_dup_idx = set()

    for i in range(len(ngrams)):
        if i in near_dup_idx:
            continue
        for j in range(i + 1, len(ngrams)):
            if j in near_dup_idx:
                continue
            a, b = ngrams[i], ngrams[j]
            union_size = len(a | b)
            if union_size == 0:
                continue
            jaccard = len(a & b) / union_size
            if jaccard >= threshold:
                near_dup_idx.add(j)

    df["is_near_duplicate"] = 0
    df.loc[df.index[list(near_dup_idx)], "is_near_duplicate"] = 1
    log.info(f"  Near-duplicates flagged: {len(near_dup_idx):,}")
    return df

def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Normalising labels...")

    def _map_label(label) -> str:
        if pd.isna(label) or str(label).strip() == "":
            return "unlabeled"
        key = str(label).strip().lower()
        return LABEL_MAP.get(key, "unlabeled")

    df["label"] = df["category"].apply(_map_label)

    counts = df["label"].value_counts()
    log.info(f"  Label distribution:\n{counts.to_string()}")
    return df

def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    test_ratio:  float = 0.15,
    seed:        int   = 42,
    label_col:   str   = "label",
) -> pd.DataFrame:
    log.info("Creating train/val/test splits...")
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, "Ratios must sum to 1.0"

    labeled_mask = df[label_col] != "unlabeled"
    df_labeled   = df[labeled_mask].copy()
    df_unlabeled = df[~labeled_mask].copy()

    if len(df_labeled) == 0:
        log.warning(
            "No labelled rows found. Performing unlabeled split (no stratification)."
        )
        df["split"] = "unlabeled"
        df = _split_array(df, train_ratio, val_ratio, test_ratio, seed, stratify_col=None)
        return df

    if "is_near_duplicate" in df_labeled.columns:
        df_safe   = df_labeled[df_labeled["is_near_duplicate"] == 0].copy()
        df_neardup_train = df_labeled[df_labeled["is_near_duplicate"] == 1].copy()
    else:
        df_safe          = df_labeled.copy()
        df_neardup_train = pd.DataFrame()

    strat = df_safe[label_col] if df_safe[label_col].nunique() > 1 else None
    try:
        train_idx, valtest_idx = train_test_split(
            df_safe.index,
            test_size=(val_ratio + test_ratio),
            stratify=strat,
            random_state=seed,
        )
    except ValueError:
        log.warning("Stratified split failed (possibly too few samples per class). Falling back to random split.")
        train_idx, valtest_idx = train_test_split(
            df_safe.index,
            test_size=(val_ratio + test_ratio),
            random_state=seed,
        )

    val_test_df = df_safe.loc[valtest_idx]
    strat2 = val_test_df[label_col] if val_test_df[label_col].nunique() > 1 else None
    try:
        val_idx, test_idx = train_test_split(
            val_test_df.index,
            test_size=test_ratio / (val_ratio + test_ratio),
            stratify=strat2,
            random_state=seed,
        )
    except ValueError:
        val_idx, test_idx = train_test_split(
            val_test_df.index,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=seed,
        )

    df.loc[train_idx, "split"]            = "train"
    df.loc[val_idx,   "split"]            = "val"
    df.loc[test_idx,  "split"]            = "test"
    if len(df_neardup_train) > 0:
    df.loc[df_unlabeled.index, "split"]   = "unlabeled"
    df["split"] = df["split"].fillna("train")

    counts = df["split"].value_counts()
    log.info(f"  Split distribution:\n{counts.to_string()}")
    return df

def _split_array(df, train_r, val_r, test_r, seed, stratify_col):
    idx = df.index.tolist()
    tr_idx, vt_idx = train_test_split(idx, test_size=(val_r + test_r), random_state=seed)
    v_idx, te_idx  = train_test_split(vt_idx, test_size=test_r / (val_r + test_r), random_state=seed)
    df.loc[tr_idx, "split"] = "train"
    df.loc[v_idx,  "split"] = "val"
    df.loc[te_idx, "split"] = "test"
    return df

def build_report(
    df_raw:           pd.DataFrame,
    df_final:         pd.DataFrame,
    removed_log:      list[dict],
    exact_dup_count:  int,
) -> dict:
    total_removed_filter = len(removed_log)
    report = {
        "timestamp": datetime.now().isoformat(),
        "rows_before_preprocessing":  len(df_raw),
        "rows_after_preprocessing":   len(df_final),
        "rows_removed_total":         len(df_raw) - len(df_final),
        "exact_duplicates_removed":   exact_dup_count,
        "filter_removed":             total_removed_filter,
        "removal_reasons": Counter(r["reason"] for r in removed_log),
        "label_distribution":         df_final["label"].value_counts().to_dict(),
        "split_distribution":         df_final["split"].value_counts().to_dict(),
        "source_distribution":        df_final["source"].value_counts().to_dict(),
        "avg_light_char_len":         round(df_final["light_char_len"].mean(), 1),
        "avg_light_word_len":         round(df_final["light_word_len"].mean(), 1),
        "avg_ml_char_len":            round(df_final["ml_char_len"].mean(), 1),
        "avg_ml_word_len":            round(df_final["ml_word_len"].mean(), 1),
        "pct_has_url":                round(df_final["has_url"].mean() * 100, 2),
        "pct_has_emoji":              round(df_final["has_emoji"].mean() * 100, 2),
        "pct_has_latin":              round(df_final["has_latin"].mean() * 100, 2),
        "pct_maybe_spam":             round(df_final["maybe_is_spam"].mean() * 100, 2),
    }
    return report

def print_report(report: dict) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print("  PREPROCESSING REPORT")
    print(sep)
    print(f"  Timestamp           : {report['timestamp']}")
    print(f"  Rows before         : {report['rows_before_preprocessing']:,}")
    print(f"  Rows after          : {report['rows_after_preprocessing']:,}")
    print(f"  Total removed       : {report['rows_removed_total']:,}")
    print(f"    Exact duplicates  : {report['exact_duplicates_removed']:,}")
    print(f"    Filter removals   : {report['filter_removed']:,}")
    print()
    print("  Removal reasons:")
    for reason, count in report["removal_reasons"].items():
        print(f"    {reason:<35} {count:>6,}")
    print()
    print("  Label distribution:")
    for label, count in sorted(report["label_distribution"].items()):
        total = report["rows_after_preprocessing"]
        pct   = count / total * 100 if total else 0
        print(f"    {label:<30} {count:>6,}  ({pct:.1f}%)")
    print()
    print("  Split distribution:")
    for split, count in sorted(report["split_distribution"].items()):
        print(f"    {split:<10} {count:>6,}")
    print()
    print("  Text statistics (after light clean):")
    print(f"    Avg char len (light) : {report['avg_light_char_len']}")
    print(f"    Avg word len (light) : {report['avg_light_word_len']}")
    print(f"    Avg char len (ml)    : {report['avg_ml_char_len']}")
    print(f"    Avg word len (ml)    : {report['avg_ml_word_len']}")
    print(f"    % has URL         : {report['pct_has_url']}%")
    print(f"    % has emoji       : {report['pct_has_emoji']}%")
    print(f"    % has Latin       : {report['pct_has_latin']}%")
    print(f"    % maybe spam      : {report['pct_maybe_spam']}%")
    print(sep + "\n")

FINAL_COLUMNS = [
    "id",
    "text_raw",
    "text_light_clean",
    "text_ml_clean",
    "label",
    "split",
    "source",
    "relation",
    "likes",
    "dislikes",
    "light_char_len",
    "light_word_len",
    "ml_char_len",
    "ml_word_len",
    "has_url",
    "has_emoji",
    "has_latin",
    "has_mention",
    "has_hashtag",
    "maybe_is_spam",
    "raw_char_len",
    "raw_word_len",
    "is_reply",
]
FINAL_COLUMNS_DROP_BEFORE_SAVE = ["char_len", "word_len"]

def save_outputs(
    df: pd.DataFrame,
    output_dir: str,
    removed_log: list[dict],
    report: dict,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    present_cols = [c for c in FINAL_COLUMNS if c in df.columns]
    extra_cols   = [c for c in df.columns if c not in FINAL_COLUMNS]
    df = df[present_cols + extra_cols]

    drop_internal = [c for c in ["text", "category", "is_near_duplicate"] + FINAL_COLUMNS_DROP_BEFORE_SAVE if c in df.columns]
    if drop_internal:
        df = df.drop(columns=drop_internal)
        log.info(f"  Dropped internal columns from output: {drop_internal}")

    full_path = os.path.join(output_dir, "comments_preprocessed_full.csv")
    df.to_csv(full_path, index=False, encoding=enc)
    log.info(f"  Saved full dataset → {full_path}")

    for split_name in ["train", "val", "test", "unlabeled"]:
        subset = df[df["split"] == split_name]
        if len(subset) == 0:
            continue
        path = os.path.join(output_dir, f"comments_{split_name}.csv")
        subset.to_csv(path, index=False, encoding=enc)
        log.info(f"  Saved {split_name} split ({len(subset):,} rows) → {path}")

    removal_log_path = os.path.join(output_dir, "removed_rows_log.json")
    with open(removal_log_path, "w", encoding="utf-8") as f:
        json.dump(removed_log[:500], f, ensure_ascii=False, indent=2)
    log.info(f"  Saved removal log (first 500 entries) → {removal_log_path}")

    report_path = os.path.join(output_dir, "preprocessing_report.json")
    report_serialisable = {
        k: dict(v) if isinstance(v, Counter) else v
        for k, v in report.items()
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_serialisable, f, ensure_ascii=False, indent=2)
    log.info(f"  Saved report → {report_path}")

class MongolianPreprocessingPipeline:

    def __init__(self, config: dict = None):
        self.cfg = {**CONFIG, **(config or {})}
        self.removed_log     = []
        self.exact_dup_count = 0
        self.df_raw          = None

    def run(self, input_path: str = None, output_dir: str = None) -> pd.DataFrame:
        input_path = input_path or self.cfg["input_csv"]
        output_dir = output_dir or self.cfg["output_dir"]

        log.info("=" * 60)
        log.info("  MONGOLIAN COMMENT PREPROCESSING PIPELINE")
        log.info("=" * 60)

        df = load_data(input_path)
        self.df_raw = df.copy()

        df = normalize_source_names(df)

        df = extract_feature_flags(df)

        df = apply_cleaning(df, max_repeat=self.cfg["max_repeat_char"])

        df, removed = filter_bad_rows(
            df,
            min_chars=self.cfg["min_chars_light"],
            min_words=self.cfg["min_words"],
            max_words=self.cfg.get("max_words", 300),
        )
        self.removed_log.extend(removed)

        df, dup_count = deduplicate_exact(df)
        self.exact_dup_count = dup_count

        if self.cfg.get("near_dedup_enabled"):
            df = flag_near_duplicates(
                df,
                ngram_size=self.cfg["near_dedup_ngram_size"],
                threshold=self.cfg["near_dedup_threshold"],
            )
        else:
            df["is_near_duplicate"] = 0

        df = normalize_labels(df)

        df = split_dataset(
            df,
            train_ratio=self.cfg["split_train"],
            val_ratio=self.cfg["split_val"],
            test_ratio=self.cfg["split_test"],
            seed=self.cfg["random_seed"],
        )

        report = build_report(self.df_raw, df, self.removed_log, self.exact_dup_count)
        print_report(report)

        save_outputs(df, output_dir, self.removed_log, report)

        log.info("Pipeline complete.")
        return df

def parse_args():
    parser = argparse.ArgumentParser(
        description="Mongolian comment preprocessing pipeline"
    )
    parser.add_argument(
        "--input", type=str, default=CONFIG["input_csv"],
        help="Path to the input CSV or XLSX file"
    )
    parser.add_argument(
        "--output", type=str, default=CONFIG["output_dir"],
        help="Directory to write output files"
    )
    parser.add_argument(
        "--min-chars", type=int, default=CONFIG["min_chars_light"],
        help="Minimum character length after light cleaning"
    )
    parser.add_argument(
        "--min-words", type=int, default=CONFIG["min_words"],
        help="Minimum word count to keep a row"
    )
    parser.add_argument(
        "--near-dedup", action="store_true",
        help="Enable near-duplicate detection (slow for large datasets)"
    )
    parser.add_argument(
        "--near-dedup-threshold", type=float, default=CONFIG["near_dedup_threshold"],
        help="Jaccard similarity threshold for near-duplicate detection"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    overrides = {
        "input_csv":           args.input,
        "output_dir":          args.output,
        "min_chars_light":     args.min_chars,
        "min_words":           args.min_words,
        "near_dedup_enabled":  args.near_dedup,
        "near_dedup_threshold": args.near_dedup_threshold,
    }

    pipeline = MongolianPreprocessingPipeline(config=overrides)
    df_clean = pipeline.run()

    print(f"Final dataset: {len(df_clean):,} rows")
    print(df_clean[["text_raw", "text_light_clean", "text_ml_clean", "label", "split"]].head(5).to_string())
