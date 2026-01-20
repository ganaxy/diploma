import re
import os
import argparse
import logging

import pandas as pd

CONFIG = {
    "input_csv":        "C:\\Users\\M Tech\\Desktop\\diplom\\annotation\\comments_preprocessed_full.csv",
    "output_csv":       "output/sampled_for_labeling.csv",
    "target_n":         5500,
    "min_ml_words":     5,
    "random_seed":      42,
}

OUTPUT_COLUMNS = [
    "id", "text_raw", "text_light_clean", "text_ml_clean",
    "source", "relation", "likes", "dislikes",
    "light_word_len", "ml_word_len",
    "has_latin", "maybe_is_spam", "is_reply",
    "label", "annotator_notes",
]

_INTERACTION_RE = re.compile(
    r"(?:^[\W]*(?:pm|dm|inbox|хувийн\s+чат)[\W]*$"
    r"|\b(?:pm|dm)\s+(?:явуул|бич|me|pls|please|send|бичээрэй)\b"
    r"|\b(?:надруу|над\s+луу|надад)\s+(?:pm|dm)\b"
    r"|\bcheck\s+inbox\b"
    r"|\binbox\s+(?:pls|please|руу)\b)",
    re.IGNORECASE | re.UNICODE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sampling")

def load_data(filepath: str) -> pd.DataFrame:
    log.info(f"Loading: {filepath}")
    ext = os.path.splitext(filepath)[1].lower()
    df = pd.read_csv(filepath, encoding="utf-8-sig") if ext == ".csv" \
        else pd.read_excel(filepath)
    log.info(f"  Loaded {len(df):,} rows x {len(df.columns)} columns")
    return df.copy()

def filter_data(df: pd.DataFrame, min_ml_words: int = 5) -> pd.DataFrame:
    before = len(df)

    df = df[df["text_ml_clean"].notna()].copy()
    df = df[df["text_ml_clean"].astype(str).str.strip() != ""].copy()

    wl_col = "ml_word_len" if "ml_word_len" in df.columns else None
    if wl_col:
        df = df[df[wl_col] >= min_ml_words].copy()
    else:
        df = df[df["text_ml_clean"].astype(str).apply(
            lambda t: len(t.split()) >= min_ml_words
        )].copy()

    df = df[~df["text_light_clean"].astype(str).apply(
        lambda t: bool(_INTERACTION_RE.search(t))
    )].copy()

    df = df.drop_duplicates(subset=["text_ml_clean"], keep="first").copy()

    log.info(f"  Quality filter: {before:,} -> {len(df):,}  (removed {before - len(df):,})")
    return df.reset_index(drop=True)

def _compute_source_quotas(
    source_counts: pd.Series,
    target_n:      int,
    max_source_pct: float,
) -> dict:
    total_avail    = source_counts.sum()
    max_per_source = int(target_n * max_source_pct)

    proportional = (source_counts / total_avail * target_n).round().astype(int)

    capped   = proportional[proportional > max_per_source].index.tolist()
    uncapped = proportional[proportional <= max_per_source].index.tolist()

    quotas   = {}
    leftover = 0

    for src in source_counts.index:
        if src in capped:
            quotas[src] = max_per_source
            leftover   += int(proportional[src]) - max_per_source
        else:
            quotas[src] = int(proportional[src])

    if leftover > 0 and uncapped:
        uncapped_total = source_counts[uncapped].sum()
        for src in uncapped:
            extra = int(leftover * source_counts[src] / uncapped_total)
            quotas[src] = min(quotas[src] + extra, int(source_counts[src]))

    for src in quotas:
        quotas[src] = min(quotas[src], int(source_counts[src]))

    log.info(f"  Source quotas  (cap = {max_source_pct:.0%} = {max_per_source} rows):")
    for src, q in sorted(quotas.items(), key=lambda x: -x[1]):
        pct = q / target_n * 100
        log.info(f"    {src:<32} quota={q:>4}  share={pct:.1f}%")

    return quotas

def sample_by_source(
    df:             pd.DataFrame,
    target_n:       int,
    max_source_pct: float,
    seed:           int,
) -> pd.DataFrame:
    source_counts = df["source"].value_counts()
    quotas        = _compute_source_quotas(source_counts, target_n, max_source_pct)

    parts = []
    for src, quota in quotas.items():
        pool = df[df["source"] == src]
        n    = min(quota, len(pool))
        if n > 0:
            parts.append(pool.sample(n=n, random_state=seed, replace=False))

    result = pd.concat(parts, ignore_index=True)
    log.info(f"  After source sampling: {len(result):,} rows")
    return result

def enforce_relation_balance(
    df:               pd.DataFrame,
    target_reply_pct: float | None,
    total_target:     int,
    seed:             int,
) -> pd.DataFrame:
    if target_reply_pct is None:
        log.info("  Reply balance: skipped (target_reply_pct=None)")
        return df

    n_replies  = (df["relation"] == 1).sum()
    n_comments = (df["relation"] == 0).sum()
    target_r   = int(total_target * target_reply_pct)
    needed     = target_r - n_replies

    if needed <= 0:
        log.info(
            f"  Reply balance: {n_replies} replies ({100*n_replies/len(df):.1f}%) "
            f">= target {target_reply_pct:.0%} — no change"
        )
        return df

    n_to_drop = min(needed, n_comments // 10)
    log.info(
        f"  Reply balance: {n_replies} replies -> target {target_r} "
        f"(+{needed} needed, dropping {n_to_drop} comments)"
    )

    comments_by_src = df[df["relation"] == 0]["source"].value_counts()
    to_drop         = []
    remaining       = n_to_drop

    for src in comments_by_src.index:
        if remaining <= 0:
            break
        max_drop = min(remaining, int(comments_by_src[src] * 0.15))
        candidates = df[(df["relation"] == 0) & (df["source"] == src)]
        dropped    = candidates.sample(n=max_drop, random_state=seed)
        to_drop.extend(dropped.index.tolist())
        remaining -= max_drop

    df = df.drop(index=to_drop).copy()
    log.info(f"  Dropped {len(to_drop)} comments — new reply ratio: "
             f"{(df['relation']==1).mean():.1%}")

    return df.reset_index(drop=True)

def validate_sample(
    df:             pd.DataFrame,
    target_n:       int,
    max_source_pct: float,
) -> bool:
    log.info("Validating sample...")
    passed = True

    n_dupes = df["text_ml_clean"].duplicated().sum()
    if n_dupes > 0:
        raise ValueError(f"FAIL: {n_dupes} duplicate text_ml_clean rows")
    log.info("  OK  No duplicates in text_ml_clean")

    n_empty = (df["text_ml_clean"].astype(str).str.strip() == "").sum()
    if n_empty > 0:
        raise ValueError(f"FAIL: {n_empty} empty text_ml_clean rows")
    log.info("  OK  No empty texts")

    size_delta = abs(len(df) - target_n) / target_n
    if size_delta > 0.10:
        log.warning(f"  WARN  Size {len(df):,} deviates {size_delta:.0%} from target {target_n:,}")
        passed = False
    else:
        log.info(f"  OK  Size {len(df):,} within 10% of target {target_n:,}")

    for src, pct in df["source"].value_counts(normalize=True).items():
        if pct > max_source_pct + 0.02:
            log.warning(f"  WARN  '{src}' = {pct:.1%} exceeds cap {max_source_pct:.0%}")
            passed = False
        else:
            log.info(f"  OK  {src}: {pct:.1%}")

    log.info(f"  Reply ratio: {(df['relation']==1).mean():.1%}")
    return passed

def print_report(df: pd.DataFrame) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print("  SAMPLING REPORT")
    print(sep)
    print(f"  Total rows sampled  : {len(df):,}")
    print()

    print("  Source distribution:")
    for src, cnt in df["source"].value_counts().items():
        pct = cnt / len(df) * 100
        bar = chr(9608) * int(pct / 2)
        print(f"    {src:<32} {cnt:>5}  {pct:5.1f}%  {bar}")
    print()

    print("  Relation distribution:")
    for rel, cnt in df["relation"].value_counts().sort_index().items():
        label = "comment (0)" if rel == 0 else "reply   (1)"
        print(f"    {label}  {cnt:>5}  ({100*cnt/len(df):.1f}%)")
    print()

    print("  Text length (ml_word_len):")
    s = df["ml_word_len"].describe()
    print(f"    min={s['min']:.0f}  p25={s['25%']:.0f}  median={s['50%']:.0f}  "
          f"mean={s['mean']:.1f}  p75={s['75%']:.0f}  max={s['max']:.0f}")
    print()

    print("  Feature flags in sample:")
    print(f"    has_latin     : {df['has_latin'].sum():>4}  ({100*df['has_latin'].mean():.1f}%)")
    print(f"    maybe_is_spam : {df['maybe_is_spam'].sum():>4}  ({100*df['maybe_is_spam'].mean():.1f}%)")
    print(sep + "\n")

def save_output(df: pd.DataFrame, filepath: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    out = df.copy()
    out["label"]           = ""
    out["annotator_notes"] = ""

    present = [c for c in OUTPUT_COLUMNS if c in out.columns]
    extra   = [c for c in out.columns   if c not in OUTPUT_COLUMNS]
    out = out[present + extra]

    out = out.sample(frac=1, random_state=99).reset_index(drop=True)
    out.index = range(1, len(out) + 1)
    out.index.name = "row_num"

    out.to_csv(filepath, encoding="utf-8-sig")
    log.info(f"  Saved -> {filepath}  ({len(out):,} rows)")

def run_sampling(cfg: dict = None) -> pd.DataFrame:
    cfg = {**CONFIG, **(cfg or {})}

    log.info("=" * 60)
    log.info("  MONGOLIAN COMMENT SAMPLING PIPELINE")
    log.info("=" * 60)

    df     = load_data(cfg["input_csv"])
    df     = filter_data(df, min_ml_words=cfg["min_ml_words"])
    sample = sample_by_source(df, cfg["target_n"], cfg["max_source_pct"], cfg["random_seed"])
    sample = enforce_relation_balance(sample, cfg["target_reply_pct"], cfg["target_n"], cfg["random_seed"])

    validate_sample(sample, cfg["target_n"], cfg["max_source_pct"])
    print_report(sample)
    save_output(sample, cfg["output_csv"])

    log.info("Done.")
    return sample

def parse_args():
    p = argparse.ArgumentParser(description="Mongolian comment sampling pipeline")
    p.add_argument("--input",          type=str,   default=CONFIG["input_csv"])
    p.add_argument("--output",         type=str,   default=CONFIG["output_csv"])
    p.add_argument("--target",         type=int,   default=CONFIG["target_n"])
    p.add_argument("--max-source-pct", type=float, default=CONFIG["max_source_pct"])
    p.add_argument("--reply-pct",      type=float, default=CONFIG["target_reply_pct"])
    p.add_argument("--seed",           type=int,   default=CONFIG["random_seed"])
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_sampling({
        "input_csv":        args.input,
        "output_csv":       args.output,
        "target_n":         args.target,
        "max_source_pct":   args.max_source_pct,
        "target_reply_pct": args.reply_pct,
        "random_seed":      args.seed,
    })
