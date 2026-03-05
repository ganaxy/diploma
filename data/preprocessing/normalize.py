import os
import sys
import re
import random
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORK_DIR)

CYRILLIC = (
)
CYR_CHAR_RE = re.compile(f"[{CYRILLIC}]")

REPEAT_RE = re.compile(f"([{CYRILLIC}])\\1{{2,}}")

CENSOR_CHARS = set("*_")
ONLY_CENSOR_RE = re.compile(r"^[*_]+$")
CENSOR_IN_WORD_RE = re.compile(r"[*_]+")

LATIN_TO_CYR = {
    "a": "а", "e": "е", "o": "о", "c": "с", "p": "р", "x": "х",
    "t": "т", "m": "м", "b": "в", "n": "н",
    "A": "А", "E": "Е", "O": "О", "C": "С", "P": "Р", "X": "Х",
    "Y": "У", "K": "К", "H": "Н", "T": "Т", "M": "М", "B": "В", "N": "Н",
}
SAFE_LOOKALIKES = {
    "a": "а", "e": "е", "o": "о", "c": "с", "p": "р", "x": "х",
    "y": "у", "k": "к",
    "A": "А", "E": "Е", "O": "О", "C": "С", "P": "Р", "X": "Х",
    "Y": "У", "K": "К",
}

def _is_mostly_cyrillic_word(tok: str) -> bool:
    n_cyr = sum(1 for ch in tok if CYR_CHAR_RE.match(ch))
    n_lat = sum(1 for ch in tok if ch.isascii() and ch.isalpha())
    return n_cyr >= 1 and n_cyr >= n_lat

def _fix_lookalikes(tok: str) -> str:
    if not _is_mostly_cyrillic_word(tok):
        return tok
    return "".join(SAFE_LOOKALIKES.get(ch, ch) for ch in tok)

def normalize_mongolian(text: str) -> str:
    if text is None:
        return ""
    s = str(text)

    s = REPEAT_RE.sub(r"\1\1", s)

    out_tokens = []
    for tok in s.split():
        stripped = tok.strip()
        if not stripped:
            continue

        if ONLY_CENSOR_RE.match(stripped):
            continue
        stripped = CENSOR_IN_WORD_RE.sub("", stripped)
        if not stripped:
            continue

        stripped = _fix_lookalikes(stripped)

        out_tokens.append(stripped)

    return " ".join(out_tokens)

def _sanity_check():
    cases = [
        ("саааайн", "саайн"),
        ("хөөөөе", "хөөе"),
        ("х*й",    "хй"),
        ("му***й", "муй"),
    ]
    for src, want in cases:
        got = normalize_mongolian(src)
        ok = "OK" if got == want else "FAIL"
        print(f"  [{ok}] normalize({src!r}) = {got!r}  (expected {want!r})")

def main():
    print("── Sanity tests ─────────────────────────────")
    _sanity_check()

    csv_path = os.path.join(WORK_DIR, "10k_fully_labeled.csv")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    valid_labels = {"POSITIVE", "NEUTRAL", "CONSTRUCTIVE", "TOXIC"}
    n_before = len(df)
    df = df[df["label"].isin(valid_labels)].reset_index(drop=True)
    print(f"\nLoaded {n_before} rows → {len(df)} valid rows "
          f"after label filter.")

    has_upper = df["text_light_clean"].astype(str).apply(
        lambda s: any(c.isupper() for c in s)
    ).sum()
    print(f"Rows with at least one uppercase char in text_light_clean: "
          f"{has_upper}")
    if has_upper == 0:
        print("✅ Step 5 verified — text_light_clean already lowercased, "
              "skipping lowercase.")

    df["text_normalized"] = df["text_light_clean"].astype(str).apply(
        normalize_mongolian
    )

    changed_mask = df["text_normalized"] != df["text_light_clean"].astype(str)
    n_changed = int(changed_mask.sum())
    print(f"\nRows where text_normalized != text_light_clean: {n_changed}"
          f"  ({100.0 * n_changed / len(df):.2f}%)")

    print("\nChanged rows by split:")
    for sp, cnt in df[changed_mask].groupby("split").size().items():
        print(f"  {sp:>5s}: {cnt}")

    train_changed = df[(df["split"] == "train") & changed_mask]
    if len(train_changed) >= 15:
        sample = train_changed.sample(n=15, random_state=42)
    else:
        sample = train_changed
    print(f"\n── 15 before/after examples (train split, "
          f"normalization changed) ──")
    for i, row in enumerate(sample.itertuples(index=False), start=1):
        src_raw = getattr(row, "text_light_clean")
        src_norm = getattr(row, "text_normalized")
        lbl = getattr(row, "label")
        print(f"\n#{i}  id={row.id}  label={lbl}")
        print(f"   before: {src_raw[:200]}")
        print(f"   after : {src_norm[:200]}")

    out_path = os.path.join(WORK_DIR, "10k_fully_labeled_normalized.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved normalized dataframe to {out_path}")
    print(f"✅ TASK 1 complete — {n_changed} rows changed by normalization.")

if __name__ == "__main__":
    main()
