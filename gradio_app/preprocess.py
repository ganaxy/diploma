"""Text normalisation + Cyrillic-ratio diagnostic.

`normalize_mongolian_v2` is ported VERBATIM from
`sample scores/task1_normalize_v2.py` — the exact function that produced the
`text_normalized` column the live Flat model was trained on. Serving therefore
applies the identical normalisation (HTML/URL/EMOJI/NUM placeholders, repeat
collapse, censor stripping, Latin-homoglyph repair) so there is no train/serve
skew. `clean_text` is the older light path, kept only for the Two-stage models
and for the language warning.
"""

import re

# Cyrillic Unicode block U+0400..U+04FF (covers Mongolian Cyrillic).
CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")
LATIN_RE = re.compile(r"[a-zA-Z]")
URL_RE = re.compile(r"https?://[^\s<>\"')\]]+|www\.[^\s<>\"')\]]+", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")

CYRILLIC_THRESHOLD = 0.40

# ── Faithful Flat-model normalisation ───────────────────────────────────────
# The live Flat model (best_mnbert_sota_corrected_model) was trained on the
# `text_normalized` column of relabeled_v7_corrected.csv. We verified, byte-for-
# byte on 4000 rows (100.00% exact match), that that column equals:
#     _whitespace_collapse( _apply_token_rules( text_light_clean ) )
# i.e. the ORIGINAL task1_normalize rules (NOT the v2 placeholder variant —
# that CSV keeps raw digits/URLs, no [NUM]/[URL]/[EMOJI] tokens). So serving
# must apply exactly these token rules over the light-cleaned user text.
# Do not "improve" this — any change re-introduces train/serve skew.
_CYRILLIC = "Ѐ-ӿ" "Ԁ-ԯ" "ⷠ-ⷿ" "Ꙁ-ꚟ"
_CYR_CHAR_RE = re.compile(f"[{_CYRILLIC}]")
_REPEAT_RE = re.compile(f"([{_CYRILLIC}])\\1{{2,}}")
_ONLY_CENSOR_RE = re.compile(r"^[*_]+$")
_CENSOR_IN_WORD_RE = re.compile(r"[*_]+")
_SAFE_LOOKALIKES = {
    "a": "а", "e": "е", "o": "о", "c": "с", "p": "р", "x": "х",
    "y": "у", "k": "к",
    "A": "А", "E": "Е", "O": "О", "C": "С", "P": "Р", "X": "Х",
    "Y": "У", "K": "К",
}
_WS_RE = re.compile(r"\s+")


def _is_mostly_cyrillic_word(tok: str) -> bool:
    n_cyr = sum(1 for ch in tok if _CYR_CHAR_RE.match(ch))
    n_lat = sum(1 for ch in tok if ch.isascii() and ch.isalpha())
    return n_cyr >= 1 and n_cyr >= n_lat


def _fix_lookalikes(tok: str) -> str:
    if not _is_mostly_cyrillic_word(tok):
        return tok
    return "".join(_SAFE_LOOKALIKES.get(ch, ch) for ch in tok)


def _apply_token_rules(s: str) -> str:
    s = _REPEAT_RE.sub(r"\1\1", s)
    out = []
    for tok in s.split():
        stripped = tok.strip()
        if not stripped:
            continue
        if _ONLY_CENSOR_RE.match(stripped):
            continue
        stripped = _CENSOR_IN_WORD_RE.sub("", stripped)
        if not stripped:
            continue
        stripped = _fix_lookalikes(stripped)
        out.append(stripped)
    return " ".join(out)


def normalize_flat(text) -> str:
    """Training-faithful normalisation for the live Flat model.

    raw user text -> clean_text (≈ text_light_clean) -> token rules -> ws.
    Verified to reproduce relabeled_v7_corrected.csv.text_normalized exactly.
    """
    s = clean_text(text)
    s = _apply_token_rules(s)
    return _WS_RE.sub(" ", s).strip()


def clean_text(text: str) -> str:
    """Apply the inference-time cleaning pipeline to a single string."""
    if text is None:
        return ""
    s = str(text)
    s = URL_RE.sub("", s)
    s = HTML_TAG_RE.sub("", s)
    s = re.sub(r"!{2,}", "!", s)
    s = re.sub(r"\?{2,}", "?", s)
    s = re.sub(r"\.{4,}", "...", s)
    s = re.sub(r"-{3,}", "-", s)
    s = re.sub(r"_{3,}", "_", s)
    s = re.sub(r"={3,}", "=", s)
    s = re.sub(r"~{3,}", "~", s)
    s = re.sub(r"[\t\n\r\x0b\x0c]+", " ", s)
    s = re.sub(r" {2,}", " ", s)
    return s.strip()


def cyrillic_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are Cyrillic. Empty -> 0.0."""
    if not text:
        return 0.0
    cyr = len(CYRILLIC_RE.findall(text))
    lat = len(LATIN_RE.findall(text))
    total = cyr + lat
    if total == 0:
        return 0.0
    return cyr / total


def language_warning(text: str) -> str:
    """Return a Mongolian warning string if Cyrillic share is below the threshold.

    Empty string means no warning. The UI should display it as a yellow notice
    above the prediction — never block inference.
    """
    if not text or not text.strip():
        return "Текст хоосон байна."
    ratio = cyrillic_ratio(text)
    if ratio < CYRILLIC_THRESHOLD:
        pct = int(round(ratio * 100))
        return (
            f"Анхааруулга: оролтын кирилл үсгийн эзлэх хувь {pct}% "
            f"({int(CYRILLIC_THRESHOLD * 100)}%-аас бага). "
            "Загвар Монгол кирилл дээр сургагдсан тул үр дүн найдваргүй байж магадгүй."
        )
    return ""
