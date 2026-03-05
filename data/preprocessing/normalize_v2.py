import os
import sys
import re
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

WORK_DIR = os.path.dirname(os.path.abspath(__file__))

CYRILLIC = (
)
CYR_CHAR_RE = re.compile(f"[{CYRILLIC}]")

REPEAT_RE = re.compile(f"([{CYRILLIC}])\\1{{2,}}")

CENSOR_CHARS = set("*_")
ONLY_CENSOR_RE   = re.compile(r"^[*_]+$")
CENSOR_IN_WORD_RE= re.compile(r"[*_]+")

SAFE_LOOKALIKES = {
    "a": "а", "e": "е", "o": "о", "c": "с", "p": "р", "x": "х",
    "y": "у", "k": "к",
    "A": "А", "E": "Е", "O": "О", "C": "С", "P": "Р", "X": "Х",
    "Y": "У", "K": "К",
}

HTML_RE  = re.compile(r"<[^>]+>")
URL_RE   = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

EMOJI_RE = re.compile(
    "["
    "]+"
)

DIGIT_RUN_RE = re.compile(r"\d+")

WS_RE = re.compile(r"\s+")

PLACEHOLDERS = ("[EMOJI]", "[URL]", "[NUM]", "[HTML]")

def _is_mostly_cyrillic_word(tok: str) -> bool:
    n_cyr = sum(1 for ch in tok if CYR_CHAR_RE.match(ch))
    n_lat = sum(1 for ch in tok if ch.isascii() and ch.isalpha())
    return n_cyr >= 1 and n_cyr >= n_lat

def _fix_lookalikes(tok: str) -> str:
    if not _is_mostly_cyrillic_word(tok):
        return tok
    return "".join(SAFE_LOOKALIKES.get(ch, ch) for ch in tok)

def _apply_token_rules(s: str) -> str:
    s = REPEAT_RE.sub(r"\1\1", s)
    out = []
    for tok in s.split():
        stripped = tok.strip()
        if not stripped:
            continue
        if stripped in PLACEHOLDERS:
            out.append(stripped)
            continue
        if ONLY_CENSOR_RE.match(stripped):
            continue
        stripped = CENSOR_IN_WORD_RE.sub("", stripped)
        if not stripped:
            continue
        stripped = _fix_lookalikes(stripped)
        out.append(stripped)
    return " ".join(out)

def normalize_mongolian_v2(text: str) -> str:
    if text is None:
        return ""
    s = str(text)

    s = HTML_RE.sub(" ", s)
    s = URL_RE.sub(" [URL] ", s)
    s = EMOJI_RE.sub(" [EMOJI] ", s)
    s = DIGIT_RUN_RE.sub(" [NUM] ", s)

    s = _apply_token_rules(s)

    s = WS_RE.sub(" ", s).strip()
    return s

PUNCT = set('.,;:!?(){}[]"\'-+_*&^%$#@~/<>=`«»–—…\\|')

def cyrillic_ratio(text: str) -> float:
    s = str(text or "")
    s = EMOJI_RE.sub("", s)
    for ph in PLACEHOLDERS:
        s = s.replace(ph, "")
    meaningful = [
        c for c in s
        if (not c.isspace())
        and (not c.isdigit())
        and (c not in PUNCT)
    ]
    if not meaningful:
        return 0.0
    n_cyr = sum(1 for c in meaningful if CYR_CHAR_RE.match(c))
    return n_cyr / len(meaningful)

def _sanity_check():
    cases = [
        ("саааайн",                          "саайн"),
        ("хөөөөе",                            "хөөе"),
        ("х*й",                               "хй"),
        ("му***й",                            "муй"),
        ("Сайхан байна 😍🔥",                  "Сайхан байна [EMOJI]"),
        ("очоорой www.test.mn",               "очоорой [URL]"),
        ("2024 онд 12 бот",                   "[NUM] онд [NUM] бот"),
        ("<b>Сайн</b> уу",                    "Сайн уу"),
    ]
    for src, want in cases:
        got = normalize_mongolian_v2(src)
        ok = "OK" if got == want else "FAIL"
        print(f"  [{ok}] normalize_v2({src!r}) = {got!r}")
        if got != want:
            print(f"        wanted = {want!r}")

def main():
    print("── normalize_v2 sanity tests ───────────────")
    _sanity_check()

    print("\n── cyrillic_ratio tests ───────────────────")
    for s in [
        "Сайн уу",
        "Сайн уу 😍",
        "hello world",
        "Сайн hello",
        "[EMOJI] [EMOJI] [EMOJI]",
    ]:
        print(f"  {cyrillic_ratio(s):.2f}  | {s!r}")

if __name__ == "__main__":
    main()
