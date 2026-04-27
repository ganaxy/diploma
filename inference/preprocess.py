import re

CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")
LATIN_RE = re.compile(r"[a-zA-Z]")
URL_RE = re.compile(r"https?://[^\s<>\"')\]]+|www\.[^\s<>\"')\]]+", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")

CYRILLIC_THRESHOLD = 0.40

def clean_text(text: str) -> str:
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
    if not text:
        return 0.0
    cyr = len(CYRILLIC_RE.findall(text))
    lat = len(LATIN_RE.findall(text))
    total = cyr + lat
    if total == 0:
        return 0.0
    return cyr / total

def language_warning(text: str) -> str:
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
