"""Extract accuracy/F1 mentions from the two previous-year diploma PDFs."""
import re, sys, os
import pdfplumber
sys.stdout.reconfigure(encoding="utf-8")

PDFS = [
    r"C:\Users\M Tech\Desktop\diplom\latex file\previous year diploma about sentiment analyzing.pdf",
    r"C:\Users\M Tech\Desktop\diplom\latex file\another previous years diploma about sentiment.pdf",
]

ACC_RE = re.compile(
    r"(accuracy|нарийвчлал|нийт нарийвчлал|нарийвчлалын|f1[- ]score|macro[- ]f1|"
    r"weighted[- ]f1|f1[- ]хэмжүүр)\b[^\n]{0,80}",
    re.IGNORECASE,
)
NUM_RE = re.compile(r"\b(0\.[5-9]\d{1,3}|[5-9]\d(\.\d{1,2})?\s?%|[8-9][0-9](\.\d{1,2})?\s?%)")

for path in PDFS:
    print("=" * 80)
    print(f"FILE: {os.path.basename(path)}")
    print("=" * 80)
    if not os.path.exists(path):
        print("  MISSING")
        continue
    with pdfplumber.open(path) as pdf:
        n_pages = len(pdf.pages)
        print(f"  pages: {n_pages}")
        all_hits = []
        for i, page in enumerate(pdf.pages):
            try:
                txt = page.extract_text() or ""
            except Exception as e:
                continue
            # Find any sentence containing accuracy / F1 keywords
            for m in ACC_RE.finditer(txt):
                start = max(0, m.start() - 40)
                end = min(len(txt), m.end() + 80)
                snippet = txt[start:end].replace("\n", " ")
                snippet = re.sub(r"\s+", " ", snippet).strip()
                all_hits.append((i + 1, snippet))
            # Also find any standalone high-percentage numbers
            for m in NUM_RE.finditer(txt):
                start = max(0, m.start() - 60)
                end = min(len(txt), m.end() + 30)
                ctx = txt[start:end].replace("\n", " ")
                ctx = re.sub(r"\s+", " ", ctx).strip()
                # Only keep if there's some sentiment/classification context nearby
                if re.search(r"(test|val|train|accuracy|нарийвчл|f1|model|ангил|"
                             r"үр дүн|results|score)", ctx, re.IGNORECASE):
                    all_hits.append((i + 1, f"[NUM] {ctx}"))
        # Dedup by snippet
        seen = set()
        for pg, s in all_hits:
            key = s[:100]
            if key in seen:
                continue
            seen.add(key)
            print(f"  p.{pg:>3d}: {s[:200]}")
        print()
