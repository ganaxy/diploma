# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('Chapters/Chapter5.tex', encoding='utf-8') as f:
    lines = f.readlines()

print('Lines before:', len(lines))

# ── §5.5 Infrastructure: remove 7-item itemize (indices 142-149) ──────────
# Keep: lines[0..141] (intro incl. GPU sentence), lines[150..] (blank + rest)
# The GPU model is already mentioned in the 4-line prose (138-141), so
# the itemize (143-149) just repeats it → remove.

infra_block_start = 142   # '\n' blank before \begin{itemize}
infra_block_end   = 150   # '\n' blank after \end{itemize} (exclusive)

# ── §5.7 Gradio: condense 13 verbose lines (183-195) to 5 ──────────────────
gradio_new = (
    '\n'
    'Эцсийн системийн хэрэглэгчийн интерфейсийг Gradio дээр гурван таб\n'
    '(Ангилах, Харьцуулах, \\Үнэлгээний) бүхий бүтэцтэй бүтээв. '
    'Ангилах таб нь нэг эсвэл batch\n'
    'текст оруулж Flat ($\\mathbf{83.26\\%}$) эсвэл Two-stage ($78.61\\%$)\n'
    'архитектурыг сонгон ангилж, confidence баар харуулна. '
    'Харьцуулах таб нь\n'
    'хоёр архитектурыг зэрэг ажиллуулж каскад алдааны нөлөөг\n'
    '(бүлэг~\\ref{Chapter6}) харуулах бол Үнэлгээний таб нь confusion matrix\n'
    'болон метрикийг статикаар үзүүлнэ. Загваруудыг lazy loading-оор\n'
    'GPU (AMD DirectML) дээр инференс хийнэ.\n'
    '\n'
)

# ── §5.8 Reproducibility: condense 26 lines (200-225) to 8 ─────────────────
repro_new = (
    '\n'
    'Туршилтын давтагдах байдлыг (reproducibility) хангахын тулд дараах\n'
    'арга хэмжээг авсан: random seed ($42$) бүх сангаар тогтворжуулалт;\n'
    'бүх кодыг Git/GitHub-д хувилбарын хяналтад оруулж\n'
    '\\texttt{requirements.txt}-д сангийн яг хувилбарыг бичсэн; epoch тутмын\n'
    'loss/F1 оноог JSON лог файлд хадгалж, confusion matrix-ийг\n'
    '\\texttt{matplotlib}-оор хадгалав; хамгийн сайн validation F1-тэй жинг\n'
    'чекпойнт болгон хадгалав; AMD RX~9070 GRE (DirectML)-ийн гүйцэтгэлийг\n'
    'CPU орчинд баталгаажуулав. Эдгээр нь guideline-ийн Ш-6 шаардлагыг\n'
    'бүрэн хангаж байна.\n'
)

# Build:
# lines[0..141]        = everything up to (but not including) itemize blank
# lines[150..182]      = blank after itemize + §5.6 Тэнцвэржүүлэлт + §5.7 header
# (idx 182 is the blank before Gradio text, 183 is text start)
# gradio_new           = condensed Gradio
# lines[196..200]      = %%--- + §5.8 header (idx 199) + blank
# repro_new            = condensed reproducibility

new_lines = (
    lines[:142]           # 0-141: everything before the itemize blank
    + lines[150:183]      # 150-182: after infra itemize through Gradio header
    + [gradio_new]
    + lines[199:201]      # 199-200: §5.8 header + blank
    + [repro_new]
)

with open('Chapters/Chapter5.tex', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

# Verify
with open('Chapters/Chapter5.tex', encoding='utf-8') as f:
    lines2 = f.readlines()
print('Lines after:', len(lines2))
for i, l in enumerate(lines2):
    s = l.strip()
    if s.startswith('\\section') or s.startswith('\\chapter'):
        print(f'  idx {i}: {s[:60]}')
