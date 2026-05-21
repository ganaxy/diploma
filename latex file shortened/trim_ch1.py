# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('Chapters/Chapter1.tex', encoding='utf-8') as f:
    lines = f.readlines()

print('Lines before:', len(lines))

# ── §1.3 SMART itemize: 0-indexed 65-85 → inline paragraph ─────────────────
# File lines 66-86: "Зорилгыг SMART зарчмын..." + blank + 5-item itemize
#   + \end{itemize} + blank  (21 lines)
smart_inline = (
    "Зорилгыг SMART зарчмаар товч томьёолбол: \\textit{(S)}~4 ангилалт,\n"
    "хоёр шатлалт MN-BERT архитектур; \\textit{(M)}~macro F1 $\\geq 0.80$,\n"
    "нарийвчлал $\\geq 90\\%$; \\textit{(A)}~pre-trained MN-BERT жин ба\n"
    "SMOTE/Focal Loss тэнцвэржүүлэлт; \\textit{(R)}~Монгол NLP-д\n"
    "орхигдсон Хортой--Бүтээлч ялгалт; \\textit{(T)}~2025.9--2026.5.\n"
    "\n"
)

# ── §1.4 research tasks: 7-item → 5-item enumerate ──────────────────────────
# 0-indexed 91-118 (file lines 92-119, 28 lines) → 13 lines
tasks_new = (
    "\\begin{enumerate}\n"
    "  \\item news.mn, Facebook нийтийн хуудас болон gogo.mn-ийн сэтгэгдлүүдийг\n"
    "    цуглуулж цэвэрлэн, 4 ангилалд гар аргаар шошголно;\n"
    "    аннотацийн нийцлийг Cohen's Kappa-аар хянана.\n"
    "  \\item Монгол хэлний agglutinative онцлогт тохирсон preprocessing pipeline\n"
    "    (токенчлол, стоп үг хасах, нормчлол) боловсруулна.\n"
    "  \\item Хоёр шатлалт архитектур зохион байгуулна: Stage~1 нь\n"
    "    Эерэг/Саармаг/Сөрөг; Stage~2 нь Хортой сөрөг vs Бүтээлч шүүмжлэл.\n"
    "  \\item MN-BERT загварыг нарийн тохируулж LR/NB/SVM суурь загваруудтай\n"
    "    macro F1, Accuracy, Precision, Recall, Confusion Matrix-ээр харьцуулна.\n"
    "  \\item Gradio framework ашиглан хэрэглэгчийн интерфейстэй прототип системийг\n"
    "    хэрэгжүүлж туршилтын орчинд үнэлнэ.\n"
    "\\end{enumerate}\n"
)

# Build:
# lines[:65]    – header through §1.3 intro prose + blank (stops before "Зорилгыг...")
# [smart_inline]– replaces lines[65:86]: 21 lines → 6 lines
# lines[86:91]  – %--- + §1.4 header + blank + intro sentence + blank
# [tasks_new]   – replaces lines[91:119]: 7-item enumerate 28 lines → 13 lines
# lines[119:]   – blank + §1.5 Таамаглал onwards

new_lines = (
    lines[:65]
    + [smart_inline]
    + lines[86:91]
    + [tasks_new]
    + lines[119:]
)

with open('Chapters/Chapter1.tex', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

with open('Chapters/Chapter1.tex', encoding='utf-8') as f:
    lines2 = f.readlines()
print('Lines after:', len(lines2))
for i, l in enumerate(lines2):
    s = l.strip()
    if s.startswith('\\section') or s.startswith('\\chapter'):
        print(f'  idx {i}: {s[:60]}')
