# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('Chapters/Chapter2.tex', encoding='utf-8') as f:
    lines = f.readlines()

print('Lines before:', len(lines))

# ── Cut 1: §2.8 intro 4-item enumerate ───────────────────────────────────────
# File lines 196-203 (0-indexed 195-202) = "судалгаа хийгдсэн..." + blank line
# + 4-item enumerate = 8 lines → 1 line
intro_merged = (
    "судалгаа хийгдсэн: (1)~суурь судалгаа; (2)~BERT-д суурилсан загвар;\n"
    "(3)~бага хэрэглэгддэг хэлний NLP; (4)~Монгол хэлний NLP судалгаа.\n"
)

# ── Cut 2: §2.8.4 research gaps — 5-item itemize + verbose prose ─────────────
# File lines 247-287 (0-indexed 246-286) = intro 2 lines + blank + 5-item
# itemize (27 lines) + blank + 10-line prose = 41 lines → 11 lines
gaps_new = (
    "Дээрх системчилсэн шинжилгээнээс таван орхигдсон чиглэл ажиглагдав:\n"
    "(1)~Монгол NLP-д Хортой--Бүтээлч ялгалт байхгүй;\n"
    "(2)~MN-BERT 4-ангилалт benchmark нийтэд байхгүй;\n"
    "(3)~хоёр шатлалт архитектур Монголд туршигдаагүй;\n"
    "(4)~нийтэд нээлттэй 4-ангилалт corpus байхгүй;\n"
    "(5)~Монгол хэл бага хэрэглэгддэг NLP benchmark-т байхгүй.\n"
    "Энэхүү ажил эдгээрийг нөхнө: (i)~4-ангилалт датасет бүрдүүлэх;\n"
    "(ii)~хоёр шатлалт MN-BERT архитектур боловсруулах;\n"
    "(iii)~agglutinative онцлогт preprocessing хэрэгжүүлэх;\n"
    "(iv)~уламжлалт ML болон 3-ангилалт MN-BERT-тэй харьцуулах;\n"
    "(v)~датасетийг нийтэд нэвтрүүлэх.\n"
)

# Build:
# lines[:195]    – up to (not incl.) "судалгаа хийгдсэн..." enumerated line
# [intro_merged] – replaces lines[195:203]: 8 lines → 2 lines
# lines[203:246] – "Энэхүү шинжилгээгээр..." + §2.8.1-2.8.3 + §2.8.4 header
# [gaps_new]     – replaces lines[246:287]: 41 lines → 11 lines
# lines[287:]    – blank + §2.9 onwards

new_lines = (
    lines[:195]
    + [intro_merged]
    + lines[203:246]
    + [gaps_new]
    + lines[287:]
)

with open('Chapters/Chapter2.tex', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

with open('Chapters/Chapter2.tex', encoding='utf-8') as f:
    lines2 = f.readlines()
print('Lines after:', len(lines2))
for i, l in enumerate(lines2):
    s = l.strip()
    if s.startswith('\\section') or s.startswith('\\subsection') or s.startswith('\\chapter'):
        print(f'  idx {i}: {s[:70]}')
