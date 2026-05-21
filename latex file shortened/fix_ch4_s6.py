# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('Chapters/Chapter4.tex', encoding='utf-8') as f:
    lines = f.readlines()

print('Total lines:', len(lines))

# §4.6 section header is at index 443 (0-indexed).
# Content block to replace: indices 444-467 (blank + 2 paragraphs + trailing blank).
# Replace with condensed 7-line block.

condensed = (
    '\n'
    # Para 1: 4 challenges condensed into 3 lines
    + 'Монгол хэлний NLP-д дөрвөн тусгай бэрхшээл тулгардаг: '
    + '(1)~agglutinative бүтэц нь vocabulary size-г огцом нэмэгдүүлж '
    + 'BoW/TF-IDF аргыг сулруулдаг; (2)~нийтэд нээлттэй шошгологдсон '
    + 'corpus туйлын хомс; (3)~цахим орчны ярианы хэл стандарт бичгийн '
    + 'хэлнээс тогтворгүйгээр ялгаатай; (4)~хортой агуулгын аннотацийн '
    + 'нийтлэг стандарт байхгүй тул шошгологчдын хоорондын нийцлийг хангах хүнд.\n'
    + '\n'
    # Para 2: Cyrillic limitation condensed into 4 lines
    + '\\textbf{Кирилл-онцлогт хязгаарлалт.} '
    + 'Датасет болон загвар нь зөвхөн кирилл Монгол хэлний текстэд хамаарна; '
    + 'латин, code-switched, болон ``bn\'\'/``zgr\'\'-хэлбэрийн бичлэг '
    + 'хамрах хүрээнд ороогүй. MN-BERT-ийн кирилл корпус дахь '
    + 'морфологийн тогтвортой байдлыг хамгаалах зорилгоор хийсэн энэхүү '
    + 'хязгаарлалтыг ирээдүйн судалгаагаар өргөтгөхөөр төлөвлөж байна.\n'
    + '\n'
)

# Replace indices 444-468 (inclusive) with condensed text
new_lines = lines[:444] + [condensed] + lines[469:]

with open('Chapters/Chapter4.tex', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print('Lines before:', len(lines), '→ Lines after:', len(new_lines))

# Verify
with open('Chapters/Chapter4.tex', encoding='utf-8') as f:
    lines2 = f.readlines()
print('=== §4.6 result (lines 441-460, 0-idx) ===')
for i in range(441, min(460, len(lines2))):
    print(f'{i}: {lines2[i]}', end='')
