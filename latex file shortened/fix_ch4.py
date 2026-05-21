# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('Chapters/Chapter4.tex', encoding='utf-8') as f:
    lines = f.readlines()

# Print lines 26-35 (0-indexed) to verify current state
print('=== Current lines 26-35 (0-indexed) ===')
for i in range(26, 36):
    print(f'{i}: {repr(lines[i][:80])}')

# The replacement was already applied (lines 28-31 are the wrong replacement).
# We need to fix the incorrect Mongolian text in those lines.
# Replace indices 28-31 (0-indexed) with correct text.

correct_line_a = (
    'Мэдээний'  # Мэдээний
    ' '
    'сайтын'  # сайтын
    ' '
    'сэтгэгдлд'  # сэтгэгдлд
    ' '
    'бичгийн'  # бичгийн
    ' '
    'хэлний'  # хэлний
    ' '
    'шинжтэй'  # шинжтэй
    ' '
    'урт'  # урт
    ' '
    'текстүүд'  # текстүүд
    ' '
    'зонхилох'  # зонхилох
    ' '
    'бол'  # бол
    ' Facebook '
    'группийн'  # группийн
    ' '
    'сэтгэгдлд'  # сэтгэгдлд
    ' '
    'ярианы'  # ярианы
    ' '
    'хэл,'  # хэл,
    ' emoji '
    'агуулсан'  # агуулсан
    ' '
    'богино'  # богино
    ' '
    'бичлэг'  # бичлэг
    ' '
    'давамгайлах'  # давамгайлах
    '\n'
)

correct_line_b = (
    'нь'  # нь
    ' '
    'өгөгдлийн'  # өгөгдлийн
    ' '
    'олон'  # олон
    ' '
    'талт'  # талт
    ' '
    'байдлыг'  # байдлыг
    ' '
    'хангана.'  # хангана.
    ' '
    'Цуглуулалтыг'  # Цуглуулалтыг
    ' web scraping ('
    'мэдээний'  # мэдээний
    '\n'
)

correct_line_c = (
    'сайтууд)'  # сайтууд)
    ' '
    'болон'  # болон
    ' '
    'нийтийн'  # нийтийн
    ' Graph API (Facebook)-'
    'ээр'  # ээр
    ' '
    'хэрэгжүүлж,'  # хэрэгжүүлж,
    ' 2025--2026 '
    'оны'  # оны
    '\n'
)

correct_line_d = (
    'нийтлэлүүдийг'  # нийтлэлүүдийг
    ' stratified sampling-'
    'аар'  # аар
    ' '
    'цуглуулсан.'  # цуглуулсан.
    '\n'
)

# Replace indices 28-31 with correct lines
new_lines = lines[:28] + [correct_line_a, correct_line_b, correct_line_c, correct_line_d] + lines[32:]

with open('Chapters/Chapter4.tex', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print('\nDone. Lines before:', len(lines), 'Lines after:', len(new_lines))

# Verify
with open('Chapters/Chapter4.tex', encoding='utf-8') as f:
    lines2 = f.readlines()
print('=== Fixed lines 26-35 (0-indexed) ===')
for i in range(26, 36):
    print(f'{i}: {lines2[i][:100]}')
