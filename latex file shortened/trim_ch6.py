# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('Chapters/Chapter6.tex', encoding='utf-8') as f:
    lines = f.readlines()

print('Lines before:', len(lines))

# ── Block 1: Stage 1 discussion  idx 80-95 (16 lines) → 6 lines ─────────────
stage1_discuss = (
    'MN-BERT нь бүх метрикээр суурь загваруудаас давуу гарсан нь Монгол хэлний\n'
    'корпус дээр pre-train хийгдсэн тул контекст мэдрэмжтэй векторжуулалт,\n'
    'SentencePiece subword tokenizer нь agglutinative дагавруудыг илүү сайн\n'
    'барьдагт тулгуурладаг. Саармаг ангилал хамгийн олон жишээтэй тул recall\n'
    'өндөр; Эерэг нь Саармаг-тай семантик давхцаж F1-д нөлөөлнө.\n'
    '\n'
)

# ── Block 2: Stage 2 discussion  idx 164-170 (7 lines) → 4 lines ────────────
stage2_discuss = (
    'Хоёр ангиллын хил нь гадаргуун лексикт ойролцоо боловч зорилгоор ялгаатай\n'
    '--- Бүтээлч шүүмжлэл нь шийдэл санал болгох, Хортой сөрөг нь хувь хүнд\n'
    'чиглэсэн доромжлол; контекст мэдрэмжтэй MN-BERT TF-IDF-ийн статистик\n'
    'аргаас илүү сайн таньж чадна.\n'
    '\n'
)

# ── Block 3: Transfer-init text before table  idx 174-189 (16 lines) → 5 lines
transfer_before = (
    '\n'
    'STILTs санааг ашиглан~\\cite{phang2018} Stage~1-ийн нарийн тохируулгаар олсон\n'
    'энкодерийн жингүүдийг Stage~2-ын эхлэлийн цэг болгон ашиглавал гүйцэтгэл\n'
    'сайжрах эсэхийг туршилтаар шалгав (яг ижил гиперпараметр, зөвхөн эхлэлийн жин\n'
    'ялгаатай).\n'
)

# ── Block 4: Transfer-init text after table  idx 208-233 (26 lines) → 7 lines
transfer_after = (
    '\n'
    'Хортой сөрөгийн precision $0.8418\\to0.9346$ өсч, recall $0.8956\\to0.8182$\n'
    'буурсан --- transfer-init загвар хортой агуулга ялгахдаа илүү хатуу болсон нь\n'
    'content moderation-д false positive бууруулах практик шаардлагад тохиромжтой.\n'
    'Stage~1-ийн жингүүдийг шилжүүлэн ачаалах нь Stage~2-ын гүйцэтгэлд жижиг\n'
    'боловч системтэй ($\\Delta$Macro F1 $=+0.0074$) сайжралт авчирсан (shared\n'
    'domain knowledge); end-to-end fine-tune стратегийг цаашдын чиглэлд тодорхойлно.\n'
    '\n'
)

# ── Block 5: "Why MN-BERT" enumerate  idx 507-529 (23 lines) → 9 lines ───────
why_bert = (
    'MN-BERT нь суурь загваруудаас илүү гүйцэтгэл үзүүлсэн гол шалтгаан:\n'
    '(1)~SentencePiece subword tokenization нь Монгол хэлний agglutinative\n'
    'бүтцийг статистик TF-IDF-ийн оронд ерөнхийлнэ; (2)~self-attention нь\n'
    'давхар утга, үгүйсгэлийг таньдаг; (3)~Монгол хэлний pre-train нь\n'
    'domain-specific нарийн тохируулгын чанартай суурь болно. Гэсэн хэдий ч\n'
    'GPU шаарддаг тул үйлдвэрлэлийн орчинд distillation/quantization авч үзэх\n'
    'нь зүйтэй (цаашдын чиглэлд тодорхойлсон).\n'
    '\n'
)

# Build new file
new_lines = (
    lines[:80]           # up to Stage 1 discussion
    + [stage1_discuss]   # condensed Stage 1 discuss
    + lines[96:164]      # §6.2 header through Stage 2 discussion
    + [stage2_discuss]   # condensed Stage 2 discuss
    + lines[171:174]     # %--- comment + \subsection + \label
    + [transfer_before]  # condensed transfer before table
    + lines[190:208]     # the table (idx 190-207)
    + [transfer_after]   # condensed transfer after table
    + lines[234:507]     # §6.3 through Why BERT
    + [why_bert]         # condensed Why BERT
    + lines[530:]        # rest: final scoreboard + hypothesis
)

with open('Chapters/Chapter6.tex', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

# Verify
with open('Chapters/Chapter6.tex', encoding='utf-8') as f:
    lines2 = f.readlines()
print('Lines after:', len(lines2))
for i, l in enumerate(lines2):
    s = l.strip()
    if s.startswith('\\section') or s.startswith('\\subsection'):
        print(f'  idx {i}: {s[:60]}')
