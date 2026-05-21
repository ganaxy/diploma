# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('Chapters/Chapter7.tex', encoding='utf-8') as f:
    lines = f.readlines()

print('Lines before:', len(lines))

# Confirmed indices (0-based):
# §7.1 header=13, content=14..48, §7.2 header=49
# §7.2 header=49, content=50..85, §7.3 header=86
# §7.3 header=86, content=87..140, §7.4 header=141
# §7.4 header=141, content=142..158, §7.5 header=159   → KEEP §7.4
# §7.5 header=159, content=160..196, §7.6 header=197
# §7.6 header=197, content=198..end  → KEEP §7.6 (AI disclosure)

sec7_1_body = (
    '\n'
    'Судалгаанд найман нийтэд нээлттэй эх сурвалжаас цуглуулсан нийтийн\n'
    'коммент ашигласан тул хувийн нэвтрэх зөвшөөрөл шаардсан агуулга орохгүй.\n'
    'Цуглуулмагч хэрэглэгчийн нэр, профайл, холбоос зэрэг PII-г автомат\n'
    'шүүлтүүрээр устгаж зөвхөн анонимчлагдсан текст хадгалсан ---\n'
    'Монгол Улсын ``Хувь хүний нууцын тухай хууль\'\' (2021)-ийн шаардлагад\n'
    'нийцүүлсэн; GDPR-ын зарчмаас санаа авч арилжааны түгээлт, гуравдагч\n'
    'этгээдэд шилжүүлэхгүйгээр зөвхөн судалгааны зорилгоор ашиглав.\n'
    'Өгөгдөл нь public, de-identified тул IRB хянан шийдвэрлэх шаардлагаас\n'
    'ангид гэж үнэлэв (бүлэг~\\ref{Chapter4},\n'
    'Хүснэгт~\\ref{tab:data_full_corpus}).\n'
    '\n'
)

sec7_2_body = (
    '\n'
    'Өгөгдлийг 4 ангилалд шошголохдоо Монгол хэлний мэдрэмжтэй, NLP-д\n'
    'сургагдсан $n=3$ annotator ашиглав. Эхлэлийн 200 жишээг хамтад нь шошголж\n'
    'удирдамжийг боловсруулсан; гол багцын $\\geq\\!20\\%$-г хоёр annotator\n'
    'тусад нь давхар шошгологдон Cohen\'\'s Kappa болон Fleiss\' Kappa-аар нийцлийг\n'
    'хэмжсэн --- эцсийн утга $\\kappa\\!=\\!0.72$ (\\textit{substantial} нийцэл,\n'
    '0.61--0.80 хүрээ), датасетын найдвартай байдлыг баталсан. Зөрүүтэй\n'
    'шошгуудыг гуравдагч annotator шийдвэрлэж, шийдвэрлэгдэхгүй тохиолдлуудыг\n'
    '``хилийн тохиолдол\'\' болгон тэмдэглэсэн. Хортой сөрөг ба Бүтээлч\n'
    'шүүмжлэлийн хооронд нийцэл Эерэг/Саармаг-ийнхаас бага байсан нь\n'
    'семантик ойролцоо байдлыг илтгэж байна.\n'
    '\n'
)

sec7_3_body = (
    '\n'
    'Ангиллын тэнцвэргүй байдалд SMOTE (суурь загварт) болон Focal Loss\n'
    '($\\gamma{=}2.0$) + WeightedRandomSampler (нэг шатлалт MN-BERT-д) ашиглаж,\n'
    'macro F1-ийг үнэлгээний гол метрик болгосон (\\textit{Ангиллын түвшний bias}).\n'
    'Facebook-ийн өгөгдөлд залуу, хотын хэрэглэгчид давамгайлдаг тул\n'
    'news.mn-ийн өгөгдөлтэй хослуулан хэлний олон талт байдлыг нэмэгдүүлсэн;\n'
    'эцсийн системийг content moderation-д ашиглахаас өмнө нэмэлт fairness audit\n'
    'хийх шаардлагатайг баримтжуулсан (\\textit{Демографийн bias}).\n'
    'Бүтээлч шүүмжлэлийг Хортой сөрөг гэж буруу ангилах false positive эрсдэлийг\n'
    'бууруулахаар Stage~2-ын precision-ыг хянаж, human-in-the-loop архитектурыг\n'
    'нэмэлт хамгаалалт болгон санал болгосон (бүлэг~\\ref{Chapter6}-ын алдааны\n'
    'шинжилгээтэй нийцнэ).\n'
    '\n'
)

sec7_5_body = (
    '\n'
    'Агуулгын зохицуулалтын NLP систем нь эерэгтэй (хортой контент шүүх) болон\n'
    'сөрөгтэй (хууль ёсны үзэл бодлыг дарангуйлах) хоёр талт шинж чанартай.\n'
    'Монгол орны контекстэд улс төрийн шүүмжлэлийг буруу ангилах болон цөөнх\n'
    'бүлгийн хэлэнд хандах bias онцгой анхаарал татна. Эрсдэлийг бууруулахаар:\n'
    'систем ``зөвлөгч\'\' үүрэгтэй байж эцсийн шийдвэрийг хүн гаргах\n'
    '(human-in-the-loop); false positive-ыг багасгахаар loss function тохируулах;\n'
    'attention/SHAP-ийн харагдуулалтаар explainability хангах; улирал бүр\n'
    'санамсаргүй түүвэр дээр human audit хийхийг зөвлөмж болгосон.\n'
    '\n'
)

# Build new file:
# lines[0..12]   = chapter header + intro (up to §7.1 header exclusive → indices 0-12)
# lines[13]      = §7.1 section header
# sec7_1_body    = condensed §7.1 content
# lines[49]      = §7.2 section header
# sec7_2_body    = condensed §7.2 content
# lines[86]      = §7.3 section header
# sec7_3_body    = condensed §7.3 content
# lines[141:160] = §7.4 header + content + §7.5 header
# sec7_5_body    = condensed §7.5 content
# lines[197:]    = §7.6 header + AI disclosure (keep intact)

new_lines = (
    lines[:14]           # indices 0-13: chapter header, intro, §7.1 section header
    + [sec7_1_body]
    + [lines[49]]        # §7.2 section header
    + [sec7_2_body]
    + [lines[86]]        # §7.3 section header
    + [sec7_3_body]
    + lines[141:160]     # §7.4 header + content (indices 141-159) + §7.5 header (159)
    + [sec7_5_body]
    + lines[197:]        # §7.6 AI disclosure
)

with open('Chapters/Chapter7.tex', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print('Lines after:', len(new_lines))

# Quick verification
with open('Chapters/Chapter7.tex', encoding='utf-8') as f:
    lines2 = f.readlines()
for i, l in enumerate(lines2):
    s = l.strip()
    if s.startswith('\\section') or s.startswith('\\subsection') or s.startswith('\\chapter'):
        print(f'  idx {i}: {s[:70]}')
