# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('Chapters/Chapter8.tex', encoding='utf-8') as f:
    lines = f.readlines()

print('Lines before:', len(lines))

# §8.3 content: idx 76-107 (32 lines, 6-item enumerate + intro)
# Condense to 4-item enumerate (drop items 4 & 6)
limits_new = (
    '\n'
    'Судалгааны хязгаарлалтыг илэн далангүй баримтжуулна:\n'
    '\n'
    '\\begin{enumerate}\n'
    '  \\item \\textbf{Өгөгдлийн цар хүрээ:} 10,000 дээжтэй corpus нь\n'
    '    тусгай нэр томьёоны (эрүүл мэнд, санхүү, хууль) хамралтыг бүрэн\n'
    '    илэрхийлэхгүй.\n'
    '  \\item \\textbf{Платформын гажиг:} Найман эх сурвалж ($\\approx$91\\%\n'
    '    мэдээний сайт) нь TikTok, YouTube, Twitter/X зэрэг платформ ороогүй\n'
    '    тул загварын ерөнхийлөх чадварт сөрөг нөлөөлж болзошгүй.\n'
    '  \\item \\textbf{Ёж, контекст хамааралтай алдаа:} MN-BERT нь ёжтой\n'
    '    илэрхийлэл, давхар утгатай үг, контекстээс хамаарсан аясыг\n'
    '    гадаргуун шинжид найдаж ангилдаг\n'
    '    (бүлэг~\\ref{Chapter6}-ын алдааны шинжилгээ).\n'
    '  \\item \\textbf{Аннотацийн үлдэгдэл алдаа ба үнэлгээний дисперс:}\n'
    '    Шошгуудыг олон үе шаттай сайжруулсан боловч аннотацийн үлдэгдэл\n'
    '    алдаа бүрэн арилаагүй; үр дүнг нэг random seed ($42$), нэг\n'
    '    hold-out хуваалтаар тайлагнасан тул k-fold давтан туршилт зүйтэй.\n'
    '\\end{enumerate}\n'
    '\n'
)

# §8.4 content: idx 110-143 (34 lines, 6-item enumerate)
# Condense to 4 items (merge 3+4, drop 6)
future_new = (
    '\n'
    '\\begin{enumerate}\n'
    '  \\item \\textbf{Өгөгдлийг өргөжүүлэх:} TikTok, YouTube, Twitter/X\n'
    '    болон online форумаас нэмэлт өгөгдөл цуглуулж, ирээдүйд\n'
    '    $\\geq$50,000 шошготой нэгдсэн Монгол хэлний toxicity corpus\n'
    '    нийтийн эзэмшилд гаргах.\n'
    '  \\item \\textbf{Илүү том загвар туршилт:} mBERT, XLM-R, GPT-based\n'
    '    causal LM-ийг few-shot/instruction нарийн тохируулга аргаар ашиглаж\n'
    '    гүйцэтгэлийн дээд хязгаарыг хэмжих.\n'
    '  \\item \\textbf{Ёж илрүүлэлт ба explainability:} Sarcasm detection-ийг\n'
    '    multi-task learning-оор нэгтгэх; SHAP/attention визуализацаар\n'
    '    хэрэглэгчдэд таамаглалын үндэслэлийг тайлбарлах.\n'
    '  \\item \\textbf{End-to-End fine-tune ба загварын хөнгөвчлөл:}\n'
    '    Хэсэг~\\ref{sec:transfer-init}-ийн туршилтад үндэслэн Stage~1 ба\n'
    '    Stage~2-ыг хамтад нь end-to-end fine-tune хийх; INT8 quantization/\n'
    '    knowledge distillation-оор инференсийн хурдыг нэмэгдүүлэх.\n'
    '\\end{enumerate}\n'
    '\n'
)

# Build
new_lines = (
    lines[:76]        # §8.3 header (idx 75) + blank (76 exclusive = up to idx 75)
    + [limits_new]    # condensed limitations
    + [lines[108]]    # %--- comment separator before §8.4 (idx 108)
    + [lines[109]]    # §8.4 section header
    + [future_new]    # condensed future work
    + lines[144:]     # §8.5 Social impact onwards
)

with open('Chapters/Chapter8.tex', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

with open('Chapters/Chapter8.tex', encoding='utf-8') as f:
    lines2 = f.readlines()
print('Lines after:', len(lines2))
for i, l in enumerate(lines2):
    s = l.strip()
    if s.startswith('\\section') or s.startswith('\\chapter'):
        print(f'  idx {i}: {s[:60]}')
