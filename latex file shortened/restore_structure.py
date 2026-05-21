# -*- coding: utf-8 -*-
"""
Restore all missing sections to every shortened chapter.
Strategy: keep all existing condensed content; splice missing sections back
          in their correct positions using condensed versions of the originals.
"""
import sys, shutil
sys.stdout.reconfigure(encoding='utf-8')

ORIG  = r'C:\Users\M Tech\Desktop\diplom\latex file\Chapters'
SHORT = r'C:\Users\M Tech\Desktop\diplom\latex file shortened\Chapters'

# ─── Ch1: already restored verbatim in previous step ─────────────────────────
print("Ch1: already verbatim — skip")

# ─── Ch2: insert §2.9.2 Олон улсын + §Бүлгийн дүгнэлт ──────────────────────
with open(f'{SHORT}/Chapter2.tex', encoding='utf-8') as f:
    lines = f.readlines()

# Find insertion point: after §2.9.1 Монгол хэлний мэдрэмжийн ... ends,
# just before §2.9.3 Монгол хэлний платформуудтай
ins = next(i for i, l in enumerate(lines)
           if r'\subsection{Монгол хэлний платформуудтай' in l)

intl_block = (
    '\n'
    r'\subsection{Олон улсын хортой агуулга илрүүлэх системүүд}' + '\n'
    '\n'
    'Perspective API (Jigsaw/Google, 2016) нь олон хэлний хортой агуулга илрүүлэх\n'
    'арилжааны API бөгөөд нейрон сүлжээнд тулгуурладаг; Монгол зэрэг бага\n'
    'хэрэглэгддэг хэлүүдэд дэмжлэг хязгаарлагдмал. Kaggle-ийн Toxic Comment\n'
    'Challenge (2018) шилдэг шийдлүүд LSTM/CNN/Transformer хослол, multi-label\n'
    'ангиллыг ашигласан нь манай архитектурт лавлагаа болно.\n'
    '\n'
)

bulgiin_ch2 = (
    '\n'
    '%-------------------------------------------------------------------------------\n'
    r'\section{Бүлгийн дүгнэлт}' + '\n'
    '\n'
    'Онолын суурь болон холбогдох ажлын шинжилгээнээс: NLP/сентимент шинжилгээ\n'
    'сайн судлагдсан салбар; Transformer/BERT нь контекстийн нарийн ялгааг барих\n'
    'хамгийн дэвшилтэт хэрэгсэл; MN-BERT Монгол хэлний NLP-д одоогоор хамгийн\n'
    'өндөр үр ашигтай. Олон улсын ажлуудад хортой--бүтээлч нарийн ялгалтын ач\n'
    'холбогдол нотлогдсон ч Монгол хэлээр ийм тусгайлсан систем дутагдалтай ---\n'
    'энэхүү ажил тэр орхигдсон чиглэлийг нөхнө. Дараагийн бүлэгт өгөгдлийн\n'
    'бэлтгэл болон арга зүйг нарийвчлан тодорхойлно.\n'
)

new2 = lines[:ins] + [intl_block] + lines[ins:] + [bulgiin_ch2]
with open(f'{SHORT}/Chapter2.tex', 'w', encoding='utf-8') as f:
    f.writelines(new2)
print(f'Ch2: {len(lines)} → {len(new2)} lines  (added Олон улсын + Бүлгийн дүгнэлт)')

# ─── Ch3: restore all missing sections ───────────────────────────────────────
# Missing: Activity Diagram subsection, Загварын гиперпараметрүүд, Тэнцвэргүй
#          өгөгдлийн шийдэл, Gradio хэрэглэгчийн интерфейс,
#          Ангиллын үзүүлэлтүүд, Бүлгийн дүгнэлт

with open(f'{ORIG}/Chapter3.tex', encoding='utf-8') as f:
    orig3 = f.readlines()
with open(f'{SHORT}/Chapter3.tex', encoding='utf-8') as f:
    s3 = f.readlines()

print(f'Ch3 shortened current: {len(s3)} lines')

# ── 3a: Activity Diagram (condensed from orig lines 86-129) ──────────────────
activity_block = (
    '\n'
    r'\subsection{Үйл ажиллагааны диаграмм (Activity Diagram)}' + '\n'
    r'\label{sec:activity-diagram}' + '\n'
    '\n'
    'Хэрэглэгчийн нийтлэл оруулахаас эхлэн ангиллын үр дүн харуулах хүртэлх\n'
    'үйл явцыг Activity Diagram-аар илэрхийлэв. Системийн урсгал:\n'
    r'\textit{Оролт} $\to$ \textit{Preprocessing} $\to$ \textit{MN-BERT инференс}' + '\n'
    r'$\to$ \textit{Flat ангилал} $\to$ \textit{Хэрэглэгчид үр дүн}.' + '\n'
    'Хэрэглэгч нийтлэл буруу уншигдвал дахин оруулах боломжтой;\n'
    'систем амжилттай ажиллах тохиолдолд 4 ангиллын нэгийг буцаана.\n'
    '\n'
)

# ── 3b: Гиперпараметрүүд subsection (condensed from orig lines 355-379) ──────
hyperparam_block = (
    '\n'
    r'\subsection{Загварын гиперпараметрүүд}' + '\n'
    r'\label{sec:hyperparams-detail}' + '\n'
    '\n'
    'Хоёр шатлалт загварын гиперпараметрүүдийг туршилтаар тогтоов: learning\n'
    r'rate $2\times10^{-5}$, batch size 16, epoch 3--5 (early stopping),' + '\n'
    r'max sequence length 128, AdamW optimizer (weight decay $0.01$),' + '\n'
    r'warmup steps нийт алхмын 10\%. Stage~1-д Cross-entropy,' + '\n'
    r'Stage~2-д Focal Loss ($\gamma{=}2.0$) ашиглав.' + '\n'
    'Эцсийн нэг шатлалт загварын тохиргоог Хүснэгт~\\ref{tab:hyperparams-mnbert}-аас үзнэ үү.\n'
    '\n'
)

# ── 3c: Тэнцвэргүй өгөгдлийн шийдэл section (condensed from orig lines 428-447)
imbalance_block = (
    '\n'
    '%-------------------------------------------------------------------------------\n'
    r'\section{Тэнцвэргүй өгөгдлийн шийдэл}' + '\n'
    r'\label{sec:imbalance}' + '\n'
    '\n'
    'Датасетийн ангиллын тэнцвэргүй байдлыг хоёр стратегиар шийдвэрлэв.\n'
    'Суурь загварт SMOTE-оор цөөн ангиллыг нийлэгжүүлж тэнцвэржүүлэв.\n'
    r'Эцсийн нэг шатлалт MN-BERT загварт Focal Loss ($\gamma{=}2.0$),' + '\n'
    r'ангиллын жин $\alpha$, WeightedRandomSampler-ийн хослолыг ашиглав.' + '\n'
    'Энэхүү хослол хортой/бүтээлч ялгалтын цөөн ангиллын recall-ийг\n'
    r'мэдэгдэхүйц сайжруулснаар Focal Loss-ын онолын давуу тал батлагдав.' + '\n'
    '\n'
)

# ── 3d: Gradio section (condensed from orig lines 448-575 = 127 lines → ~35)
gradio_block = (
    '\n'
    '%-------------------------------------------------------------------------------\n'
    r'\section{Gradio хэрэглэгчийн интерфейс}' + '\n'
    r'\label{sec:gradio}' + '\n'
    '\n'
    r'Gradio framework ашиглан хэрэглэгчийн интерфейс бүрдүүлэв. Хэрэглэгч' + '\n'
    r'текст оруулахад систем preprocessing хийж, нарийн тохируулсан MN-BERT' + '\n'
    r'загварт дамжуулан Эерэг/Саармаг/Хортой сөрөг/Бүтээлч шүүмжлэл гэсэн' + '\n'
    r'4 ангиллын нэгийг итгэлцлийн оноотой хамт буцаана.' + '\n'
    '\n'
    r'Инференсийн урсгал: $\text{Текст} \to \text{Preprocessing} \to' + '\n'
    r'\text{MN-BERT токенчлол} \to \text{Flat ангилал} \to \text{Хэрэглэгч}$.' + '\n'
    r'Flat загвар ($83.26\%$) ба Two-stage ($78.61\%$) хоёулаа нэгдсэн' + '\n'
    r'интерфейст байдаг бөгөөд хэрэглэгч сонгох боломжтой.' + '\n'
    '\n'
)

# ── 3e: Ангиллын үзүүлэлтүүд subsection (condensed from orig lines 580-595) ──
metrics_block = (
    '\n'
    r'\subsection{Ангиллын үзүүлэлтүүд}' + '\n'
    r'\label{sec:eval-metrics}' + '\n'
    '\n'
    r'Үнэлгээнд Accuracy, macro-Precision, macro-Recall, macro-F$_1$' + '\n'
    r'(\textit{macro}: ангилал тус бүрийн жинлэгдсэн дундаж) болон' + '\n'
    r'Confusion Matrix ашиглав. Тэнцвэргүй өгөгдөлд macro F$_1$ нь' + '\n'
    r'accuracy-аас илүү мэдрэмжтэй хэмжүүр тул эцсийн шалгуур болно.' + '\n'
    '\n'
)

# ── 3f: Бүлгийн дүгнэлт ──────────────────────────────────────────────────────
bulgiin_ch3 = (
    '\n'
    '%-------------------------------------------------------------------------------\n'
    r'\section{Бүлгийн дүгнэлт}' + '\n'
    '\n'
    'Энэ бүлэгт хоёр шатлалт ангиллын архитектур, MN-BERT нарийн тохируулгын\n'
    'стратеги, суурь загваруудын тохиргоо, тэнцвэргүй өгөгдлийн шийдэл,\n'
    'Gradio интерфейс болон үнэлгээний арга зүйг тодорхойлсон. Дараагийн\n'
    'бүлэгт эдгээр загваруудыг бодитоор сургаж, туршилтын үр дүнг танилцуулна.\n'
)

# Find insertion points in shortened Ch3
# 3a: after Хэрэглээний тохиолдлын диаграмм subsection ends, before Давхаргын архитектур
ins_act = next(i for i, l in enumerate(s3)
               if r'\subsection{Давхаргын архитектур}' in l)

# 3b: after Нарийн тохируулга стратеги subsection, before Суурь загварууд section
ins_hyp = next(i for i, l in enumerate(s3)
               if r'\section{Суурь загварууд}' in l)

# 3c, 3d: after Суурь загварууд section ends (before Үнэлгээний арга зүй section)
ins_imb = next(i for i, l in enumerate(s3)
               if r'\section{Үнэлгээний арга зүй}' in l)

# 3e: after Харьцуулалтын арга зүй subsection, before Технологийн стек section
ins_met = next(i for i, l in enumerate(s3)
               if r'\section{Технологийн стек}' in l)

# Build new Ch3 (splice in order, adjusting indices)
new3 = (
    s3[:ins_act]
    + [activity_block]
    + s3[ins_act:ins_hyp]
    + [hyperparam_block]
    + s3[ins_hyp:ins_imb]
    + [imbalance_block, gradio_block]
    + s3[ins_imb:ins_met]
    + [metrics_block]
    + s3[ins_met:]
    + [bulgiin_ch3]
)
with open(f'{SHORT}/Chapter3.tex', 'w', encoding='utf-8') as f:
    f.writelines(new3)
print(f'Ch3: {len(s3)} → {len(new3)} lines  (restored 6 missing blocks)')

# ─── Ch4: add only Бүлгийн дүгнэлт ──────────────────────────────────────────
with open(f'{ORIG}/Chapter4.tex', encoding='utf-8') as f:
    orig4 = f.readlines()
with open(f'{SHORT}/Chapter4.tex', encoding='utf-8') as f:
    s4 = f.readlines()

# Бүлгийн дүгнэлт content from original (lines 520-527)
bulgiin_ch4 = ''.join(orig4[519:])  # from \section{Бүлгийн дүгнэлт} to end
s4_new = s4 + ['\n', '%-------------------------------------------------------------------------------\n'] + [bulgiin_ch4]
with open(f'{SHORT}/Chapter4.tex', 'w', encoding='utf-8') as f:
    f.writelines(s4_new)
print(f'Ch4: {len(s4)} → {len(s4_new)} lines  (added Бүлгийн дүгнэлт)')

# ─── Ch5: add only Бүлгийн дүгнэлт ──────────────────────────────────────────
with open(f'{ORIG}/Chapter5.tex', encoding='utf-8') as f:
    orig5 = f.readlines()
with open(f'{SHORT}/Chapter5.tex', encoding='utf-8') as f:
    s5 = f.readlines()

# Find Бүлгийн дүгнэлт in original (line 465)
idx5 = next(i for i, l in enumerate(orig5) if r'\section{Бүлгийн дүгнэлт}' in l)
bulgiin_ch5 = ''.join(orig5[idx5-1:])  # include blank line before section
s5_new = s5 + [bulgiin_ch5]
with open(f'{SHORT}/Chapter5.tex', 'w', encoding='utf-8') as f:
    f.writelines(s5_new)
print(f'Ch5: {len(s5)} → {len(s5_new)} lines  (added Бүлгийн дүгнэлт)')

# ─── Ch6: add Бүлгийн дүгнэлт ────────────────────────────────────────────────
with open(f'{ORIG}/Chapter6.tex', encoding='utf-8') as f:
    orig6 = f.readlines()
with open(f'{SHORT}/Chapter6.tex', encoding='utf-8') as f:
    s6 = f.readlines()

idx6 = next(i for i, l in enumerate(orig6) if r'\section{Бүлгийн дүгнэлт}' in l)
bulgiin_ch6 = ''.join(orig6[idx6-1:])
s6_new = s6 + [bulgiin_ch6]
with open(f'{SHORT}/Chapter6.tex', 'w', encoding='utf-8') as f:
    f.writelines(s6_new)
print(f'Ch6: {len(s6)} → {len(s6_new)} lines  (added Бүлгийн дүгнэлт)')

# ─── Ch7: restore 3 bias subsections + Бүлгийн дүгнэлт ──────────────────────
with open(f'{ORIG}/Chapter7.tex', encoding='utf-8') as f:
    orig7 = f.readlines()
with open(f'{SHORT}/Chapter7.tex', encoding='utf-8') as f:
    s7 = f.readlines()

# From original: bias subsections start at line 92
# § Ангиллын түвшний bias (92-106), Демографийн bias (107-123),
# Бүтээлч шүүмжлэлийг буруу ангилах эрсдэл (124-141)
# Condensed versions:
bias_block = (
    '\n'
    r'\subsection{Ангиллын түвшний bias}' + '\n'
    r'\label{sec:class-bias}' + '\n'
    '\n'
    'Stage~1 загвар Саармаг ангиллыг хамгийн олон дээжтэй учраас\n'
    'overfit болох, Stage~2 загвар Бүтээлч шүүмжлэлийн цөөн дээжийг\n'
    'дутуу таних эрсдэл бий. SMOTE болон Focal Loss-ын хослол энэ\n'
    'тэнцвэргүй байдлыг зохицуулж, minority ангиллын recall-ийг нэмэгдүүлэв.\n'
    '\n'
    r'\subsection{Демографийн bias}' + '\n'
    r'\label{sec:demographic-bias}' + '\n'
    '\n'
    'Найман эх сурвалжийн датасет нь насны бүлэг, газар нутаг, нийгмийн\n'
    'давхаргаар жигд бус тул загвар тодорхой хэрэглэгчийн бүлэгт шудрага\n'
    'бус байж болно. Ирээдүйд олон төрлийн нийгмийн хэсгийг тэнцвэртэй\n'
    'төлөөлсөн өгөгдөл цуглуулах шаардлагатай.\n'
    '\n'
    r'\subsection{Бүтээлч шүүмжлэлийг Хортой сөрөг гэж буруу ангилах эрсдэл}' + '\n'
    r'\label{sec:false-positive-risk}' + '\n'
    '\n'
    'Хамгийн нухацтай алдааны төрөл: хуулийн хүрээнд хамгаалагдсан бүтээлч\n'
    'шүүмжлэлийг хортой гэж буруу ангилах нь хэрэглэгчийн эрхийг зөрчинэ.\n'
    'Confusion matrix-ийн off-diagonal утгуудыг хянаж, алдаатай ангилалтыг\n'
    'хүний хяналтаар шүүх human-in-the-loop тогтолцоо шаардлагатай.\n'
    '\n'
)

# Find insertion point: after §7.3 Bias ба тэгш байдлын шинжилгээ header
ins7 = next(i for i, l in enumerate(s7)
            if r'\section{Bias ба тэгш байдлын шинжилгээ}' in l)
# Insert bias subsections right after that section header + blank line
ins7_after = ins7 + 2  # skip section header + label/blank

# Бүлгийн дүгнэлт from original (line 250)
idx7 = next(i for i, l in enumerate(orig7) if r'\section{Бүлгийн дүгнэлт}' in l)
bulgiin_ch7 = ''.join(orig7[idx7-1:])

new7 = (
    s7[:ins7_after]
    + [bias_block]
    + s7[ins7_after:]
    + [bulgiin_ch7]
)
with open(f'{SHORT}/Chapter7.tex', 'w', encoding='utf-8') as f:
    f.writelines(new7)
print(f'Ch7: {len(s7)} → {len(new7)} lines  (restored bias subsections + Бүлгийн дүгнэлт)')

# ─── Ch8: add Бүлгийн дүгнэлт ────────────────────────────────────────────────
with open(f'{ORIG}/Chapter8.tex', encoding='utf-8') as f:
    orig8 = f.readlines()
with open(f'{SHORT}/Chapter8.tex', encoding='utf-8') as f:
    s8 = f.readlines()

idx8 = next(i for i, l in enumerate(orig8) if r'\section{Бүлгийн дүгнэлт}' in l)
bulgiin_ch8 = ''.join(orig8[idx8-1:])
s8_new = s8 + [bulgiin_ch8]
with open(f'{SHORT}/Chapter8.tex', 'w', encoding='utf-8') as f:
    f.writelines(s8_new)
print(f'Ch8: {len(s8)} → {len(s8_new)} lines  (added Бүлгийн дүгнэлт)')

print('\nAll done.')
