"""
Generate Diploma Defense Q&A Preparation PDF
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
import os

# ── Fonts ──────────────────────────────────────────────────────────────────────
# Try to register a system font that supports Mongolian / Cyrillic
FONT_PATHS = [
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
]
BOLD_PATHS = [
    "/Library/Fonts/Arial Bold.ttf",
    "/Library/Fonts/Arial-BoldMT.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]

def try_register(name, paths):
    for p in paths:
        if os.path.exists(p):
            try:
                pdfmetrics.registerFont(TTFont(name, p))
                return True
            except Exception:
                pass
    return False

has_unicode = try_register("Unicode", FONT_PATHS)
has_bold    = try_register("UnicodeBold", BOLD_PATHS)

BASE_FONT = "Unicode" if has_unicode else "Helvetica"
BOLD_FONT = "UnicodeBold" if has_bold else "Helvetica-Bold"

# ── Colors ─────────────────────────────────────────────────────────────────────
NAVY   = colors.HexColor("#0D1B3E")
TEAL   = colors.HexColor("#008B8B")
ORANGE = colors.HexColor("#E86A27")
LIGHT  = colors.HexColor("#F5F7FA")
GRAY   = colors.HexColor("#6B7280")
WHITE  = colors.white
BG_Q   = colors.HexColor("#EFF6FF")
BG_A   = colors.HexColor("#F0FDF4")
BG_N   = colors.HexColor("#FFFBEB")

# ── Styles ─────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "Title", fontName=BOLD_FONT, fontSize=20, textColor=WHITE,
    alignment=TA_CENTER, leading=28, spaceAfter=4
)
subtitle_style = ParagraphStyle(
    "Subtitle", fontName=BASE_FONT, fontSize=11, textColor=colors.HexColor("#CBD5E1"),
    alignment=TA_CENTER, leading=16
)
section_style = ParagraphStyle(
    "Section", fontName=BOLD_FONT, fontSize=13, textColor=WHITE,
    alignment=TA_LEFT, leading=18, leftIndent=8
)
q_label = ParagraphStyle(
    "QLabel", fontName=BOLD_FONT, fontSize=9, textColor=TEAL,
    alignment=TA_LEFT, leading=13, spaceAfter=2
)
q_text = ParagraphStyle(
    "QText", fontName=BOLD_FONT, fontSize=11, textColor=NAVY,
    alignment=TA_LEFT, leading=16, spaceAfter=2
)
a_label = ParagraphStyle(
    "ALabel", fontName=BOLD_FONT, fontSize=9, textColor=colors.HexColor("#166534"),
    alignment=TA_LEFT, leading=13, spaceAfter=2
)
a_text = ParagraphStyle(
    "AText", fontName=BASE_FONT, fontSize=10, textColor=colors.HexColor("#1E293B"),
    alignment=TA_JUSTIFY, leading=15, spaceAfter=2
)
n_label = ParagraphStyle(
    "NLabel", fontName=BOLD_FONT, fontSize=9, textColor=ORANGE,
    alignment=TA_LEFT, leading=13, spaceAfter=2
)
n_text = ParagraphStyle(
    "NText", fontName=BASE_FONT, fontSize=9.5, textColor=colors.HexColor("#44403C"),
    alignment=TA_JUSTIFY, leading=14, spaceAfter=0
)
num_style = ParagraphStyle(
    "Num", fontName=BOLD_FONT, fontSize=14, textColor=TEAL,
    alignment=TA_CENTER, leading=18
)
tag_style = ParagraphStyle(
    "Tag", fontName=BOLD_FONT, fontSize=8, textColor=WHITE,
    alignment=TA_CENTER, leading=12
)

# ── Q&A Data ───────────────────────────────────────────────────────────────────
SECTIONS = [
    {
        "title": "1. Судалгааны сэдэв ба зорилго",
        "color": NAVY,
        "qas": [
            {
                "q": "Яагаад хяналтын камерын бичлэгт зодооны илрүүлэлт хийх сэдэв сонгосон бэ?",
                "a": "Монголд олон нийтийн газруудад хүчирхийллийн тохиолдол нэмэгдэж байгаа ба одоогийн систем нь хүнд шалгалт хийдэг. Автомат илрүүлэлтийн систем нь бодит цагийн аюулгүй байдлыг хангах боломж олгоно. Мөн VLM + Knowledge Distillation хослол нь энэ домэйнд туршигдаагүй байсан шинэлэг талтай.",
                "n": "Практик хэрэглээний тал болон шинэлэг технологийн хосолдлыг онцол. Монгол болон дэлхийн статистик дурдаж болно."
            },
            {
                "q": "Таны судалгааны гол шинэлэг зүйл (novelty) юу вэ?",
                "a": "Гурван шинэлэг зүйл: (1) VLM-д суурилсан зодооны илрүүлэлт + текстийн тайлбар хосолсон олон даалгаврын систем; (2) 30B параметртэй teacher-ээс 7B student-ийг Knowledge Distillation-аар дамжуулан сургасан; (3) Q&A өгөгдлийн сан бий болгож (~9,040 хос) LoRA fine-tuning хийснээр inference 4 дахин хурдасгасан.",
                "n": "Teacher-Student KD + VLM + violence detection хосолдол нь литературт маш цөөн байдаг тул энэ нь чухал novelty."
            },
        ]
    },
    {
        "title": "2. Загварын сонголт ба техникийн шийдэл",
        "color": NAVY,
        "qas": [
            {
                "q": "Яагаад Qwen гэр бүлийн загвар сонгосон бэ? GPT-4V эсвэл LLaVA хэрэглэж болохгүй байсан уу?",
                "a": "Qwen2.5-VL-7B нь: (1) Apache 2.0 лиценз — коммерцийн болон судалгааны хэрэглээнд чөлөөтэй; (2) 7B параметр — А100 80GB дээр QLoRA-аар сургах боломжтой; (3) Олон хэл + Монгол Кирилл дэмжлэг; (4) VideoChat, temporal reasoning чадварт мэргэшсэн. GPT-4V нь API хязгаарлалт, өндөр зардалтай. LLaVA нь видеог бага боловсруулдаг.",
                "n": "Лиценз + нөөц + чадвар — гурван шалгуурыг дурдах нь итгэл үнэмшилтэй хариулт болно."
            },
            {
                "q": "Knowledge Distillation яагаад сонгосон бэ? Шууд 30B загвар ашиглаж болохгүй байсан уу?",
                "a": "Qwen3-VL-30B-A3B нь inference-д 62.2GB VRAM шаарддаг — дунд зэргийн серверт хүрэлцэхгүй. KD-ээр: student загвар 6.5GB VRAM-д ажиллах болж, inference 4× хурдасч (~200ms), 91% accuracy хадгалагдсан (teacher-ийн ~95%-тай харьцуулбал ердөө 4% алдагдал). Энэ нь production deployment-д шийдвэрлэх ач холбогдолтой.",
                "n": "VRAM: 62.2GB → 6.5GB, хурд: 4×, accuracy алдагдал: ~4% — тоонуудыг цээжлэ."
            },
            {
                "q": "LoRA гэж юу вэ, яагаад сонгосон бэ?",
                "a": "LoRA (Low-Rank Adaptation) нь загварын жинг бүхэлд нь тохируулахын оронд зарим матриц дахь бага зэргийн нэмэлт матрицуудыг сургадаг арга. r=16, α=32 тохиргоогоор ~85M параметр (нийтийн 1.2%) сурган full fine-tuning-тэй дүйцэхүйц үр дүн гарсан. NF4 4-bit quantization нэмснээр VRAM ~40% буурсан.",
                "n": "LoRA-г визуал дүрсэлж тайлбарлах: W = W₀ + BA (B, A нь бага rank матриц)."
            },
            {
                "q": "Macro F1 score яагаад ашигласан бэ? Accuracy хэрэглэж болохгүй байсан уу?",
                "a": "Манай өгөгдлийн сан тэнцвэргүй (imbalanced) — 'none' класс давамгайлж байна. Accuracy нь ийм нөхцөлд misleading: бүх дүрсийг 'none' гэж зарлавал 70%+ accuracy гарна, гэхдээ хяналт үгүй болно. Macro F1 нь бүх классийн precision/recall-ыг тэгш жингээр тооцох тул жижиг классийн гүйцэтгэлийг бодитоор харуулна.",
                "n": "Imbalanced dataset-д accuracy misleading болдог жишээ: 95% negative class байвал бүгдийг negative гэхэд 95% accuracy гарна."
            },
        ]
    },
    {
        "title": "3. Өгөгдлийн сан ба сургалт",
        "color": NAVY,
        "qas": [
            {
                "q": "Өгөгдлийн чанарыг хэрхэн хангасан бэ? Annotation хэн хийсэн бэ?",
                "a": "Дөрвөн суурь dataset ашигласан: RWF-2000, UCF-Crime, Hockey Fight (нийт 3,850 клип) — олон нийтийн, peer-reviewed датасет. Текстийн тайлбар (2,269 тайлан) нь Qwen3-VL-30B-A3B teacher загвараар автоматаар үүссэн, дараа нь гараар шалгагдсан. Q&A хосуудыг 4 төрлөөр (event_type, subjects, chronology, conclusion) бүтэцлэсэн.",
                "n": "Human validation + автомат үүсгэлт хосолдлыг онцол — нэг хүн бүгдийг annotate хийх боломжгүй гэдгийг ойлгуулна."
            },
            {
                "q": "5 epoch сургасан нь хангалттай уу? Overfitting байна уу?",
                "a": "Validation loss, training loss хоёулаа буурж нийцтэй байсан — overfitting тэмдэг ажиглагдаагүй. Cosine LR scheduler нь эцсийн шатанд learning rate-ийг аажим бууруулж, хэт тохируулахаас сэрдэг. LoRA-ын бага параметрийн тоо (1.2%) нь өөрөө regularization эффект үзүүлдэг.",
                "n": "Training/validation curve-ыг харуулж, converge болсон цэгийг заах нь сайн. Early stopping хэрэглэсэн эсэхийг дурдаж болно."
            },
            {
                "q": "Dataset нь Монголын нөхцөлтэй тохирч байна уу? Transfer хийж болох уу?",
                "a": "UCF-Crime болон RWF-2000 нь хотын камерын нөхцөлд хийгдсэн тул Монголын хотын орчинтой харьцангуй дөхсөн. Гэсэн ч гэрэлтүүлэг, камерын байршил, хүмүүсийн хувцаслалтын ялгаа байж болно. Domain adaptation нь ирээдүйн судалгааны чиглэл — Монгол хяналтын камерын өгөгдөл цуглуулж дахин fine-tune хийх боломжтой.",
                "n": "Limitation болгон хүлээн зөвшөөрч, future work-ийг хэлэх нь академик боловсорлыг харуулна."
            },
        ]
    },
    {
        "title": "4. Үр дүн ба харьцуулалт",
        "color": NAVY,
        "qas": [
            {
                "q": "Macro F1=0.904 гэдэг нь өндөр үнэлгээ мөн үү? Baseline-тай хэрхэн харьцуулах вэ?",
                "a": "Тийм — зодооны илрүүлэлтийн benchmark-д 0.90+ нь дэлхийн шилдэг үр дүнтэй харьцуулах боломжтой түвшин. Харьцуулалт: Two-Stream CNN ~78%, SlowFast ~82%, ResNet+LSTM ~80% (RWF-2000 дээр). Манай загвар эдгээрийг 12-24% давж гарсан ба нэмж текстийн тайлбар үүсгэх чадвартай — нэг загварт хоёр функц.",
                "n": "Гагцхүү тоогоор зогсохгүй, чанарын давуу талыг (тайлбар үүсгэх) онцол."
            },
            {
                "q": "Ablation study-д юу судалсан бэ?",
                "a": "Гурван хувилбарыг харьцуулсан: (1) Baseline: Qwen2.5-VL-7B without fine-tuning → F1=0.71; (2) + LoRA fine-tune (no KD) → F1=0.86; (3) + KD loss (full system) → F1=0.904. KD loss нь 0.044 (~4.4%) нэмэлт сайжруулт өгсөн — soft label-ын мэдлэг нь label smoothing эффект үзүүлж, хэцүү хязгаарын дүрсийг зөв ангилахад тусалсан.",
                "n": "Ablation нь шинжлэх ухааны нотолгоо — 'яагаад KD хэрэгтэй вэ'-г тоогоор баталдаг."
            },
            {
                "q": "~200ms inference хурд бодит цагийн системд хангалттай уу?",
                "a": "30 FPS видеод фрэйм бүр 33ms боловч зодооны илрүүлэлтийн системд бүх фрэймийг биш, sampling хийсэн видео сегментийг анализлах тул 200ms нь хангалттай. Практик дээр 1-2 секундын сегментийг боловсруулах тул real-time мэдэгдэл байгаа хэвээр боломжтой. Nginx + FastAPI үйлдвэрлэлийн орчинд 5 concurrent stream дэмжих боломжтой.",
                "n": "Video sampling strategy-г тайлбарлах: бүх фрэйм биш, keyframe extraction эсвэл fixed interval sampling."
            },
        ]
    },
    {
        "title": "5. Систем ба байрлуулалт",
        "color": NAVY,
        "qas": [
            {
                "q": "Энэхүү системийг бодит орчинд хэрэглэхийн тулд ямар тоног төхөөрөмж шаардлагатай вэ?",
                "a": "Минимум: NVIDIA RTX 3090 (24GB VRAM) — student загвар 6.5GB + buffer. Зөвлөмж: A40 эсвэл A100 GPU сервер, 64GB RAM, NVMe SSD. Олон камерын систем: NVIDIA Triton Inference Server + model sharding. GPU байхгүй бол CPU inference ч боломжтой (800ms-1.2s) — өндөр хэмжээний CCTV бус, сэрэмжлүүлгийн зориулалтаар.",
                "n": "Хэрэглэгчийн бюджет, техникийн дэд бүтцийг харгалзан tier-ийн шийдэл санал болгох нь практик сэтгэлгээ харуулна."
            },
            {
                "q": "Ёс зүйн асуудал байна уу? Хувийн нууцлалын асуудал яах вэ?",
                "a": "Тийм, чухал асуудал. Хариулт: (1) Зөвхөн зодооны хэв маяг илрүүлдэг — нүүрний таних систем биш; (2) Бичлэгийг локал боловсруулдаг — гуравдагч талд илгээхгүй; (3) GDPR болон Монголын хувийн мэдээллийн хуулийг дагаж мөрдөх; (4) Камерын байршилд тэмдэглэгээ тавих ёстой. Face blurring нэмэх нь нэмэлт хамгаалалт өгнө.",
                "n": "Багш энэ асуултыг заавал асуух тул бэлдсэн байхаа мэдэгд. GDPR болон МУ-ын хуулийн дугаарыг судлах."
            },
            {
                "q": "Системийн хязгаарлалт юу вэ?",
                "a": "Тодорхойлсон хязгаарлалтууд: (1) Монгол хяналтын камерын өгөгдөл дээр тусгайлан туршаагүй (domain gap); (2) Маш олон хүн байгаа хэт их дүүрэн орчинд performance буурч болно; (3) Бага нягтралтай (<480p) бичлэгт F1 буурдаг; (4) Night vision / thermal camera дэмжихгүй; (5) 30B teacher сургалтанд 62.2GB VRAM шаарддаг тул нөхцөл хязгаарлагдмал.",
                "n": "Limitation-ыг нуухгүй, шударгаар хэлэх нь academic maturity-г харуулна."
            },
            {
                "q": "Ирээдүйн судалгааны чиглэл юу байж болох вэ?",
                "a": "Дөрвөн чиглэл: (1) Монгол хяналтын камерын өгөгдлөөр domain fine-tuning; (2) Multi-camera temporal reasoning — нэг үйл явдлыг олон камераар хянах; (3) Edge device deployment — NVIDIA Jetson Orin дээр INT8 quantization; (4) Real-time streaming API — WebSocket болон RTSP stream шууд боловсруулах.",
                "n": "Ирээдүйн ажлыг тодорхой, хэрэгжих боломжтойгоор тайлбарла — 'мөрөөдлийн ажил' биш."
            },
        ]
    },
    {
        "title": "6. Нэмэлт техникийн асуулт",
        "color": NAVY,
        "qas": [
            {
                "q": "NF4 4-bit quantization гэж юу вэ, accuracy-д нөлөө үзүүлдэг үү?",
                "a": "NF4 (NormalFloat4) нь нормал тархалттай жинг 4-bit-ээр хадгалах оновчтой quantization формат. QLoRA paper (Dettmers 2023) дагуу NF4 нь 8-bit baseline-тай ойролцоо accuracy хадгалдаг. Манай туршилтд quantization-аас үүдэлсэн accuracy алдагдал <1% байсан ба VRAM ашиглалт ~40% буурсан. Ийнхүү 80GB A100 дээр 30B загварыг сургах боломжтой болсон.",
                "n": "QLoRA = QA (NF4 quantization) + LoRA (low-rank adaptation) хосолдол гэдгийг онцол."
            },
            {
                "q": "Knowledge Distillation-ын KL divergence loss яагаад ашигласан бэ?",
                "a": "KD training-д хоёр loss хосолсон: (1) Task loss — ground truth label-тай cross-entropy; (2) KD loss — teacher soft probability-тай KL divergence (temperature T=2). Soft label нь teacher-ийн 'мэдлэгийн бүтэц'-ийг агуулдаг: жишээлбэл 'medium violence' дүрсийг teacher 60% medium, 35% high, 5% none гэж харж болно — энэ мэдлэг student-ийн сургалтыг баяжуулна.",
                "n": "Temperature scaling (T=2) нь probability distribution-ыг зөөлрүүлж dark knowledge-ийг илүү сайн дамжуулдаг — Hinton 2015 paper-ийг дурдах."
            },
            {
                "q": "Яагаад Qwen2.5-VL-7B сонгосон, 3B эсвэл 13B биш?",
                "a": "3B нь хэт жижиг — видеог ойлгох reasoning чадвар хязгаарлагдмал, temporal dependency бага. 13B нь A100 80GB дээр QLoRA сургалт хийхэд хэцүү (batch size буурна). 7B нь: AMD MI300X, NVIDIA A100 аль алинд сургах боломжтой, 6.5GB deployment VRAM, бизнесийн сервер дээр ажиллах боломжтой, хангалттай reasoning чадвартай — golden middle.",
                "n": "Scaling law-ийн үүднээс 7B нь fine-tune + deployment efficiency-д оновчтой цэг болохыг тайлбарлана."
            },
        ]
    },
]

# ── Build PDF ──────────────────────────────────────────────────────────────────
def build_pdf(filename="defense_qa.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        leftMargin=1.8*cm,
        rightMargin=1.8*cm,
        topMargin=2*cm,
        bottomMargin=2*cm,
    )
    story = []
    W = A4[0] - 3.6*cm

    # ── Cover banner ──────────────────────────────────────────────────────────
    cover_data = [[
        Paragraph("DIPLOMA DEFENSE", title_style),
    ],[
        Paragraph("Q&amp;A PREPARATION GUIDE", title_style),
    ],[
        Paragraph(
            "Хяналтын камерын бичлэгнээс зодооны зөрчил илрүүлж, тайлбар гаргах дүн шинжилгээ хийх",
            subtitle_style
        ),
    ],[
        Paragraph("О.Бат-Эрдэнэ · B221960003 · ШУТИС МХТС · 2026", subtitle_style),
    ]]
    cover_table = Table(cover_data, colWidths=[W])
    cover_table.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), NAVY),
        ("TOPPADDING",   (0,0), (-1,0),  22),
        ("BOTTOMPADDING",(0,-1),(-1,-1), 22),
        ("TOPPADDING",   (0,1), (-1,2),  4),
        ("BOTTOMPADDING",(0,0), (-1,2),  4),
        ("LEFTPADDING",  (0,0), (-1,-1), 20),
        ("RIGHTPADDING", (0,0), (-1,-1), 20),
    ]))
    story.append(cover_table)
    story.append(Spacer(1, 0.4*cm))

    # legend
    legend_data = [[
        Paragraph("ASУУЛТ", ParagraphStyle("L", fontName=BOLD_FONT, fontSize=8,
            textColor=TEAL, alignment=TA_CENTER)),
        Paragraph("ХАРИУЛТ", ParagraphStyle("L", fontName=BOLD_FONT, fontSize=8,
            textColor=colors.HexColor("#166534"), alignment=TA_CENTER)),
        Paragraph("НЭМЭЛТ ТЭМДЭГЛЭЛ", ParagraphStyle("L", fontName=BOLD_FONT, fontSize=8,
            textColor=ORANGE, alignment=TA_CENTER)),
    ]]
    lt = Table(legend_data, colWidths=[W/3]*3)
    lt.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(0,0), BG_Q),
        ("BACKGROUND",   (1,0),(1,0), BG_A),
        ("BACKGROUND",   (2,0),(2,0), BG_N),
        ("BOX",          (0,0),(-1,-1), 0.5, GRAY),
        ("TOPPADDING",   (0,0),(-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
    ]))
    story.append(legend_data and lt)
    story.append(Spacer(1, 0.5*cm))

    q_total = sum(len(s["qas"]) for s in SECTIONS)
    story.append(Paragraph(
        f"Нийт {q_total} асуулт · {len(SECTIONS)} бүлэг",
        ParagraphStyle("Info", fontName=BASE_FONT, fontSize=9, textColor=GRAY,
                       alignment=TA_CENTER)
    ))
    story.append(Spacer(1, 0.6*cm))
    story.append(HRFlowable(width=W, thickness=1, color=TEAL))
    story.append(Spacer(1, 0.4*cm))

    # ── Sections ──────────────────────────────────────────────────────────────
    q_num = 1
    for sec in SECTIONS:
        # Section header
        sec_data = [[Paragraph(sec["title"], section_style)]]
        sec_table = Table(sec_data, colWidths=[W])
        sec_table.setStyle(TableStyle([
            ("BACKGROUND",   (0,0),(-1,-1), NAVY),
            ("TOPPADDING",   (0,0),(-1,-1), 8),
            ("BOTTOMPADDING",(0,0),(-1,-1), 8),
            ("LEFTPADDING",  (0,0),(-1,-1), 12),
        ]))
        story.append(KeepTogether([sec_table, Spacer(1, 0.3*cm)]))

        for qa in sec["qas"]:
            # Number badge + question
            q_content = [
                [
                    Paragraph(str(q_num), num_style),
                    Paragraph(qa["q"], q_text),
                ]
            ]
            q_tbl = Table(q_content, colWidths=[1*cm, W - 1.2*cm])
            q_tbl.setStyle(TableStyle([
                ("BACKGROUND",   (0,0),(-1,-1), BG_Q),
                ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
                ("TOPPADDING",   (0,0),(-1,-1), 8),
                ("BOTTOMPADDING",(0,0),(-1,-1), 8),
                ("LEFTPADDING",  (0,0),(0,0),   6),
                ("LEFTPADDING",  (1,0),(1,0),   8),
                ("RIGHTPADDING", (0,0),(-1,-1), 8),
                ("LINEABOVE",    (0,0),(-1,0),  1.5, TEAL),
            ]))

            # Answer
            a_content = [[
                Paragraph("ХАРИУЛТ:", a_label),
                Paragraph(qa["a"], a_text),
            ]]
            a_tbl = Table([[
                Paragraph("ХАРИУЛТ:  " + qa["a"], a_text)
            ]], colWidths=[W])
            a_tbl.setStyle(TableStyle([
                ("BACKGROUND",   (0,0),(-1,-1), BG_A),
                ("TOPPADDING",   (0,0),(-1,-1), 7),
                ("BOTTOMPADDING",(0,0),(-1,-1), 7),
                ("LEFTPADDING",  (0,0),(-1,-1), 12),
                ("RIGHTPADDING", (0,0),(-1,-1), 10),
                ("LINEABOVE",    (0,0),(-1,0),  0.5, colors.HexColor("#166534")),
            ]))

            # Notes
            n_tbl = Table([[
                Paragraph("ТЭМДЭГЛЭЛ:  " + qa["n"], n_text)
            ]], colWidths=[W])
            n_tbl.setStyle(TableStyle([
                ("BACKGROUND",   (0,0),(-1,-1), BG_N),
                ("TOPPADDING",   (0,0),(-1,-1), 5),
                ("BOTTOMPADDING",(0,0),(-1,-1), 7),
                ("LEFTPADDING",  (0,0),(-1,-1), 12),
                ("RIGHTPADDING", (0,0),(-1,-1), 10),
                ("LINEABOVE",    (0,0),(-1,0),  0.5, ORANGE),
            ]))

            story.append(KeepTogether([
                q_tbl,
                a_tbl,
                n_tbl,
                Spacer(1, 0.35*cm),
            ]))
            q_num += 1

        story.append(Spacer(1, 0.2*cm))

    # ── Key numbers cheat sheet ───────────────────────────────────────────────
    story.append(HRFlowable(width=W, thickness=1, color=TEAL))
    story.append(Spacer(1, 0.4*cm))

    cheat_header = Table([[
        Paragraph("ГУРВАН МИНУТЫН ТООНУУДЫН ЖАГ САРЛАЛТ", section_style)
    ]], colWidths=[W])
    cheat_header.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), TEAL),
        ("TOPPADDING",   (0,0),(-1,-1), 8),
        ("BOTTOMPADDING",(0,0),(-1,-1), 8),
        ("LEFTPADDING",  (0,0),(-1,-1), 12),
    ]))
    story.append(cheat_header)
    story.append(Spacer(1, 0.2*cm))

    numbers = [
        ["Macro F1 Score",         "0.904",          "Student загварын гол үр дүн"],
        ["Token Accuracy",         "~93%",            "Текст тайлбарын нарийвчлал"],
        ["Teacher VRAM",           "62.2 GB",         "Qwen3-VL-30B-A3B inference"],
        ["Student VRAM",           "6.5 GB",          "QLoRA-д хийсний дараа"],
        ["Inference хурд",         "~200 ms",         "Student vs Teacher 4× хурдан"],
        ["Trainable params",       "~85M (1.2%)",     "LoRA r=16, α=32"],
        ["Сургалтын цаг",          "~40 цаг",         "5 epoch, A100 80GB"],
        ["Q&A хосуудын тоо",       "~9,040",          "4 төрлийн асуулт-хариулт"],
        ["Dataset нийт клип",      "3,850",           "RWF-2000 + UCF-Crime + Hockey"],
        ["Текстийн тайлбар тоо",   "2,269",           "Teacher-ийн автомат тайлан"],
        ["Хүчирхийллийн түвшин",   "4 (none/low/med/high)", "Ангиллын бүтэц"],
        ["Batch size",             "16",              "Gradient accumulation 4×"],
    ]

    num_table = Table(
        [[Paragraph(r[0], ParagraphStyle("NK", fontName=BOLD_FONT, fontSize=9, textColor=NAVY, leading=13)),
          Paragraph(r[1], ParagraphStyle("NV", fontName=BOLD_FONT, fontSize=10, textColor=TEAL, leading=14, alignment=TA_CENTER)),
          Paragraph(r[2], ParagraphStyle("ND", fontName=BASE_FONT, fontSize=9, textColor=GRAY, leading=13))]
         for r in numbers],
        colWidths=[5.5*cm, 3.5*cm, W-9.2*cm]
    )
    num_table.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), LIGHT),
        ("ROWBACKGROUNDS",(0,0),(-1,-1), [LIGHT, WHITE]),
        ("GRID",         (0,0), (-1,-1), 0.3, colors.HexColor("#CBD5E1")),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
        ("LEFTPADDING",  (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("LINEABOVE",    (0,0), (-1,0),  1.5, TEAL),
    ]))
    story.append(num_table)
    story.append(Spacer(1, 0.5*cm))

    # Footer note
    story.append(Paragraph(
        "Энэхүү гарын авлага нь диплом хамгаалалтын бэлтгэлд зориулагдсан. "
        "Асуулт бүрийн хариултыг өөрийн үгээр дасгал хийж, тоонуудыг цээжлэн бэлд.",
        ParagraphStyle("Footer", fontName=BASE_FONT, fontSize=8.5, textColor=GRAY,
                       alignment=TA_CENTER, leading=13)
    ))

    doc.build(story)
    print(f"✓  Saved: {filename}  ({q_total} Q&A pairs)")

if __name__ == "__main__":
    build_pdf("defense_qa.pdf")
