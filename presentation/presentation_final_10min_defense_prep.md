# 10-minute defense prep for `presentation_final.pptx`

Use this as a speaking guide, not as a script to memorize word-for-word. The goal is to keep one clear story:

**Би Монгол кирилл онлайн сэтгэгдлийг 4 ангиллаар ялгах өгөгдөл, MN-BERT загвар, API/прототип систем боловсруулж, хортой сөрөг агуулга болон бүтээлч шүүмжлэлийг тусад нь ялгах боломжтойг харуулсан.**

## Timing plan

Total target: **9:55**. Leave 5 seconds buffer.

| Slide | Time | Main job |
|---|---:|---|
| 1 | 0:20 | Introduce topic |
| 2 | 0:10 | Show flow |
| 3 | 0:45 | Motivation/problem |
| 4 | 0:40 | Goal/objectives |
| 5 | 0:50 | Related work + gap |
| 6 | 0:45 | Dataset summary |
| 7 | 0:30 | Labeling/source process |
| 8 | 0:40 | System architecture |
| 9 | 0:40 | Training setup |
| 10 | 0:35 | MN-BERT architecture |
| 11 | 0:45 | Baseline comparison |
| 12 | 0:45 | One-stage vs two-stage |
| 13 | 0:45 | Shared encoder achievement |
| 14 | 0:45 | Final model result |
| 15 | 0:40 | Conclusion/future work |

If you are running late, skip detailed explanation on slides 2, 5, and 9. Spend time on slides 6, 8, 11, 13, 14.

---

## Slide 1 - Title

**Say:**

Сайн байна уу. Миний дипломын ажлын сэдэв бол "Монгол хэлний цахим орчин дахь хандлагыг тодорхойлох NLP системийн хөгжүүлэлт" юм. Энэ ажлаар би Монгол кирилл онлайн сэтгэгдлийг эерэг, саармаг, хортой сөрөг, бүтээлч шүүмжлэл гэсэн 4 ангиллаар ялгах өгөгдөл, MN-BERT загвар, мөн турших боломжтой прототип/API систем боловсруулсан.

**Likely questions:**

Q: Why is this NLP, not just sentiment analysis?  
A: Уламжлалт sentiment analysis ихэвчлэн positive/negative/neutral гэж ялгадаг. Энэ ажилд хортой сөрөг ба бүтээлч шүүмжлэлийг тусад нь ялгаж байгаа тул moderation-oriented NLP classification гэж үзэж болно.

Q: What is the main contribution in one sentence?  
A: Монгол кирилл онлайн сэтгэгдэлд зориулсан 4 ангиллын өгөгдөл, MN-BERT суурьтай ангилагч, API/прототип системийг боловсруулж үнэлсэн.

---

## Slide 2 - Agenda

**Say:**

Илтгэлээ дараах дарааллаар танилцуулна: эхлээд судалгааны үндэслэл, дараа нь зорилго ба холбоотой судалгаа, өгөгдлийн багц, системийн архитектур, MN-BERT загвар, туршилтын үр дүн, эцэст нь дүгнэлт ба цаашдын ажлыг хэлнэ.

**Likely questions:**

Q: Which part is the most important?  
A: Үндсэн нотолгоо нь өгөгдлийн багц, MN-BERT-ийн туршилтын үр дүн, мөн shared encoder архитектурын харьцуулалт дээр байна.

---

## Slide 3 - Research motivation

**Say:**

Судалгааны үндэслэл нь Монгол хэрэглэгчдийн цахим орчны хэрэглээ өндөр болсон, мөн сэтгэгдэл дотор хортой, доромжилсон, эсвэл бүтээлч шүүмжлэл агуулсан текстүүд их байдагтай холбоотой. Зөвхөн "сөрөг" гэж тэмдэглэх нь хангалтгүй. Жишээ нь шүүмжлэл нь нэг талаас асуудал зааж өгч байгаа ашигтай feedback байж болно, нөгөө талаас хортой хэллэг байж болно. Иймээс энэ хоёр төрлийг тусад нь ялгах хэрэгтэй.

**Likely questions:**

Q: Why separate toxic negative and constructive criticism?  
A: Moderation system-д энэ ялгаа чухал. Хортой сөрөг агуулгыг flag/review хийх хэрэгтэй, харин бүтээлч шүүмжлэлийг устгах биш хадгалах нь зөв.

Q: Are the 83% and 76.5% numbers your experimental results?  
A: Үгүй. Эдгээр нь сэдвийн хэрэгцээг харуулах motivation/statistics. Миний туршилтын үндсэн үр дүн дараагийн result slides дээр байна.

Q: What is the practical use?  
A: Мэдээний сайт, social media, байгууллагын comment review системд эхний шатны автомат шүүлтүүр, audit, human review prioritization хийхэд ашиглаж болно.

---

## Slide 4 - Goal and objectives

**Say:**

Ажлын зорилго нь Монгол хэлний цахим орчны сэтгэгдлийг 4 ангиллаар автоматаар ялгах MN-BERT суурьтай NLP систем боловсруулах. Үүний тулд 10,000 сэтгэгдлийн өгөгдлийн багц бүрдүүлж, шошгологчдын нийцлийг хэмжсэн, Монгол кирилл текстэд тохирсон боловсруулалт хийсэн, MN-BERT болон baseline загваруудыг харьцуулсан, эцэст нь Gradio болон FastAPI прототип гаргасан.

**Likely questions:**

Q: Why 4 classes?  
A: Эерэг, саармаг, хортой сөрөг, бүтээлч шүүмжлэл гэсэн 4 ангилал нь moderation-д илүү хэрэгтэй. Ялангуяа "хортой" ба "бүтээлч" гэсэн ялгаа нь шийдвэр гаргалтад шууд нөлөөлнө.

Q: What is the difference between Gradio and API?  
A: Gradio бол human-facing demo UI. FastAPI бол бусад систем дуудаж ашиглах backend/service layer. API нь `/predict`, `/batch`, `/audit` endpoint-той.

Q: Which objective was hardest?  
A: Өгөгдөл шошгололтын чанар ба хортой сөрөг/бүтээлч шүүмжлэлийн заагийг тогтвортой болгох хамгийн хэцүү байсан.

---

## Slide 5 - Related work and novelty

**Say:**

Холбоотой ажлуудыг харьцуулахад англи хэл дээр hate/offensive болон toxic comment classification сайн судлагдсан. Монгол хэл дээр sentiment судалгаа байгаа ч ихэнх нь 3 ангилалтай эсвэл уламжлалт baseline загвартай байсан. Миний ажлын шинэлэг тал нь Монгол кирилл онлайн сэтгэгдэл дээр хортой сөрөг ба бүтээлч шүүмжлэлийг тусад нь ялгах 4 ангиллын корпус, MN-BERT суурьтай систем, прототип/API шийдэл гаргасанд оршино.

**Likely questions:**

Q: What is the exact research gap?  
A: Монгол кирилл цахим сэтгэгдэлд зориулсан 4 ангиллын, moderation-oriented өгөгдөл ба системийн шийдэл хомс байсан.

Q: Why not use multilingual BERT or XLM-R?  
A: Боломжтой хувилбар. Гэхдээ MN-BERT нь Монгол хэл дээр pretrain хийгдсэн тул Монгол кирилл текстийн tokenization, contextual representation-д илүү тохиромжтой гэж үзсэн. Future work-д XLM-R/mBERT-тэй өргөн харьцуулж болно.

Q: Is your novelty dataset or model?  
A: Аль аль нь. Dataset/class definition нь domain contribution, MN-BERT pipeline/API/shared-encoder experiments нь system/model contribution.

---

## Slide 6 - Dataset and processing overview

**Say:**

Өгөгдлийн багц нийт 10,000 шошголсон сэтгэгдлээс бүрдсэн. Эх сурвалж нь news.mn, gogo.mn болон Facebook. 3 шошгологч оролцож, шошгологчдын нийцэл κ=0.72 түвшинд гарсан. Ангиллын хувьд constructive болон neutral илүү олон, positive харьцангуй бага тул class imbalance асуудал байсан.

**Likely questions:**

Q: What does κ=0.72 mean?  
A: Энэ нь шошгологчдын хоорондын agreement-ийг хэмжсэн үзүүлэлт. 0.72 нь substantial agreement гэж тайлбарлаж болно. Өөрөөр хэлбэл ангиллын заавар харьцангуй тогтвортой байсан.

Q: Why is positive class small?  
A: Онлайн comment орчинд эерэг сэтгэгдэл харьцангуй бага, neutral/constructive/negative агуулга илүү давамгай байдаг. Тиймээс class imbalance үүссэн.

Q: How did you handle imbalance?  
A: Training үед focal loss γ=2 ашигласан. Мөн evaluation-д accuracy-аас гадна Macro-F1 харсан, учир нь Macro-F1 жижиг ангиллуудыг илүү шударга тусгадаг.

Q: How big was the test set?  
A: Final evaluation report дээр test set n=1529 байна.

---

## Slide 7 - Source distribution / labeling flow

**Say:**

Энэ слайд дээр өгөгдлийн эх сурвалжийн тархалт болон шошгололтын урсгалыг харуулсан. Комментуудыг эхлээд цэвэрлэж, 3 шошгологчоор шошголуулж, κ хэмжиж, зөрөлдөөнтэй тохиолдлуудыг нягталсны дараа эцсийн шошгыг гаргасан. Энэ нь өгөгдлийн чанарыг хамгаалах чухал хэсэг байсан.

**Likely questions:**

Q: Did you remove duplicates or noisy comments?  
A: Тийм, preprocessing үе шатанд URL/HTML/emoji/noisy tokens зэрэг зүйлсийг цэвэрлэх, normalized text үүсгэх алхамууд орсон. Давхардал болон чанарын асуудлыг аль болох багасгасан.

Q: Could source distribution bias the model?  
A: Тийм, боломжтой. News site ба Facebook-ийн хэллэг ялгаатай. Тиймээс source diversity ашигласан. Future work-д time-based эсвэл source-held-out evaluation хийх нь generalization-ийг илүү сайн шалгана.

Q: Why use real online comments?  
A: Synthetic өгөгдөл бодит moderation орчны хэллэг, slang, алдаа, informal style-ийг хангалттай тусгахгүй. Бодит comment нь practical system-д илүү тохиромжтой.

---

## Slide 8 - System architecture

**Say:**

Системийн урсгал нь оролтын кирилл сэтгэгдлээс эхэлнэ. Дараа нь HTML/URL болон emoji цэвэрлэх, SentencePiece tokenization хийх, MN-BERT encoder ашиглан contextual representation гаргах, 4 ангиллын softmax output авах, эцэст нь Gradio интерфейс эсвэл API-аар үр дүнг харуулах гэсэн бүтэцтэй.

**Likely questions:**

Q: What happens before MN-BERT?  
A: Text normalization, cleaning, source/input formatting, tokenization. Энэ нь model-д training үеийн форматтай ойролцоо оролт өгөхөд хэрэгтэй.

Q: Why SentencePiece?  
A: MN-BERT tokenizer нь subword tokenization ашигладаг. Монгол хэлэнд үгийн хувилбар, залгавар их тул subword tokenization нь unknown word асуудлыг багасгадаг.

Q: What is the output?  
A: 4 class probability distribution ба хамгийн өндөр магадлалтай label. API дээр confidence болон moderation action бас өгдөг.

---

## Slide 9 - MN-BERT training setup

**Say:**

MN-BERT загварыг AdamW optimizer, learning rate 3e-5, batch size 16, max length 256 гэсэн тохиргоогоор fine-tune хийсэн. Cosine scheduler болон warmup ашигласан. Class imbalance-ийг багасгахын тулд focal loss γ=2 хэрэглэсэн. Early stopping patience=3 ашиглаж overfitting-ээс сэргийлсэн.

**Likely questions:**

Q: Why focal loss?  
A: Focal loss нь амархан ангилагдаж байгаа олонх жишээнүүдийн нөлөөг багасгаж, хэцүү болон ховор class дээр model-ийг илүү төвлөрүүлдэг. Энэ dataset дээр positive/toxic зэрэг class imbalance байсан.

Q: Why max length 256?  
A: Comment text ихэвчлэн богино. 256 token нь ихэнх сэтгэгдлийг багтаах бөгөөд latency/memory зардлыг хэт нэмэхгүй.

Q: Did you tune hyperparameters extensively?  
A: Үндсэн hyperparameter-үүдийг BERT fine-tuning-д нийтлэг хэрэглэгддэг range дээр сонгож, validation performance болон early stopping-оор хянасан. Илүү өргөн hyperparameter search нь future improvement байж болно.

---

## Slide 10 - MN-BERT model architecture

**Say:**

Энэ слайд MN-BERT-ийн classification pipeline-ийг харуулж байна. Оролтын кирилл текст SentencePiece tokenizer-оор subword token болно. Дараа нь `[CLS]` token-ийн pooled representation гарч, classification head буюу linear + softmax давхаргаар 4 ангиллын магадлал гарна. Загвар ойролцоогоор 110M параметртэй, hidden size 768.

**Likely questions:**

Q: What is `[CLS]` used for?  
A: `[CLS]` representation нь бүх өгүүлбэрийн contextual summary гэж ашиглагддаг. Classification head энэ vector дээр тулгуурлаж label prediction хийдэг.

Q: Why use one-stage final model?  
A: Туршилтаар one-stage MN-BERT хамгийн өндөр final result өгсөн: 83.26% accuracy, 0.8062 Macro-F1. Тиймээс final deployed/API model болгон сонгосон.

Q: Is the model explainable?  
A: Full explainability биш. Гэхдээ prototype дээр probability distribution, confidence, зарим highlight/decision explanation харуулах боломжтой. Future work-д attention/gradient-based explanation нэмэгдүүлж болно.

---

## Slide 11 - Baseline comparison

**Say:**

Энд Naive Bayes, SVM, Logistic Regression зэрэг baseline-уудтай MN-BERT-ийг харьцуулсан. MN-BERT нь contextual representation ашигладаг тул уламжлалт TF-IDF суурьтай загваруудаас илүү сайн үр дүн үзүүлсэн. Гол үзүүлэлтүүд нь accuracy ба Macro-F1 бөгөөд Macro-F1 нь бүх class-ийг адил жинтэй авч үздэг.

**Likely questions:**

Q: Why compare with classical baselines?  
A: Baseline хэрэгтэй, учир нь MN-BERT үнэхээр value нэмсэн эсэхийг шалгана. Хэрэв simple SVM ойролцоо байсан бол BERT-ийн complexity justified биш байх байсан.

Q: Why Macro-F1?  
A: Dataset imbalance-тэй тул accuracy дангаараа хангалтгүй. Macro-F1 нь positive/toxic гэх мэт жижиг class-үүдийн performance-ийг илүү сайн харуулна.

Q: Did Logistic Regression almost catch BERT?  
A: Зарим class дээр ойролцоо байж болно, гэхдээ overall final accuracy болон Macro-F1 дээр MN-BERT илүү. Мөн contextual understanding шаардсан toxic vs constructive ялгалтад transformer илүү тохиромжтой.

---

## Slide 12 - One-stage vs two-stage comparison

**Say:**

Энэ слайд final one-stage MN-BERT болон хоёр шатлалт хувилбарыг харьцуулж байна. Хоёр шатлалт хувилбар нь эхлээд эерэг/саармаг/сөрөг гэж ялгаад, дараа нь сөрөг хэсгийг хортой эсвэл бүтээлч гэж салгах санаатай байсан. Гэхдээ cascade error буюу эхний шатны алдаа дараагийн шатанд дамжих асуудал гарсан. One-stage MN-BERT нэг pass-аар 4 ангиллыг шууд ялгаж, илүү өндөр accuracy, Macro-F1, latency үзүүлсэн.

**Likely questions:**

Q: Why did two-stage perform worse?  
A: Stage 1 буруу route хийвэл Stage 2 зөв засах боломжгүй. Энэ cascade error. Мөн хоёр тусдаа decision process нь final 4-class objective-той шууд optimize хийгдээгүй.

Q: Then why did you try two-stage?  
A: Toxic vs constructive ялгаа нь semantic hierarchy-тэй мэт санагдсан: эхлээд negative эсэх, дараа нь negative subtype. Судалгааны хувьд reasonable hypothesis байсан, харин experiment-аар one-stage илүү болохыг харуулсан.

Q: Is speed important?  
A: Тийм. Real-time moderation эсвэл API service-д latency чухал. One-stage model нь нэг удаа inference хийдэг тул хоёр шатлалтаас хурдан.

---

## Slide 13 - Architecture difference: shared encoder

**Say:**

Хоёр шатлалт хувилбарын сул талын нэг нь хоёр тусдаа BERT encoder ашигласан явдал байсан. Нэг comment-ийг хоёр encoder өөр representation болгон хувиргах боломжтой, мөн model давхардал их. Үүнийг сайжруулахын тулд shared encoder multi-head архитектур туршсан. Энэ хувилбарт нэг MN-BERT encoder text-ийг нэг удаа уншаад, гарсан representation-ийг Stage 1 head болон Stage 2 head хамт ашигладаг. Үр дүн нь two-stage pipeline-ээс сайжирч 82.41% accuracy, 0.7972 Macro-F1 болсон. Гэхдээ final one-stage MN-BERT 83.26%, 0.8062 Macro-F1 тул эцсийн model болгон one-stage-ийг сонгосон.

**Likely questions:**

Q: What is a shared encoder in simple words?  
A: Нэг BERT "уншигч" text-ийг нэг удаа уншаад, хоёр classifier head тэр нэг ижил representation дээр шийдвэр гаргана.

Q: What problem does it solve?  
A: Хоёр тусдаа encoder-ийн representation mismatch болон model duplication-ийг багасгана. Stage 1 ба Stage 2 нэг ижил ойлголт дээр ажиллана.

Q: If shared encoder improved two-stage, why not final model?  
A: Shared encoder нь original two-stage-ээс сайжирсан ч final one-stage MN-BERT хамгийн өндөр score өгсөн. Тиймээс shared encoder бол architecture achievement/experiment, харин final deployed model нь one-stage.

Q: Does shared encoder remove cascade error completely?  
A: Үгүй. Representation mismatch багасна, гэхдээ Stage 1 route буруу байвал cascade error үлдэж болно. Тиймээс one-stage илүү тогтвортой гарсан.

---

## Slide 14 - Final MN-BERT model

**Say:**

Эцсийн MN-BERT загвар test set дээр 83.26% accuracy, 0.8062 Macro-F1 үзүүлсэн. Confusion matrix-аас харахад constructive criticism хамгийн сайн танигдсан class-уудын нэг байна. Positive class-ийн support бага тул тэр class дээр performance бага зэрэг эмзэг байж болно. Гол үр дүн нь хортой сөрөг ба бүтээлч шүүмжлэлийг тусад нь ялгаж чадсан явдал.

**Likely questions:**

Q: Which class is weakest?  
A: Positive class support бага тул positive class илүү эмзэг. Final report дээр positive F1 ойролцоогоор 0.7416, neutral 0.8030, toxic 0.8041, constructive 0.8760 байна.

Q: What errors does the model make?  
A: Neutral vs constructive, toxic vs constructive зэрэг boundary ойролцоо class-үүдэд төөрөгдөл гардаг. Энэ нь human annotators-д ч хэцүү semantic boundary.

Q: Is 83.26% enough?  
A: Production-д шууд human replacement гэж үзэхгүй. Харин moderation assistance буюу flag/review prioritization-д хэрэгтэй түвшин. Human-in-the-loop шаардлагатай.

Q: Why not only report accuracy?  
A: Accuracy class imbalance-д хуурмаг байж болно. Macro-F1 нь class бүрийг тэнцүү авч үздэг тул илүү шударга.

---

## Slide 15 - Conclusion and future work

**Say:**

Дүгнэж хэлбэл, энэ ажлаар 10,000 сэтгэгдэлтэй 4 ангиллын өгөгдлийн багц бүрдүүлж, шошгологчдын нийцлийг κ=0.72 түвшинд баталгаажуулсан. MN-BERT суурьтай final one-stage model хамгийн сайн үр дүнтэй гарч, хортой сөрөг болон бүтээлч шүүмжлэлийг тусад нь ялгах боломжтойг харуулсан. Мөн Gradio demo болон FastAPI service гаргасан. Цаашид илүү том олон эх сурвалжтай corpus, human-in-the-loop хяналт, production deployment болон real-time moderation integration хийх шаардлагатай.

**Likely questions:**

Q: Did you already build the API?  
A: Тийм. FastAPI service prototype хийсэн. `/predict`, `/batch`, `/audit` endpoint-уудтай. Future work нь API-г production орчинд байршуулж real-time moderation workflow-той бүрэн нэгтгэх.

Q: What is the biggest limitation?  
A: Dataset size/source diversity, class imbalance, болон real-world deployment evaluation. Илүү олон эх сурвалж, time-based/source-held-out split, human-in-the-loop production test хэрэгтэй.

Q: What would you improve first?  
A: Илүү том corpus болон active learning/human-in-the-loop labeling. Ингэснээр hard examples дээр model сайжирна.

Q: What is the practical final output?  
A: Final MN-BERT classifier, Gradio UI demo, FastAPI REST API prototype, evaluation outputs, and dataset/model pipeline.

---

## Very likely hard questions

### Why final one-stage if you spent time on two-stage and shared encoder?

Two-stage was a research hypothesis. Shared encoder improved the two-stage architecture by reducing duplicated encoders and representation mismatch. But final one-stage MN-BERT had the best measured performance: **83.26% accuracy / 0.8062 Macro-F1**, so it was selected as final.

### What exactly is the API contribution?

Gradio is a demo UI for humans. FastAPI is a machine-to-machine service. It exposes:

- `/predict`: one comment classification
- `/batch`: batch classification
- `/audit`: moderation-style summary

Future work is not "make API"; it is **production deployment + real-time moderation integration**.

### What does "constructive criticism" mean?

It is negative or critical feedback that contains useful information, suggestion, or problem report without harmful/toxic attack. The system should preserve this instead of blocking it.

### How do you know labels are reliable?

Three annotators were used and agreement was measured with κ=0.72. This is not perfect, but it is substantial agreement and shows the guideline was reasonably stable.

### What if professors ask about ethics?

Say: This should be used as decision support, not automatic censorship. Low-confidence cases should go to human review. The goal is to reduce harmful content while preserving constructive criticism.

### What if they ask about data leakage?

Safe answer: The final evaluation is done on a held-out test split of 1,529 rows. A limitation is that future work should also test source-held-out or time-held-out splits to measure generalization across platforms and periods.

### What if they ask why positive class is weak?

Positive has fewer examples and online comments are naturally skewed toward complaint/neutral/criticism. This imbalance makes positive harder, so future data expansion should target underrepresented classes.

### What if they ask "what is MN-BERT?"

MN-BERT is a BERT-style transformer pretrained for Mongolian text. It produces contextual embeddings, so the meaning of a word depends on surrounding words. This is useful for toxic vs constructive classification because context matters.

## 3-minute emergency version

If they cut your time, use this:

1. Problem: Монгол онлайн сэтгэгдэлд хортой сөрөг ба бүтээлч шүүмжлэлийг ялгах хэрэгтэй.
2. Data: 10,000 comments, 4 classes, 3 annotators, κ=0.72.
3. Method: MN-BERT fine-tuning, text cleaning, SentencePiece tokenization, focal loss for imbalance.
4. Results: Final one-stage MN-BERT = 83.26% accuracy, 0.8062 Macro-F1.
5. Architecture: two-stage had cascade error; shared encoder improved two-stage, but one-stage remained best.
6. System: Gradio demo + FastAPI `/predict`, `/batch`, `/audit`.
7. Conclusion: harmful content can be filtered while preserving constructive criticism.

## Practice rule

For each slide, say only:

1. What the slide proves.
2. One number or object from the slide.
3. Why it matters for the final system.

Do not read every label. The professors can read; your job is to connect the evidence.

## Professor trap: why MN-BERT, why this architecture, and token count

### If they ask: "Why MN-BERT? There are other high-performing models."

Say:

I did not choose MN-BERT because it is the biggest model. I chose it because the task is Mongolian Cyrillic comment classification and MN-BERT is a local, fine-tunable BERT model for Mongolian text. For this thesis, I needed a reproducible model that can be trained and evaluated on my labeled dataset, run locally through my API, and avoid depending on a paid external model API.

Then add:

Large external models can be compared in future work, but my contribution is the dataset, the architecture comparison, the fine-tuned MN-BERT classifier, and the working API/prototype for this specific Mongolian moderation task.

### If they ask: "Why this architecture?"

Say:

The final task has four output classes, so the main architecture is a one-stage 4-class MN-BERT classifier. I also tested two-stage and shared-encoder alternatives. The old two-stage design had cascade error: if stage 1 routes a comment incorrectly, stage 2 cannot fix it. The shared encoder reduced the mismatch by using one BERT reader with two classifier heads, but the one-stage model still gave the best final result.

Numbers:

- SVM TF-IDF: 66.91% accuracy, 0.5733 Macro-F1
- Two-stage pipeline: 78.94% accuracy, 0.7628 Macro-F1
- Shared encoder multi-head: 82.41% accuracy, 0.7972 Macro-F1
- Final one-stage MN-BERT: 83.26% accuracy, 0.8062 Macro-F1

### If they ask: "Could you use your BERT classifier plus another model API?"

Say:

Yes. A practical extension would be a hybrid moderation system. MN-BERT would make the first fast local prediction. If confidence is high, the API returns the label directly. If confidence is low or the comment is risky, the system can send that case to a larger external model API or to human review.

Example flow:

1. Comment enters API.
2. MN-BERT predicts label and confidence.
3. If confidence is above threshold, return MN-BERT result.
4. If confidence is below threshold, call an external model API or send to reviewer.
5. Log the decision for future dataset improvement.

Important defense sentence:

I did not make the external API the core model because it creates cost, latency, privacy, and reproducibility issues. But it is a strong future-work direction as a second-opinion model or teacher model.

### If they ask: "How many tokens is your model, 20k or 50k?"

Clarify first:

If you mean context length, my final API uses max_length 256 tokens per comment. BERT's architecture supports 512 positional embeddings, but I serve the model at 256 because comments are short and this keeps inference faster.

If you mean vocabulary size, the tokenizer vocabulary is 32,001 tokens. The tokenizer uses a Unigram/SentencePiece-style subword vocabulary, so unknown or rare Mongolian words can be broken into smaller pieces instead of being completely lost.

Safe short answer:

The current serving input length is 256 tokens per comment. The tokenizer vocabulary size is 32,001. The BERT config has 512 positional embeddings, but I chose 256 for this comment-classification API.
