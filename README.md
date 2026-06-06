# Монгол хэлний цахим орчин дахь хортой ба бүтээлч агуулгыг ялгах NLP систем

MN-BERT загварыг нарийн тохируулга хийж Монгол хэлний цахим орчны сэтгэгдлийг **4 ангилалд** автоматаар ялгах систем. Хортой сөрөг ба Бүтээлч шүүмжлэлийг ялган таних нь Монгол NLP-д анх удаа системчилсэн байдлаар хэрэгжүүлж буй судалгаа юм.

Бакалаврын дипломын ажил — ШУТИС, Мэдээлэл, Харилцаа Холбооны Технологийн Сургууль, Хиймэл оюун ухаан, 2026.

---

## Ангилалууд

| Ангилал | Тайлбар |
|---------|---------|
| 🟢 Эерэг | Магтаал, талархал, дэмжлэг |
| ⚪ Саармаг | Мэдээлэл дамжуулах, хэлэлцүүлэг |
| 🔵 Бүтээлч шүүмжлэл | Үндэслэлтэй, шийдэл агуулсан шүүмжлэл |
| 🔴 Хортой сөрөг | Доромжлол, гүтгэлэг, кибер дарамт |

---

## Өгөгдлийн эх үүсвэр

Нийтэд нээлттэй 8 эх сурвалжаас цуглуулсан **10,000** гараар шошголсон Монгол кирилл сэтгэгдэл:

- news.mn, gogo.mn, IKON.mn — мэдээний сайтуудын сэтгэгдэл
- Facebook — нийтийн групп, хуудасны сэтгэгдэл
- E-Mongolia — засгийн газрын платформын сэтгэгдэл
- YouTube, Twitter/X — нийгмийн сүлжээний сэтгэгдэл

Шошгололт: 3 бие даасан шошгологч, Fleiss' κ = 0.72 (хангалттай нийцэл), зөрөлдсөн тохиолдлуудыг консенсусаар шийдсэн.

| Ангилал | Тоо | Хувь |
|---------|-----|------|
| Бүтээлч шүүмжлэл | 4,582 | 45.8% |
| Саармаг | 2,898 | 29.0% |
| Хортой сөрөг | 1,870 | 18.7% |
| Эерэг | 650 | 6.5% |
| **Нийт** | **10,000** | **100%** |

---

## Загварын үр дүн

| Загвар | Accuracy | Macro F1 |
|--------|----------|----------|
| TF-IDF + Logistic Regression | 68.20% | 0.6500 |
| TF-IDF + Naive Bayes | 62.10% | 0.5900 |
| TF-IDF + LinearSVC | 67.90% | 0.6400 |
| MN-BERT (суурь SOTA) | 81.43% | 0.7760 |
| Хоёр шатлалт MN-BERT (end-to-end) | 78.61% | — |
| **MN-BERT Flat (Focal Loss + Cosine Warmup)** | **83.26%** | **0.8062** |

Эцсийн загвар нь Focal Loss (γ=2.0), Cosine Warmup scheduler, WeightedRandomSampler ашиглан ангиллын тэнцвэргүй байдлыг шийдсэн.

---

## Төслийн бүтэц

```
diploma/
├── training/
│   ├── pipeline/             # Хоёр шатлалт ангилагч, үнэлгээний скриптүүд
│   ├── baselines/            # TF-IDF + LR / NB / SVM суурь загварууд
│   ├── bert/                 # MN-BERT сургалтын скриптүүд
│   └── experiments/          # Туршилтын итерациуд
│
├── gradio_app/               # Gradio веб интерфэйс (inference)
│   ├── app.py
│   ├── models.py
│   ├── preprocess.py
│   └── requirements.txt
│
├── sample scores/
│   └── relabeled_v7_corrected.csv   # Final 10k annotated dataset used by the app
├── requirements.txt
└── .gitignore
```

---

## Хурдан эхлэх

```bash
pip install -r requirements.txt
```

### Gradio интерфэйс ажиллуулах

```bash
python gradio_app/app.py
```

### MN-BERT сургах

```bash
python training/bert/train_bert.py
```

### Хоёр шатлалт загвар сургах

```bash
python training/pipeline/train_stage1.py
python training/pipeline/train_stage2_from_stage1.py
```

---

## Технологийн стек

- Python 3.10+
- [MN-BERT](https://huggingface.co/tugstugi/bert-base-mongolian-cased) — Монгол хэлний BERT
- HuggingFace Transformers · PyTorch
- Gradio — inference интерфэйс
- scikit-learn · imbalanced-learn (SMOTE) · pandas · numpy
