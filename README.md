# Mongolian NLP Sentiment Classifier

MN-BERT fine-tuned on a manually annotated 10,000-sample Mongolian social media dataset.
Classifies comments into four categories: **Эерэг (Positive)**, **Саармаг (Neutral)**,
**Хортой сөрөг (Toxic)**, **Бүтээлч шүүмжлэл (Constructive Criticism)**.

Bachelor's diploma project — School of Information and Communication Technology, MUST.

---

## Project structure

```
diploma/
├── mongolian-scraper/      # news.mn and gogo.mn comment scrapers (Selenium)
├── facebookmnscraper/      # Facebook public comment scraper (Playwright)
├── annotation/             # Random sampling for manual labelling batches
├── cleaning_data/          # Raw comment cleaning pipeline
├── preprocessing/          # Cyrillic validation, tokenisation, stop-word removal
├── sample scores/
│   ├── project/src/        # Baseline models (LR, NB, SVM) + MN-BERT training
│   ├── step2_retrain_v*.py # Iterative retraining with label fixes (v3–v9)
│   ├── train_mnbert_4class.py  # Final flat 4-class fine-tuning script
│   └── mnbert_4class.ipynb     # Training notebook
└── gradio_app/             # Inference web app (Gradio)
```

## Quickstart — inference app

```bash
pip install -r gradio_app/requirements.txt
python -m gradio_app.app
```

Requires fine-tuned model weights in the paths defined in `gradio_app/config.py`.

## Training pipeline

```
Sprint 1  mongolian-scraper / facebookmnscraper   — data collection
Sprint 2  annotation / cleaning_data / preprocessing — data prep
Sprint 3  sample scores/project/src/train_baselines.py — TF-IDF baselines
Sprint 4  sample scores/project/src/train_bert_v1.py  — MN-BERT v1
Sprint 5  sample scores/step2_retrain_v3..v6.py        — iterative fixes
Sprint 6  sample scores/step2_retrain_v7*.py           — best model (v7)
Sprint 7  sample scores/train_mnbert_4class.py         — flat 4-class
Sprint 8  sample scores/generate_eda.py + project/src/analyze_errors.py
Sprint 9  gradio_app/                                  — web interface
```

## Dataset

- **10,000 manually labelled** Mongolian social media comments
- Sources: news.mn, gogo.mn, Facebook
- Inter-annotator agreement: Cohen's κ = 0.72
- Label distribution: Constructive 45.8% · Neutral 29.0% · Toxic 18.7% · Positive 6.5%

## Results

| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| TF-IDF + LogisticRegression | ~65% | ~0.61 |
| TF-IDF + LinearSVC | ~67% | ~0.63 |
| MN-BERT (flat 4-class, v7) | **81.4%** | **0.776** |

## Requirements

- Python 3.10+
- PyTorch + torch-directml (AMD GPU) or CUDA
- transformers, gradio, scikit-learn, pandas, selenium, playwright
