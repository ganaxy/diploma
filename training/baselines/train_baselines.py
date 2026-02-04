import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from src.config import (
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGES, MODELS, RANDOM_SEED, MODELS_DIR,
)

def build_pipeline(model_name: str, ngram_range: tuple) -> Pipeline:
    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=ngram_range,
        sublinear_tf=True,
    )

    if model_name == "LogisticRegression":
        clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            solver="lbfgs",
        )
    elif model_name == "LinearSVC":
        clf = LinearSVC(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )
    elif model_name == "MultinomialNB":
        clf = MultinomialNB(alpha=1.0)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline([("tfidf", tfidf), ("clf", clf)])

def train_all_models(X_train, y_train):
    trained = []

    for model_name in MODELS:
        for ngram_key, ngram_range in TFIDF_NGRAM_RANGES.items():
            full_name = f"{model_name}_{ngram_key}"
            print(f"\n[train] Training {full_name} ...")
            pipe = build_pipeline(model_name, ngram_range)
            pipe.fit(X_train, y_train)
            print(f"[train] {full_name} — done")
            trained.append((full_name, pipe))

    return trained

def save_model(pipe: Pipeline, model_name: str):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    joblib.dump(pipe, path)
    print(f"  Saved model: {path}")
