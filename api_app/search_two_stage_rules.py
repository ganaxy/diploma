"""Search no-retraining decision rules for the two-stage MN-BERT pipeline.

The expensive part is running Stage 1 and Stage 2 BERT. This script runs both
models once per row, caches their probabilities, then tests many lightweight
architecture rules on top of those cached probabilities.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

os.environ.setdefault("GRADIO_FORCE_CPU", "1")

from gradio_app import config
from gradio_app.models import TwoStageClassifier, pick_device


LABELS = list(config.FLAT_LABELS)
DEFAULT_OUT_DIR = Path("api_app") / "eval_outputs"
DEFAULT_CACHE = DEFAULT_OUT_DIR / "two_stage_probability_cache.csv"
EPS = 1e-12


def _row_text(row: pd.Series) -> str:
    return str(row.get("text_light_clean") or row.get("text_raw") or "")


def build_probability_cache(
    splits: list[str],
    cache_path: Path,
    limit_per_split: int = 0,
) -> pd.DataFrame:
    source = pd.read_csv(config.TEST_CSV, encoding="utf-8-sig")
    source["split"] = source["split"].astype(str).str.lower()
    source = source[source["split"].isin(splits)].reset_index(drop=True)

    if limit_per_split > 0:
        source = (
            source.groupby("split", group_keys=False)
            .head(limit_per_split)
            .reset_index(drop=True)
        )

    device, device_label = pick_device()
    print(f"Loading two-stage models on {device_label}", flush=True)
    model = TwoStageClassifier(device)

    rows: list[dict] = []
    start = time.perf_counter()
    for idx, row in source.iterrows():
        text = _row_text(row)
        s1 = model.stage1.predict(text)
        s2 = model.stage2.predict(text)
        rows.append(
            {
                "id": row.get("id"),
                "split": str(row["split"]).lower(),
                "true_label": str(row["label"]).upper(),
                "s1_label": s1.label,
                "s1_conf": s1.confidence,
                "s1_positive": float(s1.probs.get("POSITIVE", 0.0)),
                "s1_neutral": float(s1.probs.get("NEUTRAL", 0.0)),
                "s1_negative": float(s1.probs.get("NEGATIVE", 0.0)),
                "s2_label": s2.label,
                "s2_conf": s2.confidence,
                "s2_constructive": float(s2.probs.get("CONSTRUCTIVE", 0.0)),
                "s2_toxic": float(s2.probs.get("TOXIC", 0.0)),
            }
        )
        if (idx + 1) % 100 == 0:
            print(f"Cached {idx + 1}/{len(source)} rows", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = pd.DataFrame(rows)
    cached.to_csv(cache_path, index=False, encoding="utf-8-sig")
    elapsed = time.perf_counter() - start
    print(
        f"Saved probability cache to {cache_path.resolve()} "
        f"({len(cached)} rows, {elapsed:.1f}s)",
        flush=True,
    )
    return cached


def load_or_build_cache(args: argparse.Namespace) -> pd.DataFrame:
    cache_path = Path(args.cache)
    wanted_splits = {args.tune_split.lower(), args.eval_split.lower()}
    if not getattr(args, "skip_tiny_models", False):
        wanted_splits.add(args.tiny_train_split.lower())

    if cache_path.exists() and not args.refresh_cache:
        cached = pd.read_csv(cache_path, encoding="utf-8-sig")
        cached["split"] = cached["split"].astype(str).str.lower()
        have_splits = set(cached["split"].unique())
        if wanted_splits.issubset(have_splits):
            return cached[cached["split"].isin(wanted_splits)].reset_index(drop=True)
        missing_splits = sorted(wanted_splits - have_splits)
        print(f"Cache is missing splits {missing_splits}; adding them.", flush=True)
        temp_path = cache_path.with_name(cache_path.stem + "_missing_tmp.csv")
        missing = build_probability_cache(
            splits=missing_splits,
            cache_path=temp_path,
            limit_per_split=args.limit_per_split,
        )
        combined = pd.concat([cached, missing], ignore_index=True)
        combined = combined.drop_duplicates(subset=["id", "split"], keep="last")
        combined.to_csv(cache_path, index=False, encoding="utf-8-sig")
        temp_path.unlink(missing_ok=True)
        return combined[combined["split"].isin(wanted_splits)].reset_index(drop=True)

    return build_probability_cache(
        splits=sorted(wanted_splits),
        cache_path=cache_path,
        limit_per_split=args.limit_per_split,
    )


def _argmax_label(scores: dict[str, float]) -> str:
    return max(scores, key=scores.get)


def _non_negative_label(row: pd.Series) -> str:
    return "POSITIVE" if row["s1_positive"] >= row["s1_neutral"] else "NEUTRAL"


def _temperature_scale(values: list[float], temperature: float) -> np.ndarray:
    arr = np.maximum(np.asarray(values, dtype=float), EPS)
    scaled = arr ** (1.0 / float(temperature))
    return scaled / scaled.sum()


def predict_rule(df: pd.DataFrame, rule: dict) -> list[str]:
    kind = rule["kind"]
    preds: list[str] = []

    for _, row in df.iterrows():
        if kind == "hard_cascade":
            preds.append(row["s2_label"] if row["s1_label"] == "NEGATIVE" else row["s1_label"])

        elif kind == "soft_multiply":
            preds.append(
                _argmax_label(
                    {
                        "POSITIVE": row["s1_positive"],
                        "NEUTRAL": row["s1_neutral"],
                        "CONSTRUCTIVE": row["s1_negative"] * row["s2_constructive"],
                        "TOXIC": row["s1_negative"] * row["s2_toxic"],
                    }
                )
            )

        elif kind == "power_gate":
            gamma = float(rule["gamma"])
            gate = float(row["s1_negative"]) ** gamma
            preds.append(
                _argmax_label(
                    {
                        "POSITIVE": row["s1_positive"],
                        "NEUTRAL": row["s1_neutral"],
                        "CONSTRUCTIVE": gate * row["s2_constructive"],
                        "TOXIC": gate * row["s2_toxic"],
                    }
                )
            )

        elif kind == "negative_scale":
            scale = float(rule["scale"])
            preds.append(
                _argmax_label(
                    {
                        "POSITIVE": row["s1_positive"],
                        "NEUTRAL": row["s1_neutral"],
                        "CONSTRUCTIVE": scale * row["s1_negative"] * row["s2_constructive"],
                        "TOXIC": scale * row["s1_negative"] * row["s2_toxic"],
                    }
                )
            )

        elif kind == "threshold_router":
            threshold = float(rule["threshold"])
            preds.append(row["s2_label"] if row["s1_negative"] >= threshold else _non_negative_label(row))

        elif kind == "margin_router":
            margin = float(rule["margin"])
            best_non_negative = max(row["s1_positive"], row["s1_neutral"])
            route = (row["s1_negative"] - best_non_negative) >= margin
            preds.append(row["s2_label"] if route else _non_negative_label(row))

        elif kind == "ratio_router":
            ratio = float(rule["ratio"])
            best_non_negative = max(row["s1_positive"], row["s1_neutral"])
            route = row["s1_negative"] >= ratio * best_non_negative
            preds.append(row["s2_label"] if route else _non_negative_label(row))

        elif kind == "calibrated_soft":
            s1_temp = float(rule["s1_temp"])
            s2_temp = float(rule["s2_temp"])
            s1_pos, s1_neu, s1_neg = _temperature_scale(
                [row["s1_positive"], row["s1_neutral"], row["s1_negative"]],
                s1_temp,
            )
            s2_con, s2_tox = _temperature_scale(
                [row["s2_constructive"], row["s2_toxic"]],
                s2_temp,
            )
            preds.append(
                _argmax_label(
                    {
                        "POSITIVE": float(s1_pos),
                        "NEUTRAL": float(s1_neu),
                        "CONSTRUCTIVE": float(s1_neg * s2_con),
                        "TOXIC": float(s1_neg * s2_tox),
                    }
                )
            )

        elif kind == "calibrated_threshold_router":
            s1_temp = float(rule["s1_temp"])
            threshold = float(rule["threshold"])
            s1_pos, s1_neu, s1_neg = _temperature_scale(
                [row["s1_positive"], row["s1_neutral"], row["s1_negative"]],
                s1_temp,
            )
            preds.append(row["s2_label"] if s1_neg >= threshold else ("POSITIVE" if s1_pos >= s1_neu else "NEUTRAL"))

        elif kind == "stage2_override":
            pred = row["s2_label"] if row["s1_label"] == "NEGATIVE" else row["s1_label"]
            if (
                row["s1_label"] != "NEGATIVE"
                and row["s2_conf"] >= float(rule["s2_threshold"])
                and row["s1_conf"] <= float(rule["s1_max_conf"])
                and row["s1_negative"] >= float(rule["neg_floor"])
            ):
                pred = row["s2_label"]
            preds.append(pred)

        elif kind == "negative_rollback":
            pred = row["s2_label"] if row["s1_label"] == "NEGATIVE" else row["s1_label"]
            if (
                row["s1_label"] == "NEGATIVE"
                and row["s1_conf"] <= float(rule["s1_max_conf"])
                and row["s2_conf"] <= float(rule["s2_max_conf"])
            ):
                pred = _non_negative_label(row)
            preds.append(pred)

        else:
            raise ValueError(f"Unknown rule kind: {kind}")

    return preds


def rule_name(rule: dict) -> str:
    kind = rule["kind"]
    params = {k: v for k, v in rule.items() if k != "kind"}
    if not params:
        return kind
    return kind + ":" + ",".join(f"{key}={value:g}" for key, value in params.items())


def candidate_rules() -> list[dict]:
    rules: list[dict] = [
        {"kind": "hard_cascade"},
        {"kind": "soft_multiply"},
    ]

    for gamma in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        rules.append({"kind": "power_gate", "gamma": gamma})

    for scale in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]:
        rules.append({"kind": "negative_scale", "scale": scale})

    for threshold in np.round(np.arange(0.05, 0.50, 0.05), 2):
        rules.append({"kind": "threshold_router", "threshold": float(threshold)})
    for threshold in np.round(np.arange(0.50, 0.991, 0.01), 2):
        rules.append({"kind": "threshold_router", "threshold": float(threshold)})

    for margin in np.round(np.arange(-0.20, 0.801, 0.02), 2):
        rules.append({"kind": "margin_router", "margin": float(margin)})

    for ratio in [0.50, 0.65, 0.80, 0.90, 1.0, 1.10, 1.25, 1.50, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0]:
        rules.append({"kind": "ratio_router", "ratio": ratio})

    for s1_temp in [0.40, 0.50, 0.65, 0.75, 0.90, 1.0, 1.15, 1.30, 1.50, 2.0, 3.0]:
        for s2_temp in [0.50, 0.75, 1.0, 1.50, 2.0]:
            rules.append({"kind": "calibrated_soft", "s1_temp": s1_temp, "s2_temp": s2_temp})

    for s1_temp in [0.40, 0.50, 0.65, 0.75, 0.90, 1.0, 1.15, 1.30, 1.50, 2.0, 3.0]:
        for threshold in np.round(np.arange(0.50, 0.991, 0.05), 2):
            rules.append(
                {
                    "kind": "calibrated_threshold_router",
                    "s1_temp": s1_temp,
                    "threshold": float(threshold),
                }
            )

    for s2_threshold in [0.75, 0.8, 0.85, 0.9, 0.95, 0.98]:
        for s1_max_conf in [0.55, 0.65, 0.75, 0.85, 0.95]:
            for neg_floor in [0.05, 0.10, 0.20, 0.30]:
                rules.append(
                    {
                        "kind": "stage2_override",
                        "s2_threshold": s2_threshold,
                        "s1_max_conf": s1_max_conf,
                        "neg_floor": neg_floor,
                    }
                )

    for s1_max_conf in [0.45, 0.55, 0.65, 0.75, 0.85]:
        for s2_max_conf in [0.55, 0.65, 0.75, 0.85]:
            rules.append(
                {
                    "kind": "negative_rollback",
                    "s1_max_conf": s1_max_conf,
                    "s2_max_conf": s2_max_conf,
                }
            )

    return rules


def score_predictions(y_true: list[str], y_pred: list[str]) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0),
    }


def evaluate_rules(df: pd.DataFrame, rules: list[dict], split: str) -> pd.DataFrame:
    split_df = df[df["split"] == split].reset_index(drop=True)
    y_true = split_df["true_label"].astype(str).tolist()
    rows = []
    for rule in rules:
        preds = predict_rule(split_df, rule)
        metrics = score_predictions(y_true, preds)
        rows.append(
            {
                "rule": rule_name(rule),
                "kind": rule["kind"],
                "params_json": json.dumps({k: v for k, v in rule.items() if k != "kind"}, ensure_ascii=False),
                "split": split,
                "n": len(split_df),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Small-model features derived only from existing model probabilities."""
    features = pd.DataFrame(
        {
            "s1_positive": df["s1_positive"].astype(float),
            "s1_neutral": df["s1_neutral"].astype(float),
            "s1_negative": df["s1_negative"].astype(float),
            "s1_conf": df["s1_conf"].astype(float),
            "s2_constructive": df["s2_constructive"].astype(float),
            "s2_toxic": df["s2_toxic"].astype(float),
            "s2_conf": df["s2_conf"].astype(float),
        }
    )
    features["joint_constructive"] = features["s1_negative"] * features["s2_constructive"]
    features["joint_toxic"] = features["s1_negative"] * features["s2_toxic"]
    features["best_non_negative"] = features[["s1_positive", "s1_neutral"]].max(axis=1)
    features["negative_margin"] = features["s1_negative"] - features["best_non_negative"]
    features["stage2_margin"] = (features["s2_constructive"] - features["s2_toxic"]).abs()
    features["negative_to_non_negative_ratio"] = features["s1_negative"] / (
        features["best_non_negative"] + EPS
    )
    return features


def tiny_model_candidates() -> list[tuple[str, object]]:
    return [
        (
            "tiny_logistic_regression_balanced",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ),
        (
            "tiny_logistic_regression",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, random_state=42),
            ),
        ),
        (
            "tiny_decision_tree_depth2",
            DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=42),
        ),
        (
            "tiny_decision_tree_depth3",
            DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42),
        ),
        (
            "tiny_decision_tree_depth4",
            DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=42),
        ),
        (
            "tiny_random_forest_depth3",
            RandomForestClassifier(
                n_estimators=50,
                max_depth=3,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]


def evaluate_tiny_models(df: pd.DataFrame, train_split: str, eval_split: str) -> pd.DataFrame:
    train_df = df[df["split"] == train_split].reset_index(drop=True)
    eval_df = df[df["split"] == eval_split].reset_index(drop=True)
    if train_df.empty:
        print(f"Skipping tiny models: split '{train_split}' is not in the probability cache.", flush=True)
        return pd.DataFrame()
    if eval_df.empty:
        print(f"Skipping tiny models: split '{eval_split}' is not in the probability cache.", flush=True)
        return pd.DataFrame()

    x_train = feature_frame(train_df)
    y_train = train_df["true_label"].astype(str)
    x_eval = feature_frame(eval_df)
    y_eval = eval_df["true_label"].astype(str).tolist()

    rows = []
    for name, model in tiny_model_candidates():
        model.fit(x_train, y_train)
        pred = model.predict(x_eval).tolist()
        metrics = score_predictions(y_eval, pred)
        rows.append(
            {
                "model": name,
                "train_split": train_split,
                "eval_split": eval_split,
                "n_train": len(train_df),
                "n_eval": len(eval_df),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["macro_f1", "accuracy", "weighted_f1"],
        ascending=False,
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--tune-split", default="val")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--limit-per-split", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--tiny-train-split", default="val")
    parser.add_argument("--skip-tiny-models", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_or_build_cache(args)
    df["split"] = df["split"].astype(str).str.lower()

    rules = candidate_rules()
    print(f"Testing {len(rules)} no-retraining decision rules", flush=True)
    val_results = evaluate_rules(df, rules, args.tune_split.lower())
    val_results = val_results.sort_values(
        ["macro_f1", "accuracy", "weighted_f1"],
        ascending=False,
    ).reset_index(drop=True)

    top_rules = []
    for _, row in val_results.head(args.top_k).iterrows():
        params = json.loads(row["params_json"])
        top_rules.append({"kind": row["kind"], **params})

    test_results = evaluate_rules(df, top_rules, args.eval_split.lower())
    test_results = test_results.sort_values(
        ["macro_f1", "accuracy", "weighted_f1"],
        ascending=False,
    ).reset_index(drop=True)

    val_path = out_dir / "two_stage_rule_search_validation.csv"
    test_path = out_dir / "two_stage_rule_search_test_top.csv"
    val_results.to_csv(val_path, index=False, encoding="utf-8-sig")
    test_results.to_csv(test_path, index=False, encoding="utf-8-sig")

    tiny_results = pd.DataFrame()
    if not args.skip_tiny_models:
        tiny_results = evaluate_tiny_models(
            df,
            train_split=args.tiny_train_split.lower(),
            eval_split=args.eval_split.lower(),
        )
        if not tiny_results.empty:
            tiny_results.to_csv(
                out_dir / "two_stage_tiny_model_results.csv",
                index=False,
                encoding="utf-8-sig",
            )

    best_rule = top_rules[0]
    eval_df = df[df["split"] == args.eval_split.lower()].reset_index(drop=True)
    y_true = eval_df["true_label"].astype(str).tolist()
    y_pred = predict_rule(eval_df, best_rule)
    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )
    (out_dir / "two_stage_best_rule_report.json").write_text(
        json.dumps(
            {
                "selected_on": args.tune_split.lower(),
                "evaluated_on": args.eval_split.lower(),
                "best_rule": best_rule,
                "best_rule_name": rule_name(best_rule),
                "classification_report": report,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nBest validation rules", flush=True)
    print(val_results.head(args.top_k).to_string(index=False), flush=True)
    print("\nTest results for validation-selected rules", flush=True)
    print(test_results.to_string(index=False), flush=True)
    if not tiny_results.empty:
        print("\nTiny decision model results", flush=True)
        print(tiny_results.to_string(index=False), flush=True)
    print(f"\nSaved outputs to {out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
