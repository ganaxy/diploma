"""Evaluate API architectures on the saved test split.

This gives evidence for whether the no-retraining soft two-stage connection is
better than the original hard cascade and the final flat model.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

os.environ.setdefault("GRADIO_FORCE_CPU", "1")

from api_app.main import _predict_text
from gradio_app import config


ARCHITECTURES = ("flat", "two_stage", "soft_two_stage", "strict_two_stage")
DEFAULT_LIMIT = 0


def evaluate_architecture(df: pd.DataFrame, architecture: str) -> tuple[dict, list[dict]]:
    y_true: list[str] = []
    y_pred: list[str] = []
    rows: list[dict] = []
    start = time.perf_counter()
    for idx, row in df.iterrows():
        text = str(row.get("text_light_clean") or row.get("text_raw") or "")
        true_label = str(row["label"]).upper()
        pred = _predict_text(text, architecture, review_threshold=0.70)
        y_true.append(true_label)
        y_pred.append(pred["label"])
        rows.append(
            {
                "id": row.get("id"),
                "true_label": true_label,
                "predicted_label": pred["label"],
                "confidence": pred["confidence"],
                "architecture": architecture,
                "action": pred["action"],
                "latency_ms": pred["latency_ms"],
            }
        )
        if (idx + 1) % 100 == 0:
            print(f"[{architecture}] evaluated {idx + 1}/{len(df)} rows")

    elapsed = time.perf_counter() - start
    report = classification_report(
        y_true,
        y_pred,
        labels=list(config.FLAT_LABELS),
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "architecture": architecture,
        "n": len(df),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=list(config.FLAT_LABELS), average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=list(config.FLAT_LABELS), average="weighted", zero_division=0),
        "elapsed_sec": elapsed,
        "avg_wall_ms_per_row": (elapsed * 1000.0 / len(df)) if len(df) else 0.0,
        "classification_report": report,
    }
    return metrics, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Limit test rows for a quick smoke run. 0 means full test split.")
    parser.add_argument("--architectures", nargs="+", default=list(ARCHITECTURES), choices=list(ARCHITECTURES))
    parser.add_argument("--out-dir", default=str(Path("api_app") / "eval_outputs"))
    args = parser.parse_args()

    test_csv = config.TEST_CSV
    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    df = pd.read_csv(test_csv, encoding="utf-8-sig")
    df = df[df["split"].astype(str).str.lower() == "test"].reset_index(drop=True)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = []

    print(f"Evaluating {len(df)} test rows from {test_csv}")
    print(f"Architectures: {', '.join(args.architectures)}")
    for architecture in args.architectures:
        metrics, rows = evaluate_architecture(df, architecture)
        all_metrics.append(metrics)
        pd.DataFrame(rows).to_csv(out_dir / f"predictions_{architecture}.csv", index=False, encoding="utf-8-sig")
        print(
            f"{architecture:>14s}: "
            f"accuracy={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"weighted_f1={metrics['weighted_f1']:.4f}"
        )

    summary = [
        {
            "architecture": m["architecture"],
            "n": m["n"],
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
            "avg_wall_ms_per_row": m["avg_wall_ms_per_row"],
        }
        for m in all_metrics
    ]
    (out_dir / "architecture_metrics.json").write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(summary).to_csv(out_dir / "architecture_summary.csv", index=False, encoding="utf-8-sig")

    print("\nSummary")
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"\nSaved outputs to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
