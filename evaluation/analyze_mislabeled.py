import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import LABEL_NAMES, REPORTS_DIR, FIGURES_DIR

V1_PATH = os.path.join(REPORTS_DIR, "mislabeled_rows.csv")
V2_PATH = os.path.join(REPORTS_DIR, "mislabeled_rows_v2.csv")

TEXT_PREVIEW_LEN = 100

def load_mislabeled(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    for col in ["id", "split", "true_label", "predicted_label", "source"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    if "confidence" in df.columns:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    return df

def compute_overlap(v1, v2):
    v1_keys = set(zip(v1["split"], v1["id"]))
    v2_keys = set(zip(v2["split"], v2["id"]))

    overlap_keys = v1_keys & v2_keys
    only_v1_keys = v1_keys - v2_keys
    only_v2_keys = v2_keys - v1_keys

    def filter_by_keys(df, keys):
        mask = df.apply(lambda r: (r["split"], r["id"]) in keys, axis=1)
        return df[mask].copy()

    return (
        filter_by_keys(v1, overlap_keys),
        filter_by_keys(v2, overlap_keys),
        filter_by_keys(v1, only_v1_keys),
        filter_by_keys(v2, only_v2_keys),
    )

def confusion_counts(df):
    return Counter(zip(df["true_label"], df["predicted_label"]))

def plot_error_count_comparison(v1, v2):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    v1_counts = v1["true_label"].value_counts().reindex(LABEL_NAMES, fill_value=0)
    v2_counts = v2["true_label"].value_counts().reindex(LABEL_NAMES, fill_value=0)

    x = np.arange(len(LABEL_NAMES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, v1_counts.values, width, label="V1", color="#2196F3")
    bars2 = ax.bar(x + width / 2, v2_counts.values, width, label="V2", color="#FF9800")

    ax.set_xlabel("True Label")
    ax.set_ylabel("Mislabeled Count")
    ax.set_title("Mislabeled Rows by True Label — V1 vs V2")
    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_NAMES, rotation=15)
    ax.legend()

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "mislabeled_v1_v2_by_class.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path

def plot_confusion_heatmaps(v1, v2):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, df, title in [(axes[0], v1, "V1 Confusion"), (axes[1], v2, "V2 Confusion")]:
        matrix = pd.crosstab(
            df["true_label"], df["predicted_label"],
        ).reindex(index=LABEL_NAMES, columns=LABEL_NAMES, fill_value=0)
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Oranges", ax=ax,
                    xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "mislabeled_v1_v2_confusion.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path

def plot_confidence_distributions(v1, v2):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, df, title, color in [
        (axes[0], v1, "V1 Confidence", "#2196F3"),
        (axes[1], v2, "V2 Confidence", "#FF9800"),
    ]:
        vals = df["confidence"].dropna()
        ax.hist(vals, bins=20, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(vals.median(), color="red", linestyle="--", label=f"median={vals.median():.3f}")
        ax.set_title(f"{title} (mislabeled rows)")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Count")
        ax.legend()

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "mislabeled_v1_v2_confidence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path

def plot_overlap_venn_simple(n_v1_only, n_overlap, n_v2_only):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    categories = ["Only V1\n(V2 fixed)", "Both V1 & V2\n(persistent)", "Only V2\n(V2 introduced)"]
    values = [n_v1_only, n_overlap, n_v2_only]
    colors = ["#4CAF50", "#F44336", "#FF9800"]

    bars = ax.bar(categories, values, color=colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(val), ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("Number of Mislabeled Rows")
    ax.set_title("Mislabeled Row Overlap — V1 vs V2")
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "mislabeled_v1_v2_overlap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path

def build_overlap_csv(v1_overlap, v2_overlap):
    merged = v1_overlap.merge(
        v2_overlap,
        on=["split", "id"],
        suffixes=("_v1", "_v2"),
    )
    keep_cols = [
        "split", "id",
        "text_raw_v1",
        "true_label_v1",
        "predicted_label_v1", "confidence_v1",
        "predicted_label_v2", "confidence_v2",
        "source_v1",
    ]
    keep_cols = [c for c in keep_cols if c in merged.columns]
    result = merged[keep_cols].copy()
    result = result.rename(columns={
        "text_raw_v1": "text_raw",
        "true_label_v1": "true_label",
        "source_v1": "source",
    })
    result["same_prediction"] = result["predicted_label_v1"] == result["predicted_label_v2"]
    return result

def generate_report(v1, v2, v1_overlap, v2_overlap, only_v1, only_v2):
    md = []
    md.append("# Mislabeled Rows Analysis — V1 vs V2\n")
    md.append(f"**V1 file:** `mislabeled_rows.csv` ({len(v1)} rows)  ")
    md.append(f"**V2 file:** `mislabeled_rows_v2.csv` ({len(v2)} rows)  ")
    md.append(f"**Labels:** {', '.join(LABEL_NAMES)}\n")
    md.append("---\n")

    n_overlap = len(v1_overlap)
    n_only_v1 = len(only_v1)
    n_only_v2 = len(only_v2)

    md.append("## 1. Overview\n")
    md.append(f"| Metric | Count |")
    md.append(f"|--------|-------|")
    md.append(f"| V1 total mislabeled | {len(v1)} |")
    md.append(f"| V2 total mislabeled | {len(v2)} |")
    md.append(f"| Overlap (mislabeled in both) | {n_overlap} |")
    md.append(f"| Only V1 (V2 fixed these) | {n_only_v1} |")
    md.append(f"| Only V2 (V2 introduced these) | {n_only_v2} |")
    md.append(f"| Net change (V2 − V1) | {len(v2) - len(v1):+d} |")
    md.append("")

    pct_fixed = n_only_v1 / max(len(v1), 1) * 100
    pct_new = n_only_v2 / max(len(v2), 1) * 100
    pct_persist = n_overlap / max(len(v1), 1) * 100
    md.append(f"- **{pct_fixed:.1f}%** of V1 errors were fixed in V2")
    md.append(f"- **{pct_new:.1f}%** of V2 errors are new (not in V1)")
    md.append(f"- **{pct_persist:.1f}%** of V1 errors persist in V2\n")

    md.append("## 2. Per-Split Breakdown\n")
    md.append("| Split | V1 Errors | V2 Errors | Overlap | Only V1 | Only V2 |")
    md.append("|-------|-----------|-----------|---------|---------|---------|")
    for split in ["val", "test"]:
        v1_s = len(v1[v1["split"] == split])
        v2_s = len(v2[v2["split"] == split])
        ov_s = len(v1_overlap[v1_overlap["split"] == split])
        o1_s = len(only_v1[only_v1["split"] == split])
        o2_s = len(only_v2[only_v2["split"] == split])
        md.append(f"| {split} | {v1_s} | {v2_s} | {ov_s} | {o1_s} | {o2_s} |")
    md.append("")

    md.append("## 3. Per-Class Mislabeled Counts (by True Label)\n")
    md.append("| True Label | V1 | V2 | Delta | Only V1 | Only V2 | Overlap |")
    md.append("|------------|----|----|-------|---------|---------|---------|")
    for lbl in LABEL_NAMES:
        c1 = len(v1[v1["true_label"] == lbl])
        c2 = len(v2[v2["true_label"] == lbl])
        delta = c2 - c1
        sign = "+" if delta >= 0 else ""
        ov = len(v1_overlap[v1_overlap["true_label"] == lbl])
        o1 = len(only_v1[only_v1["true_label"] == lbl])
        o2 = len(only_v2[only_v2["true_label"] == lbl])
        md.append(f"| {lbl} | {c1} | {c2} | {sign}{delta} | {o1} | {o2} | {ov} |")
    md.append("")

    md.append("## 4. Confusion Pair Comparison\n")
    v1_conf = confusion_counts(v1)
    v2_conf = confusion_counts(v2)
    all_pairs = sorted(set(v1_conf.keys()) | set(v2_conf.keys()),
                       key=lambda p: -(v1_conf.get(p, 0) + v2_conf.get(p, 0)))

    md.append("| True Label | Predicted As | V1 | V2 | Delta |")
    md.append("|------------|-------------|----|----|-------|")
    for true, pred in all_pairs[:15]:
        c1 = v1_conf.get((true, pred), 0)
        c2 = v2_conf.get((true, pred), 0)
        delta = c2 - c1
        sign = "+" if delta >= 0 else ""
        md.append(f"| {true} | {pred} | {c1} | {c2} | {sign}{delta} |")
    md.append("")

    md.append("## 5. Overlap Analysis — Persistent Errors\n")
    if n_overlap > 0:
        overlap_merged = v1_overlap.merge(
            v2_overlap, on=["split", "id"], suffixes=("_v1", "_v2"),
        )
        same_pred = (overlap_merged["predicted_label_v1"] == overlap_merged["predicted_label_v2"]).sum()
        diff_pred = n_overlap - same_pred
        md.append(f"Of the **{n_overlap}** rows mislabeled in both V1 and V2:\n")
        md.append(f"- **{same_pred}** ({same_pred/n_overlap*100:.1f}%) got the **same wrong prediction**")
        md.append(f"- **{diff_pred}** ({diff_pred/n_overlap*100:.1f}%) got a **different wrong prediction**\n")

        if diff_pred > 0:
            diff_rows = overlap_merged[
                overlap_merged["predicted_label_v1"] != overlap_merged["predicted_label_v2"]
            ]
            md.append("### Rows with Different Wrong Predictions\n")
            md.append("| ID | True | V1 Predicted | V2 Predicted | V1 Conf | V2 Conf |")
            md.append("|----|------|-------------|-------------|---------|---------|")
            for _, r in diff_rows.head(20).iterrows():
                md.append(
                    f"| {r['id']} | {r['true_label_v1']} | "
                    f"{r['predicted_label_v1']} | {r['predicted_label_v2']} | "
                    f"{r['confidence_v1']:.4f} | {r['confidence_v2']:.4f} |"
                )
            if len(diff_rows) > 20:
                md.append(f"\n*(showing 20 of {len(diff_rows)} rows)*")
            md.append("")

        md.append("### Persistent Error Confusion Pairs\n")
        persist_conf = confusion_counts(v2_overlap)
        md.append("| True Label | Predicted As (V2) | Count |")
        md.append("|------------|-------------------|-------|")
        for (true, pred), count in sorted(persist_conf.items(), key=lambda x: -x[1])[:10]:
            md.append(f"| {true} | {pred} | {count} |")
        md.append("")
    else:
        md.append("No overlapping mislabeled rows found.\n")

    md.append("## 6. Errors Fixed by V2 (Only in V1)\n")
    if n_only_v1 > 0:
        md.append(f"V2 correctly classified **{n_only_v1}** rows that V1 got wrong.\n")
        v1_only_conf = confusion_counts(only_v1)
        md.append("### Confusion Pairs (V1 errors that V2 fixed)\n")
        md.append("| True Label | V1 Predicted As | Count |")
        md.append("|------------|----------------|-------|")
        for (true, pred), count in sorted(v1_only_conf.items(), key=lambda x: -x[1])[:10]:
            md.append(f"| {true} | {pred} | {count} |")
        md.append("")

        md.append("### Sample Texts (V2 fixed these)\n")
        for _, r in only_v1.head(10).iterrows():
            text = str(r.get("text_raw", ""))[:TEXT_PREVIEW_LEN]
            if len(str(r.get("text_raw", ""))) > TEXT_PREVIEW_LEN:
                text += "..."
            md.append(
                f"- **[{r['true_label']}→{r['predicted_label']}]** `{text}`"
            )
        md.append("")
    else:
        md.append("No errors were fixed by V2.\n")

    md.append("## 7. New Errors in V2 (Only in V2)\n")
    if n_only_v2 > 0:
        md.append(f"V2 introduced **{n_only_v2}** new mislabeled rows not present in V1.\n")
        v2_only_conf = confusion_counts(only_v2)
        md.append("### Confusion Pairs (new V2 errors)\n")
        md.append("| True Label | V2 Predicted As | Count |")
        md.append("|------------|----------------|-------|")
        for (true, pred), count in sorted(v2_only_conf.items(), key=lambda x: -x[1])[:10]:
            md.append(f"| {true} | {pred} | {count} |")
        md.append("")

        md.append("### Sample Texts (V2 introduced these errors)\n")
        for _, r in only_v2.head(10).iterrows():
            text = str(r.get("text_raw", ""))[:TEXT_PREVIEW_LEN]
            if len(str(r.get("text_raw", ""))) > TEXT_PREVIEW_LEN:
                text += "..."
            md.append(
                f"- **[{r['true_label']}→{r['predicted_label']}]** `{text}`"
            )
        md.append("")
    else:
        md.append("V2 did not introduce any new errors.\n")

    md.append("## 8. Confidence Score Comparison\n")
    v1_med = v1["confidence"].median()
    v2_med = v2["confidence"].median()
    v1_mean = v1["confidence"].mean()
    v2_mean = v2["confidence"].mean()

    md.append("| Statistic | V1 | V2 |")
    md.append("|-----------|----|----|")
    md.append(f"| Mean confidence | {v1_mean:.4f} | {v2_mean:.4f} |")
    md.append(f"| Median confidence | {v1_med:.4f} | {v2_med:.4f} |")
    md.append(f"| Min confidence | {v1['confidence'].min():.4f} | {v2['confidence'].min():.4f} |")
    md.append(f"| Max confidence | {v1['confidence'].max():.4f} | {v2['confidence'].max():.4f} |")
    md.append("")

    if n_overlap > 0:
        overlap_merged = v1_overlap.merge(
            v2_overlap, on=["split", "id"], suffixes=("_v1", "_v2"),
        )
        v1_ov_conf = overlap_merged["confidence_v1"].mean()
        v2_ov_conf = overlap_merged["confidence_v2"].mean()
        md.append(f"For the **{n_overlap}** persistent errors, mean confidence: "
                  f"V1={v1_ov_conf:.4f}, V2={v2_ov_conf:.4f}\n")

    if v2_med > v1_med:
        md.append("V2 mislabeled rows have **higher** confidence on average, "
                  "meaning the model is more confidently wrong on its errors.\n")
    elif v2_med < v1_med:
        md.append("V2 mislabeled rows have **lower** confidence on average, "
                  "meaning the model is less sure about its wrong predictions.\n")
    else:
        md.append("V1 and V2 mislabeled rows have similar confidence levels.\n")

    md.append("## 9. Source Breakdown\n")
    if "source" in v1.columns and "source" in v2.columns:
        v1_sources = v1["source"].value_counts().head(10)
        v2_sources = v2["source"].value_counts().head(10)
        all_sources = sorted(
            set(v1_sources.index) | set(v2_sources.index),
            key=lambda s: -(v1_sources.get(s, 0) + v2_sources.get(s, 0)),
        )
        md.append("| Source | V1 Errors | V2 Errors | Delta |")
        md.append("|--------|-----------|-----------|-------|")
        for src in all_sources:
            c1 = v1_sources.get(src, 0)
            c2 = v2_sources.get(src, 0)
            delta = c2 - c1
            sign = "+" if delta >= 0 else ""
            md.append(f"| {src} | {c1} | {c2} | {sign}{delta} |")
        md.append("")

    md.append("---\n")
    md.append("## Summary\n")
    if len(v2) < len(v1):
        md.append(f"V2 has **{len(v1) - len(v2)} fewer** mislabeled rows than V1 "
                  f"({len(v2)} vs {len(v1)}).\n")
    elif len(v2) > len(v1):
        md.append(f"V2 has **{len(v2) - len(v1)} more** mislabeled rows than V1 "
                  f"({len(v2)} vs {len(v1)}).\n")
    else:
        md.append(f"V1 and V2 have the **same number** of mislabeled rows ({len(v1)}).\n")

    md.append(f"- {n_overlap} errors are persistent (appear in both)")
    md.append(f"- {n_only_v1} V1 errors were corrected by V2")
    md.append(f"- {n_only_v2} new errors were introduced in V2\n")

    md.append("---\n")

    return "\n".join(md)

def main():
    print("=" * 60)
    print("  Mislabeled Rows Analysis — V1 vs V2")
    print("=" * 60)

    print(f"\n  V1: {V1_PATH}")
    print(f"  V2: {V2_PATH}")

    v1 = load_mislabeled(V1_PATH)
    v2 = load_mislabeled(V2_PATH)
    print(f"\n  V1: {len(v1)} mislabeled rows")
    print(f"  V2: {len(v2)} mislabeled rows")

    v1_overlap, v2_overlap, only_v1, only_v2 = compute_overlap(v1, v2)
    print(f"\n  Overlap (both):    {len(v1_overlap)}")
    print(f"  Only V1 (fixed):   {len(only_v1)}")
    print(f"  Only V2 (new):     {len(only_v2)}")

    print("\n[analysis] Generating plots ...")
    plot_error_count_comparison(v1, v2)
    plot_confusion_heatmaps(v1, v2)
    plot_confidence_distributions(v1, v2)
    plot_overlap_venn_simple(len(only_v1), len(v1_overlap), len(only_v2))

    os.makedirs(REPORTS_DIR, exist_ok=True)
    overlap_csv = build_overlap_csv(v1_overlap, v2_overlap)
    overlap_path = os.path.join(REPORTS_DIR, "mislabeled_overlap_v1_v2.csv")
    overlap_csv.to_csv(overlap_path, index=False, encoding="utf-8-sig")
    print(f"\n  Saved overlap CSV: {overlap_path} ({len(overlap_csv)} rows)")

    print("\n[analysis] Generating report ...")
    report = generate_report(v1, v2, v1_overlap, v2_overlap, only_v1, only_v2)
    report_path = os.path.join(REPORTS_DIR, "mislabeled_v1_vs_v2_analysis.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved report: {report_path}")

    try:
        print("\n" + report)
    except UnicodeEncodeError:
        print("\n" + report.encode("ascii", errors="replace").decode("ascii"))

    print("\n" + "=" * 60)
    print("  ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
