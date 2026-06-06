"""Gradio inference UI for the Mongolian comment classifier.

Two architectures (Flat 4-class MN-BERT vs. Two-stage Stage1 -> Stage2) are loaded
at startup and toggled live so the demo can demonstrate the error-propagation
discussion from Chapter 6 directly. Both single-text and batch-CSV tabs share
the same toggle.

Run:    python -m gradio_app.app
or:     cd gradio_app && python app.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

import gradio as gr
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import pandas as pd

# Allow `python app.py` from inside the gradio_app folder.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from gradio_app import config
    from gradio_app.explain import token_attributions
    from gradio_app.models import (
        DeviceSuspendedError, FlatClassifier, PipelineResult,
        TwoStageClassifier, pick_device,
    )
    from gradio_app.preprocess import language_warning
else:
    from . import config
    from .explain import token_attributions
    from .models import (
        DeviceSuspendedError, FlatClassifier, PipelineResult,
        TwoStageClassifier, pick_device,
    )
    from .preprocess import language_warning


# ── UI labels ───────────────────────────────────────────────────────────────
ARCH_FLAT = "Flat — нэг шатлалт (4 ангилал)"
ARCH_TWO_STAGE = "Two-stage — хоёр шатлалт (Stage 1 → Stage 2)"
ARCH_CHOICES = [ARCH_FLAT, ARCH_TWO_STAGE]


# ── Model lifetime: load once at startup ────────────────────────────────────
print("=" * 60)
print(" Mongolian classifier — Gradio inference app")
print("=" * 60)
DEVICE, DEVICE_LABEL = pick_device()
print(f"[startup] Device: {DEVICE_LABEL}")

print(f"[startup] Loading Flat 4-class model from {config.FLAT_MODEL_DIR} ...")
FLAT_MODEL = FlatClassifier(DEVICE)
print("[startup] Flat model ready.")

print(f"[startup] Loading Stage 1 model from {config.STAGE1_MODEL_DIR} ...")
print(f"[startup] Loading Stage 2 model from {config.STAGE2_MODEL_DIR} ...")
TWO_STAGE_MODEL = TwoStageClassifier(DEVICE)
print("[startup] Two-stage pipeline ready.")
print("=" * 60)


DEVICE_SUSPEND_MSG = (
    "GPU төхөөрөмж түр зогссон байна (torch-directml). "
    "Серверийг дахин эхлүүлнэ үү, эсвэл `GRADIO_FORCE_CPU=1` орчны хувьсагчтай "
    "ажиллуулна уу (CPU дээр тогтвортой, ~200 мс/текст)."
)


def _classify(text: str, architecture: str) -> PipelineResult:
    """Dispatch to the chosen model. Each model applies its own
    training-faithful input construction, so RAW text is passed through."""
    if architecture == ARCH_TWO_STAGE:
        return TWO_STAGE_MODEL.predict(text)
    return FLAT_MODEL.predict(text)


# ── Probability projection for the unified 4-class chart ────────────────────

def _four_class_view(result: PipelineResult) -> dict:
    """Project either pipeline output onto the same 4-class axis for plotting.

    For Flat: returns its native distribution unchanged.
    For Two-stage: P(POSITIVE)/P(NEUTRAL) come from Stage 1; the Stage 1
        NEGATIVE mass is split between TOXIC/CONSTRUCTIVE using Stage 2 (if it
        fired) or 50/50 otherwise.
    """
    if result.flat_probs is not None:
        return {lbl: float(result.flat_probs.get(lbl, 0.0)) for lbl in config.FLAT_LABELS}

    s1 = result.stage1.probs
    p_pos = s1.get("POSITIVE", 0.0)
    p_neu = s1.get("NEUTRAL", 0.0)
    p_neg = s1.get("NEGATIVE", 0.0)
    if result.stage2 is not None:
        p_tox = p_neg * result.stage2.probs.get("TOXIC", 0.0)
        p_con = p_neg * result.stage2.probs.get("CONSTRUCTIVE", 0.0)
    else:
        p_tox = p_neg / 2
        p_con = p_neg / 2
    return {"POSITIVE": p_pos, "NEUTRAL": p_neu, "TOXIC": p_tox, "CONSTRUCTIVE": p_con}


def _bar_chart(probs: dict, title: str) -> Figure:
    """Horizontal bar chart of the 4-class projection.

    Uses config.FLAT_LABELS as the canonical class order so this chart matches
    the order shown in the decision-trace panel and the batch distribution plot.
    Built via the Figure() constructor (not pyplot) to avoid pyplot's global
    figure registry, which would leak one entry per prediction.
    """
    values = [probs.get(l, 0.0) for l in config.FLAT_LABELS]
    display = [config.LABEL_MN[l] for l in config.FLAT_LABELS]
    colors = [config.LABEL_COLOR[l] for l in config.FLAT_LABELS]

    fig = Figure(figsize=(6.4, 3.4))
    ax = fig.add_subplot(111)
    bars = ax.barh(display, values, color=colors)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Магадлал")
    ax.set_title(title)
    ax.invert_yaxis()
    for bar, v in zip(bars, values):
        ax.text(min(v + 0.01, 0.95), bar.get_y() + bar.get_height() / 2,
                f"{v:.1%}", va="center", fontsize=9)
    fig.tight_layout()
    return fig


def _explanation_markdown(text: str, architecture: str, result: PipelineResult,
                          warning: str, latency_ms: float) -> str:
    """Build the decision-trace markdown panel."""
    lines: List[str] = []
    if warning:
        lines.append(f"> {warning}\n")
    lines.append(f"**Архитектур:** {architecture}")
    lines.append(f"**Төхөөрөмж:** `{DEVICE_LABEL}` · **Хариу хугацаа:** {latency_ms:.0f} мс\n")

    final_mn = config.LABEL_MN.get(result.final_label, result.final_label)
    lines.append(f"**Эцсийн ангилал:** **{final_mn}** "
                 f"(итгэл {result.final_confidence:.1%})\n")

    if architecture == ARCH_FLAT:
        lines.append("### Flat MN-BERT — нэг шатлалт инференс")
        lines.append("4-ангиллын softmax магадлал:")
        for en in config.FLAT_LABELS:
            mn = config.LABEL_MN[en]
            p = result.flat_probs.get(en, 0.0)
            lines.append(f"- {mn} (`{en}`): {p:.2%}")
    else:
        lines.append("### Stage 1 — 3-ангилал (Эерэг / Саармаг / Сөрөг)")
        for en in config.STAGE1_LABELS:
            mn = config.LABEL_MN[en]
            p = result.stage1.probs.get(en, 0.0)
            lines.append(f"- {mn} (`{en}`): {p:.2%}")
        lines.append("")
        if result.branched_to_stage2:
            lines.append("Stage 1 нь **Сөрөг (NEGATIVE)** гэж таамагласан тул Stage 2 руу салаалав.")
            lines.append("")
            lines.append("### Stage 2 — 2-ангилал (Хортой сөрөг / Бүтээлч шүүмжлэл)")
            for en in config.STAGE2_LABELS:
                mn = config.LABEL_MN[en]
                p = result.stage2.probs.get(en, 0.0)
                lines.append(f"- {mn} (`{en}`): {p:.2%}")
            lines.append("")
            lines.append(
                "**Нэгтгэсэн итгэл = P(NEGATIVE | Stage 1) × P(эцсийн | Stage 2) "
                f"= {result.stage1.confidence:.2%} × {result.stage2.confidence:.2%} "
                f"= {result.final_confidence:.2%}**"
            )
            lines.append("")
            lines.append(
                "_Үржвэр учраас нэгтгэсэн итгэл нь хоёр шатны итгэлээс тус бүр доогуур "
                "гарна — энэ нь бүлэг 6-д тайлбарласан 'алдааны нийлбэр' (error compounding) "
                "үзэгдэл юм._"
            )
        else:
            lines.append("Stage 1 нь **Сөрөг биш** ангилал сонгосон тул Stage 2 ажиллаагүй.")

    lines.append("")
    lines.append("---")
    lines.append(
        "_Тайлбар: магадлалын утгууд нь softmax-ын түүхий гаралт бөгөөд "
        "калибрчлагдаагүй (бодит магадлал биш). Үгсийн тодруулга нь "
        "Input×Gradient-д суурилсан ойролцоо ач холбогдол бөгөөд "
        "тайлбарлан үзүүлэх зорилготой._"
    )
    return "\n".join(lines)


def _highlights_for(result: PipelineResult, text: str, architecture: str
                    ) -> List[Tuple[str, float]]:
    """Attribute against the model that produced the final label. Raw text is
    passed; token_attributions rebuilds the model's faithful input itself."""
    if architecture == ARCH_FLAT:
        return token_attributions(FLAT_MODEL.model, text, result.final_label)
    if result.branched_to_stage2:
        return token_attributions(TWO_STAGE_MODEL.stage2, text, result.final_label)
    return token_attributions(TWO_STAGE_MODEL.stage1, text, result.final_label)


# ── Single-text handler ─────────────────────────────────────────────────────

def _placeholder_fig(msg: str) -> Figure:
    fig = Figure(figsize=(6.4, 3.4))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=11,
            color="#b00", wrap=True)
    return fig


def _error_outputs(msg: str):
    """4-tuple matching predict_single's outputs, for a graceful failure."""
    return (f"### Алдаа\n{msg}", _placeholder_fig("⚠"), [], f"> {msg}")


def predict_single(text: str, architecture: str):
    text = (text or "").strip()
    warning = language_warning(text)
    if not text:
        empty_fig = Figure(figsize=(6.4, 3.4))
        ax = empty_fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, "Текст оруулна уу", ha="center", va="center",
                fontsize=12, color="#888")
        return ("### Текст оруулна уу", empty_fig, [], "_Текст оруулаагүй байна._")

    t0 = time.perf_counter()
    try:
        result = _classify(text, architecture)
    except DeviceSuspendedError:
        return _error_outputs(DEVICE_SUSPEND_MSG)
    except Exception as exc:  # noqa: BLE001 — never show a raw traceback in a demo
        return _error_outputs(f"Алдаа гарлаа: {exc!r}")
    try:
        highlights = _highlights_for(result, text, architecture)
    except Exception as exc:  # gradient pass can fail on some DirectML builds
        highlights = [(text, 0.0)]
        print(f"[explain] gradient attribution failed: {exc!r}")
    latency_ms = (time.perf_counter() - t0) * 1000.0

    final_mn = config.LABEL_MN.get(result.final_label, result.final_label)
    headline = f"### {final_mn}\n**Итгэл:** {result.final_confidence:.1%}  ·  `{result.final_label}`"

    if architecture == ARCH_FLAT:
        chart_title = "4-ангиллын магадлал (Flat)"
    else:
        chart_title = "Нэгтгэсэн 4-ангиллын магадлал (Two-stage)"
    fig = _bar_chart(_four_class_view(result), chart_title)

    explanation = _explanation_markdown(text, architecture, result, warning, latency_ms)
    return headline, fig, highlights, explanation


# ── Batch handler ───────────────────────────────────────────────────────────

def _detect_text_column(df: pd.DataFrame) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for cand in config.TEXT_COLUMN_CANDIDATES:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def _class_distribution_chart(labels: List[str]) -> Figure:
    counts = {l: 0 for l in config.FLAT_LABELS}
    for l in labels:
        if l in counts:
            counts[l] += 1
    display = [config.LABEL_MN[l] for l in config.FLAT_LABELS]
    values = [counts[l] for l in config.FLAT_LABELS]
    colors = [config.LABEL_COLOR[l] for l in config.FLAT_LABELS]

    fig = Figure(figsize=(6.4, 3.4))
    ax = fig.add_subplot(111)
    bars = ax.bar(display, values, color=colors)
    ax.set_ylabel("Тоо")
    ax.set_title("Багц таамаглалын ангиллын тархалт")
    for bar, v in zip(bars, values):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, v, str(v),
                    ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig


def predict_batch(file, architecture: str):
    if file is None:
        return ("CSV файл оруулна уу.", None, None, None)

    try:
        df = pd.read_csv(file.name)
    except UnicodeDecodeError:
        df = pd.read_csv(file.name, encoding="utf-8-sig")
    except Exception as exc:
        return (f"CSV уншилт амжилтгүй: {exc}", None, None, None)

    text_col = _detect_text_column(df)
    if text_col is None:
        cols = ", ".join(df.columns)
        return (f"`text` эсвэл `comment` багана олдсонгүй. Олдсон баганууд: {cols}",
                None, None, None)

    if len(df) == 0:
        return ("CSV хоосон байна — мөр алга.", None, None, None)

    texts = df[text_col].fillna("").astype(str).tolist()
    rows = []
    eng_labels = []
    t0 = time.perf_counter()
    try:
        for raw in texts:
            result = _classify(raw, architecture)
            eng_labels.append(result.final_label)
            rows.append({
                "text": raw,
                "predicted_label_en": result.final_label,
                "predicted_label_mn": config.LABEL_MN.get(
                    result.final_label, result.final_label),
                "confidence": round(result.final_confidence, 4),
            })
    except DeviceSuspendedError:
        return (DEVICE_SUSPEND_MSG, None, None, None)
    except Exception as exc:  # noqa: BLE001
        return (f"Багц боловсруулалт амжилтгүй: {exc!r}", None, None, None)
    elapsed = time.perf_counter() - t0

    out_df = pd.DataFrame(rows)
    # tempfile so the download path doesn't depend on cwd and never collides
    # with a file the user might already have open in Excel.
    out_handle = tempfile.NamedTemporaryFile(
        prefix="gradio_batch_predictions_", suffix=".csv",
        delete=False, mode="w", encoding="utf-8-sig", newline="",
    )
    out_handle.close()
    out_df.to_csv(out_handle.name, index=False, encoding="utf-8-sig")

    per_row_ms = 1000 * elapsed / len(rows)
    status = (f"**Амжилттай:** {len(rows)} мөр боловсруулав "
              f"({elapsed:.2f} сек, {per_row_ms:.1f} мс/мөр).  "
              f"Архитектур: `{architecture}`.")
    return status, out_df, _class_distribution_chart(eng_labels), out_handle.name


# ── B: Random held-out test sample (shows ground truth, including failures) ──

_TEST_DF = None


def _load_test_df():
    global _TEST_DF
    if _TEST_DF is None:
        df = pd.read_csv(config.TEST_CSV, encoding="utf-8-sig")
        _TEST_DF = df[df["split"] == "test"].reset_index(drop=True)
    return _TEST_DF


def random_test_sample(architecture: str):
    """Pull a random held-out test row, classify it, and reveal the dataset's
    true label so the demo can show honest failures, not only wins.
    Returns: text_in, headline, bar_plot, highlights, explanation, truth_md."""
    try:
        df = _load_test_df()
        row = df.sample(1).iloc[0]
    except Exception as exc:  # noqa: BLE001
        return ("", "### Алдаа", _placeholder_fig("⚠"), [],
                f"> Тест олонлог уншиж чадсангүй: {exc!r}", "")

    text = str(row["text_light_clean"])
    headline, fig, highlights, explanation = predict_single(text, architecture)

    # Recover predicted English label from the headline's backtick code.
    pred_en = headline.split("`")[1] if "`" in headline else "?"
    true_en = str(row["label"])
    ok = "✓ зөв таамагласан" if pred_en == true_en else "✗ буруу таамагласан"
    truth = [
        "#### Бодит шошго (held-out тест олонлог)",
        f"- **Жинхэнэ ангилал:** {config.LABEL_MN.get(true_en, true_en)} "
        f"(`{true_en}`)",
        f"- **Үр дүн:** {ok}",
        "",
        "_Санамсаргүй сонголт тул буруу таамаглал ч гарч болно — энэ нь "
        "загварын бодит чанарыг шударгаар харуулах зорилготой._",
    ]
    return text, headline, fig, highlights, explanation, "\n".join(truth)


# ── C: Side-by-side Flat vs Two-stage on the same input ─────────────────────

def compare_both(text: str):
    """Run BOTH architectures on one input so the Chapter 6 error-propagation
    argument is visible in a single screen.
    Returns: flat_head, flat_fig, two_head, two_fig, verdict_md."""
    text = (text or "").strip()
    if not text:
        ph = _placeholder_fig("Текст оруулна уу")
        return ("### —", ph, "### —", ph, "_Текст оруулна уу._")
    try:
        rf = _classify(text, ARCH_FLAT)
        rt = _classify(text, ARCH_TWO_STAGE)
    except DeviceSuspendedError:
        ph = _placeholder_fig("⚠")
        return ("### Алдаа", ph, "### Алдаа", ph, f"> {DEVICE_SUSPEND_MSG}")

    fmn = config.LABEL_MN.get(rf.final_label, rf.final_label)
    tmn = config.LABEL_MN.get(rt.final_label, rt.final_label)
    fhead = f"### Flat → {fmn}\n**Итгэл:** {rf.final_confidence:.1%} · `{rf.final_label}`"
    thead = f"### Two-stage → {tmn}\n**Итгэл:** {rt.final_confidence:.1%} · `{rt.final_label}`"
    ffig = _bar_chart(_four_class_view(rf), "Flat — 4 ангиллын магадлал")
    tfig = _bar_chart(_four_class_view(rt), "Two-stage — нэгтгэсэн магадлал")

    agree = rf.final_label == rt.final_label
    v = ["#### Дүгнэлт"]
    if agree:
        v.append(f"Хоёр архитектур ижил үр дүн (**{fmn}**) гаргалаа.")
    else:
        v.append(f"**Зөрүүтэй:** Flat → {fmn}, Two-stage → {tmn}.")
    if rt.branched_to_stage2:
        v.append(
            f"\nTwo-stage нь Stage 1-д Сөрөг гэж үзээд Stage 2 руу салаалсан. "
            f"Нэгтгэсэн итгэл {rt.final_confidence:.1%} нь Flat-ийн "
            f"{rf.final_confidence:.1%}-аас доогуур байгаа нь бүлэг 6-д "
            f"тайлбарласан **алдааны нийлбэр** (хоёр шатны үржвэр)-ийн "
            f"шууд жишээ юм."
        )
    else:
        v.append(
            "\nTwo-stage нь Stage 1-дээ шийдсэн тул Stage 2 ажиллаагүй "
            "(нэг л дамжуулалт)."
        )
    return fhead, ffig, thead, tfig, "\n".join(v)


# ── D: Static evaluation assets (precomputed, faithful path) ────────────────

def _eval_markdown() -> str:
    import json as _json
    try:
        rep = _json.loads(
            (Path(__file__).parent / "_eval_report.json").read_text(encoding="utf-8")
        )
    except Exception:  # noqa: BLE001
        rep = None

    md = [
        "### Үнэлгээ — Flat MN-BERT (SOTA) загвар",
        "",
        f"Held-out тест олонлог: **{config.N_TEST}** мөр.",
        "",
        "| Үзүүлэлт | Утга |",
        "|---|---|",
        f"| Нарийвчлал (Accuracy) | **{config.FLAT_ACCURACY:.2%}** |",
        f"| Macro-F1 | **{config.FLAT_MACRO_F1:.4f}** |",
    ]
    if rep:
        md += ["", "#### Ангилал тус бүрийн үзүүлэлт", "",
               "| Ангилал | Precision | Recall | F1 | Дэмжлэг |", "|---|---|---|---|---|"]
        for en in config.FLAT_LABELS:
            r = rep["corrected"].get(en, {})
            md.append(
                f"| {config.LABEL_MN[en]} (`{en}`) | {r.get('precision',0):.3f} "
                f"| {r.get('recall',0):.3f} | {r.get('f1-score',0):.3f} "
                f"| {int(r.get('support',0))} |"
            )
        md.append(
            "\n_POSITIVE ангилал хамгийн цөөн дээжтэй тул F1 нь бусдаас "
            "доогуур — энэ нь датасетийн ангиллын тэнцвэргүй байдлын илрэл._"
        )
    return "\n".join(md)


# ── Build the Gradio Blocks UI ──────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="MN-BERT — Монгол сэтгэгдлийн ангилал") as demo:
        gr.Markdown(
            "# Монгол сэтгэгдлийн ангилал — MN-BERT демо\n\n"
            "Дипломын ажилд бэлтгэсэн MN-BERT загваруудыг (Flat болон Two-stage) "
            "оруулсан текст дээр шууд ажиллуулна."
        )

        with gr.Row():
            architecture = gr.Radio(
                choices=ARCH_CHOICES,
                value=ARCH_FLAT,
                label="Архитектур",
                info=("Flat: нэг шатлалт, 4 ангилал.  "
                      "Two-stage: Stage 1 (3 ангилал) → нөхцөлт Stage 2 (2 ангилал)."),
            )

        with gr.Tabs():
            # ── Tab 1: Single text ────────────────────────────────────────
            with gr.Tab("Нэг сэтгэгдэл"):
                text_in = gr.Textbox(
                    label="Монгол кирилл текст",
                    lines=4,
                    placeholder="Жишээ: Үнэн ялна аа. Баяр хүргэе",
                )
                gr.Markdown("**Бэлэн жишээ:**")
                with gr.Row():
                    for label_mn, sample in config.EXAMPLES:
                        btn = gr.Button(label_mn, size="sm")
                        btn.click(lambda s=sample: s, outputs=text_in)
                with gr.Row():
                    run_btn = gr.Button("Ангилах", variant="primary")
                    rand_btn = gr.Button("🎲 Санамсаргүй тест жишээ", variant="secondary")

                headline_md = gr.Markdown()
                with gr.Row():
                    bar_plot = gr.Plot(label="Магадлалын тархалт")
                    highlights = gr.HighlightedText(
                        label="Үгсийн ач холбогдол (Input × Gradient)",
                        combine_adjacent=False,
                        show_legend=False,
                    )
                truth_md = gr.Markdown()
                gr.Markdown("### Шийдвэрийн тайлбар")
                explanation = gr.Markdown()

                def _predict_single_ui(t, a):
                    # clears the ground-truth panel (it belongs to a prior
                    # random sample, not to user-typed text)
                    h, f, hl, e = predict_single(t, a)
                    return h, f, hl, e, ""

                run_btn.click(
                    _predict_single_ui,
                    inputs=[text_in, architecture],
                    outputs=[headline_md, bar_plot, highlights, explanation, truth_md],
                )
                text_in.submit(
                    _predict_single_ui,
                    inputs=[text_in, architecture],
                    outputs=[headline_md, bar_plot, highlights, explanation, truth_md],
                )
                rand_btn.click(
                    random_test_sample,
                    inputs=[architecture],
                    outputs=[text_in, headline_md, bar_plot, highlights,
                             explanation, truth_md],
                )

            # ── Tab 2: Batch CSV ──────────────────────────────────────────
            with gr.Tab("Багц боловсруулалт (CSV)"):
                gr.Markdown(
                    "CSV файл оруулна уу: `text` эсвэл `comment` нэртэй багана "
                    "автоматаар илрүүлнэ. Үр дүн нь хүснэгт, ангиллын тархалт болон "
                    "татаж авах файл хэлбэрээр гарна."
                )
                with gr.Row():
                    csv_in = gr.File(label="CSV файл", file_types=[".csv"])
                    batch_btn = gr.Button("Багцыг боловсруулах", variant="primary")
                status = gr.Markdown()
                table_out = gr.Dataframe(label="Үр дүн", wrap=True)
                dist_plot = gr.Plot(label="Ангиллын тархалт")
                csv_download = gr.File(label="Татаж авах CSV")

                batch_btn.click(
                    predict_batch,
                    inputs=[csv_in, architecture],
                    outputs=[status, table_out, dist_plot, csv_download],
                )

            # ── Tab 3: Flat vs Two-stage side by side ─────────────────────
            with gr.Tab("Flat ⟷ Two-stage харьцуулалт"):
                gr.Markdown(
                    "Нэг текстийг **хоёр архитектураар зэрэг** ангилж, бүлэг 6-д "
                    "тайлбарласан алдаа дамжих (error propagation) үзэгдлийг нэг "
                    "дэлгэцэнд харуулна."
                )
                cmp_in = gr.Textbox(
                    label="Монгол кирилл текст", lines=3,
                    placeholder="Жишээ: Сугалааны азтан тодорлоо гэдэгт итгэхэд бэрх.",
                )
                cmp_btn = gr.Button("Хоёуланг нь ажиллуул", variant="primary")
                with gr.Row():
                    with gr.Column():
                        cmp_flat_head = gr.Markdown()
                        cmp_flat_fig = gr.Plot(label="Flat")
                    with gr.Column():
                        cmp_two_head = gr.Markdown()
                        cmp_two_fig = gr.Plot(label="Two-stage")
                cmp_verdict = gr.Markdown()
                cmp_btn.click(
                    compare_both, inputs=[cmp_in],
                    outputs=[cmp_flat_head, cmp_flat_fig, cmp_two_head,
                             cmp_two_fig, cmp_verdict],
                )
                cmp_in.submit(
                    compare_both, inputs=[cmp_in],
                    outputs=[cmp_flat_head, cmp_flat_fig, cmp_two_head,
                             cmp_two_fig, cmp_verdict],
                )

            # ── Tab 4: Evaluation (static, faithful test eval) ────────────
            with gr.Tab("Үнэлгээ"):
                gr.Markdown(_eval_markdown())
                _cm_png = Path(__file__).parent / "_eval_confusion.png"
                if _cm_png.exists():
                    gr.Image(value=str(_cm_png), label="Confusion matrix",
                             height=460)

        gr.Markdown(
            "---\n"
            "_ШУТИС · Компьютерын ухааны тэнхим · Бакалаврын дипломын ажил · 2026_"
        )
    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(share=True)
