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

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from gradio_app import config
    from gradio_app.explain import token_attributions
    from gradio_app.models import FlatClassifier, PipelineResult, TwoStageClassifier, pick_device
    from gradio_app.preprocess import clean_text, language_warning
else:
    from . import config
    from .explain import token_attributions
    from .models import FlatClassifier, PipelineResult, TwoStageClassifier, pick_device
    from .preprocess import clean_text, language_warning

ARCH_FLAT = "Flat — нэг шатлалт (4 ангилал)"
ARCH_TWO_STAGE = "Two-stage — хоёр шатлалт (Stage 1 → Stage 2)"
ARCH_CHOICES = [ARCH_FLAT, ARCH_TWO_STAGE]

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

def _classify(text: str, architecture: str) -> PipelineResult:
    cleaned = clean_text(text)
    if architecture == ARCH_TWO_STAGE:
        return TWO_STAGE_MODEL.predict(cleaned)
    return FLAT_MODEL.predict(cleaned)

def _four_class_view(result: PipelineResult) -> dict:
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
    return "\n".join(lines)

def _highlights_for(result: PipelineResult, text: str, architecture: str
                    ) -> List[Tuple[str, float]]:
    cleaned = clean_text(text)
    if architecture == ARCH_FLAT:
        return token_attributions(FLAT_MODEL.model, cleaned, result.final_label,
                                  max_length=config.MAX_LENGTH)
    if result.branched_to_stage2:
        return token_attributions(TWO_STAGE_MODEL.stage2, cleaned, result.final_label,
                                  max_length=config.MAX_LENGTH)
    return token_attributions(TWO_STAGE_MODEL.stage1, cleaned, result.final_label,
                              max_length=config.MAX_LENGTH)

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
    result = _classify(text, architecture)
    try:
        highlights = _highlights_for(result, text, architecture)
        highlights = [(text, 0.0)]
        print(f"[explain] gradient attribution failed: {exc!r}")
    latency_ms = (time.perf_counter() - t0) * 1000.0

    final_mn = config.LABEL_MN.get(result.final_label, result.final_label)
    headline = f"### {final_mn}\n**Итгэл:** {result.final_confidence:.1%}  ·  `{result.final_label}`"

    chart_title = ("4-ангиллын магадлал (Flat)" if architecture == ARCH_FLAT
                   else "Нэгтгэсэн 4-ангиллын магадлал (Two-stage)")
    fig = _bar_chart(_four_class_view(result), chart_title)

    explanation = _explanation_markdown(text, architecture, result, warning, latency_ms)
    return headline, fig, highlights, explanation

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
    for raw in texts:
        result = _classify(raw, architecture)
        eng_labels.append(result.final_label)
        rows.append({
            "text": raw,
            "predicted_label_en": result.final_label,
            "predicted_label_mn": config.LABEL_MN.get(result.final_label, result.final_label),
            "confidence": round(result.final_confidence, 4),
        })
    elapsed = time.perf_counter() - t0

    out_df = pd.DataFrame(rows)
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

def _model_info_md() -> str:
    flat_labels = ", ".join(f"{config.LABEL_MN[l]} (`{l}`)" for l in config.FLAT_LABELS)
    return f"""### Загварын мэдээлэл

- **Үндсэн загвар:** `tugstugi/bert-base-mongolian-cased` (MN-BERT)
- **Идэвхтэй төхөөрөмж:** `{DEVICE_LABEL}`
- **Токенжүүлэгч:** AlbertTokenizer (SentencePiece, `max_length={config.MAX_LENGTH}`)

#### Flat — нэг шатлалт, 4 ангилал
- Зам: `{config.FLAT_MODEL_DIR.name}`
- Шошго: {flat_labels}
- **Туршилтын нарийвчлал:** {config.FLAT_ACCURACY:.2%} · **Macro-F1:** {config.FLAT_MACRO_F1:.4f}

#### Two-stage — хоёр шатлалт дамжлага
- Stage 1 (3 ангилал): `{config.STAGE1_MODEL_DIR.name}`
- Stage 2 (2 ангилал): `{config.STAGE2_MODEL_DIR.name}`
- **Stage 2 нарийвчлал:** {config.STAGE2_ACCURACY:.2%}
"""

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
            with gr.Tab("Нэг сэтгэгдэл"):
                with gr.Row():
                    with gr.Column(scale=3):
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
                        run_btn = gr.Button("Ангилах", variant="primary")
                    with gr.Column(scale=2):
                        with gr.Accordion("Загварын мэдээлэл", open=False):
                            gr.Markdown(_model_info_md())

                headline_md = gr.Markdown()
                with gr.Row():
                    bar_plot = gr.Plot(label="Магадлалын тархалт")
                    highlights = gr.HighlightedText(
                        label="Үгсийн ач холбогдол (Input × Gradient)",
                        combine_adjacent=False,
                        show_legend=False,
                    )
                gr.Markdown("### Шийдвэрийн тайлбар")
                explanation = gr.Markdown()

                run_btn.click(
                    predict_single,
                    inputs=[text_in, architecture],
                    outputs=[headline_md, bar_plot, highlights, explanation],
                )
                text_in.submit(
                    predict_single,
                    inputs=[text_in, architecture],
                    outputs=[headline_md, bar_plot, highlights, explanation],
                )

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

        gr.Markdown(
            "---\n"
            "_ШУТИС · Компьютерын ухааны тэнхим · Бакалаврын дипломын ажил · 2026_"
        )
    return demo

if __name__ == "__main__":
    ui = build_ui()
    ui.launch()
