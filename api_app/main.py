"""FastAPI service for the trained MN-BERT comment classifier.

This module keeps model loading lazy: importing the API does not allocate GPU
memory. The first prediction request loads the selected model and later
requests reuse it.
"""

from __future__ import annotations

import os
import threading
import time
from collections import Counter
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

if os.environ.get("API_FORCE_CPU", "").strip().lower() in {"1", "true", "yes"}:
    os.environ.setdefault("GRADIO_FORCE_CPU", "1")

from gradio_app import config
from gradio_app.models import (
    DeviceSuspendedError,
    FlatClassifier,
    PipelineResult,
    StagePrediction,
    TwoStageClassifier,
    pick_device,
)
from gradio_app.preprocess import cyrillic_ratio, language_warning


ARCH_FLAT = "flat"
ARCH_TWO_STAGE = "two_stage"
ARCH_SOFT_TWO_STAGE = "soft_two_stage"
ARCH_STRICT_TWO_STAGE = "strict_two_stage"
STRICT_NEGATIVE_THRESHOLD = 0.90
DEFAULT_ARCH = os.environ.get("API_DEFAULT_ARCH", ARCH_FLAT)


def _flat_serving_metadata() -> dict:
    """Metadata for the final model path shared with the Gradio app."""
    return {
        "architecture": ARCH_FLAT,
        "checkpoint": str(config.FLAT_MODEL_DIR),
        "checkpoint_exists": config.FLAT_MODEL_DIR.exists(),
        "max_length": config.FLAT_MAX_LENGTH,
        "source_prefix": config.FLAT_SOURCE_PREFIX,
        "preprocessing": (
            "clean_text -> normalize_flat -> prepend "
            f"{config.FLAT_SOURCE_PREFIX!r}"
        ),
        "labels": list(config.FLAT_LABELS),
        "reported_metrics": {
            "test_accuracy": config.FLAT_ACCURACY,
            "test_macro_f1": config.FLAT_MACRO_F1,
            "test_rows": config.N_TEST,
            "evaluation_path": "corrected v7 test split, faithful [source] + normalize @256 path",
        },
    }


def _normalise_architecture(value: str) -> str:
    value = (value or DEFAULT_ARCH).strip().lower().replace("-", "_")
    aliases = {
        "flat": ARCH_FLAT,
        "one_stage": ARCH_FLAT,
        "one_stage_mnbert": ARCH_FLAT,
        "single_stage": ARCH_FLAT,
        "two_stage": ARCH_TWO_STAGE,
        "cascade": ARCH_TWO_STAGE,
        "hierarchical": ARCH_TWO_STAGE,
        "soft_two_stage": ARCH_SOFT_TWO_STAGE,
        "soft_cascade": ARCH_SOFT_TWO_STAGE,
        "probability_fusion": ARCH_SOFT_TWO_STAGE,
        "strict_two_stage": ARCH_STRICT_TWO_STAGE,
        "strict_router": ARCH_STRICT_TWO_STAGE,
        "threshold_two_stage": ARCH_STRICT_TWO_STAGE,
    }
    if value not in aliases:
        raise HTTPException(
            status_code=400,
            detail=(
                "architecture must be one of: flat, one_stage, two_stage, "
                "cascade, soft_two_stage, strict_two_stage"
            ),
        )
    return aliases[value]


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1)
    architecture: str = Field(default=DEFAULT_ARCH)
    review_threshold: float = Field(default=0.70, ge=0.0, le=1.0)


class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1)
    architecture: str = Field(default=DEFAULT_ARCH)
    review_threshold: float = Field(default=0.70, ge=0.0, le=1.0)


class AuditRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1)
    architecture: str = Field(default=DEFAULT_ARCH)
    review_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    include_predictions: bool = True


class ModelRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._device = None
        self._device_label: Optional[str] = None
        self._flat: Optional[FlatClassifier] = None
        self._two_stage: Optional[TwoStageClassifier] = None

    @property
    def device_label(self) -> str:
        return self._device_label or "not loaded"

    @property
    def loaded_models(self) -> Dict[str, bool]:
        return {
            ARCH_FLAT: self._flat is not None,
            ARCH_TWO_STAGE: self._two_stage is not None,
            ARCH_SOFT_TWO_STAGE: self._two_stage is not None,
            ARCH_STRICT_TWO_STAGE: self._two_stage is not None,
        }

    def _ensure_device(self):
        if self._device is None:
            self._device, self._device_label = pick_device()
        return self._device

    def get(self, architecture: str):
        architecture = _normalise_architecture(architecture)
        with self._lock:
            device = self._ensure_device()
            if architecture == ARCH_FLAT:
                if self._flat is None:
                    self._flat = FlatClassifier(device)
                return self._flat
            # The two-stage variants reuse the same already trained Stage 1 /
            # Stage 2 weights. The architecture difference is only in the
            # inference-time decision rule.
            if self._two_stage is None:
                self._two_stage = TwoStageClassifier(device)
            return self._two_stage


registry = ModelRegistry()
app = FastAPI(
    title="MN-BERT Mongolian Comment Classifier API",
    version="0.1.0",
    description=(
        "REST API prototype for the final flat MN-BERT inference path. "
        "Use /predict for single comments, /batch for JSON batches, and "
        "/audit for moderation-style summaries."
    ),
)

DEMO_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MN-BERT API Demo</title>
  <style>
    :root {
      --ink: #17153f;
      --muted: #5e6475;
      --line: #d9dce6;
      --bg: #f6f7fb;
      --panel: #ffffff;
      --accent: #241f72;
      --accent-2: #f8b800;
      --good: #1c7c46;
      --warn: #a05a00;
      --bad: #b92d2d;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }

    header {
      padding: 28px clamp(18px, 4vw, 48px) 18px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }

    h1 {
      margin: 0 0 8px;
      font-size: clamp(28px, 4vw, 44px);
      line-height: 1.05;
      letter-spacing: 0;
    }

    .sub {
      margin: 0;
      max-width: 900px;
      color: var(--muted);
      font-size: 16px;
      line-height: 1.45;
    }

    main {
      display: grid;
      grid-template-columns: minmax(0, 1.05fr) minmax(320px, 0.95fr);
      gap: 18px;
      padding: 18px clamp(18px, 4vw, 48px) 36px;
    }

    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }

    h2 {
      margin: 0 0 14px;
      font-size: 20px;
      letter-spacing: 0;
    }

    label {
      display: block;
      margin: 14px 0 7px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
    }

    textarea, select, input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 11px 12px;
      font: inherit;
      color: var(--ink);
      background: #fff;
    }

    textarea {
      min-height: 132px;
      resize: vertical;
      line-height: 1.45;
    }

    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
      align-items: center;
    }

    button {
      border: 0;
      border-radius: 6px;
      padding: 11px 14px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      color: #fff;
      background: var(--accent);
    }

    button.secondary {
      color: var(--ink);
      background: #eef0f7;
      border: 1px solid var(--line);
    }

    button:disabled {
      cursor: wait;
      opacity: 0.65;
    }

    .mini-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 10px;
    }

    .mini-grid button {
      min-height: 42px;
      padding: 9px 10px;
      white-space: normal;
      text-align: center;
    }

    .result-head {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      align-items: start;
      border-bottom: 1px solid var(--line);
      padding-bottom: 14px;
      margin-bottom: 14px;
    }

    .label {
      font-size: clamp(26px, 4vw, 38px);
      font-weight: 800;
      line-height: 1.05;
      overflow-wrap: anywhere;
    }

    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 34px;
      border-radius: 999px;
      padding: 7px 12px;
      font-size: 13px;
      font-weight: 800;
      color: #fff;
      background: var(--good);
      white-space: nowrap;
    }

    .pill.flag { background: var(--bad); }
    .pill.review { background: var(--warn); }

    .metrics {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }

    .metric {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      background: #fafbff;
    }

    .metric strong {
      display: block;
      margin-top: 5px;
      font-size: 19px;
    }

    .bars {
      display: grid;
      gap: 10px;
    }

    .bar-row {
      display: grid;
      grid-template-columns: 150px minmax(0, 1fr) 56px;
      gap: 10px;
      align-items: center;
      font-size: 14px;
    }

    .bar-track {
      height: 13px;
      overflow: hidden;
      border-radius: 999px;
      background: #e9ebf4;
    }

    .bar-fill {
      height: 100%;
      border-radius: 999px;
      background: var(--accent);
    }

    pre {
      max-height: 280px;
      overflow: auto;
      padding: 12px;
      border-radius: 6px;
      border: 1px solid var(--line);
      background: #111327;
      color: #f7f7ff;
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
    }

    .hint {
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }

    .summary {
      display: grid;
      gap: 8px;
      margin-top: 12px;
    }

    .summary-row {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fafbff;
    }

    .error {
      color: var(--bad);
      font-weight: 700;
    }

    @media (max-width: 860px) {
      main {
        grid-template-columns: 1fr;
      }

      .metrics, .mini-grid {
        grid-template-columns: 1fr;
      }

      .bar-row {
        grid-template-columns: 1fr;
      }

      .result-head {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>MN-BERT API Demo</h1>
    <p class="sub">FastAPI service for the final flat MN-BERT model used by the Gradio app. It serves <code>best_mnbert_sota_corrected_model</code> with the faithful <code>[news.mn]</code> + normalized text path at max length 256. Reported test metrics: 83.26% accuracy and 0.8062 macro-F1 on 1,529 held-out rows.</p>
  </header>

  <main>
    <section>
      <h2>Single Comment</h2>
      <label for="architecture">Architecture</label>
      <select id="architecture">
        <option value="flat">Flat one-stage MN-BERT</option>
        <option value="two_stage">Hard two-stage cascade</option>
        <option value="soft_two_stage">Soft two-stage fusion, no retraining</option>
        <option value="strict_two_stage">Strict two-stage router, no retraining</option>
      </select>
      <p class="hint">Default: final Flat MN-BERT. The other choices are retained only for architecture comparison.</p>
      <label for="comment">Comment text</label>
      <textarea id="comment"></textarea>

      <div class="mini-grid" id="examples"></div>

      <div class="toolbar">
        <button id="predictBtn">Classify</button>
        <button class="secondary" id="clearBtn">Clear</button>
      </div>

      <label for="batchText">Batch audit comments, one per line</label>
      <textarea id="batchText"></textarea>
      <div class="toolbar">
        <button id="auditBtn">Run Audit Summary</button>
      </div>
    </section>

    <section>
      <h2>Result</h2>
      <div class="result-head">
        <div>
          <div class="label" id="label">Ready</div>
          <p class="hint" id="hint">Use the simple page for the demo. Keep Swagger at <a href="/docs">/docs</a> as technical evidence.</p>
        </div>
        <span class="pill" id="action">idle</span>
      </div>

      <div class="metrics">
        <div class="metric">Confidence<strong id="confidence">-</strong></div>
        <div class="metric">Latency<strong id="latency">-</strong></div>
        <div class="metric">Architecture<strong id="arch">flat</strong></div>
      </div>

      <div class="bars" id="bars"></div>

      <div class="summary" id="summary"></div>

      <label>Raw API response</label>
      <pre id="raw">{}</pre>
    </section>
  </main>

  <script>
    const labelNames = {
      POSITIVE: "Positive",
      NEUTRAL: "Neutral",
      CONSTRUCTIVE: "Constructive criticism",
      TOXIC: "Toxic negative"
    };

    const colors = {
      POSITIVE: "#1c7c46",
      NEUTRAL: "#6d7283",
      CONSTRUCTIVE: "#1f67b2",
      TOXIC: "#b92d2d"
    };

    const examples = [
      ["Constructive", "\u042d\u043d\u044d \u04af\u0439\u043b\u0447\u0438\u043b\u0433\u044d\u044d \u043c\u0430\u0448 \u0443\u0434\u0430\u0430\u043d \u0431\u0430\u0439\u043d\u0430, \u0437\u0430\u0441\u0430\u0445 \u0445\u044d\u0440\u044d\u0433\u0442\u044d\u0439."],
      ["Neutral", "\u0421\u0430\u0439\u043d \u0431\u0430\u0439\u043d\u0430 \u0443\u0443. \u0422\u0430 11-11 \u0434\u0443\u0433\u0430\u0430\u0440\u0442 \u0445\u043e\u043b\u0431\u043e\u0433\u0434\u043e\u043d \u043c\u044d\u0434\u044d\u044d\u043b\u044d\u043b \u0430\u0432\u0430\u0445 \u0431\u043e\u043b\u043e\u043c\u0436\u0442\u043e\u0439."],
      ["Toxic", "\u042f\u043c\u0430\u0440 \u0447 \u0443\u0442\u0433\u0430\u0433\u04af\u0439 \u0442\u044d\u043d\u044d\u0433 \u0445\u04af\u043c\u04af\u04af\u0441 \u0432\u044d."],
      ["Positive", "\u04ae\u043d\u044d\u0445\u044d\u044d\u0440 \u0441\u0430\u0439\u043d \u0430\u0436\u0438\u043b \u0431\u043e\u043b\u0436\u044d\u044d, \u0431\u0430\u044f\u0440 \u0445\u04af\u0440\u0433\u044d\u0435."]
    ];

    const comment = document.getElementById("comment");
    const batchText = document.getElementById("batchText");
    const label = document.getElementById("label");
    const action = document.getElementById("action");
    const hint = document.getElementById("hint");
    const confidence = document.getElementById("confidence");
    const latency = document.getElementById("latency");
    const arch = document.getElementById("arch");
    const bars = document.getElementById("bars");
    const raw = document.getElementById("raw");
    const summary = document.getElementById("summary");
    const predictBtn = document.getElementById("predictBtn");
    const auditBtn = document.getElementById("auditBtn");
    const archSelect = document.getElementById("architecture");

    function setBusy(isBusy) {
      predictBtn.disabled = isBusy;
      auditBtn.disabled = isBusy;
    }

    function setAction(value) {
      action.textContent = value;
      action.className = "pill";
      if (value === "flag") action.classList.add("flag");
      if (value === "review") action.classList.add("review");
    }

    function renderBars(probabilities) {
      bars.innerHTML = "";
      for (const key of ["POSITIVE", "NEUTRAL", "CONSTRUCTIVE", "TOXIC"]) {
        const value = Number(probabilities[key] || 0);
        const row = document.createElement("div");
        row.className = "bar-row";
        row.innerHTML = `
          <span>${labelNames[key]}</span>
          <span class="bar-track"><span class="bar-fill" style="width:${Math.round(value * 100)}%; background:${colors[key]}"></span></span>
          <strong>${Math.round(value * 100)}%</strong>
        `;
        bars.appendChild(row);
      }
    }

    function renderPrediction(data) {
      label.textContent = data.label_mn || labelNames[data.label] || data.label;
      setAction(data.action || "allow");
      hint.textContent = data.warning || `Final label: ${data.label}`;
      confidence.textContent = `${Math.round(Number(data.confidence || 0) * 100)}%`;
      latency.textContent = `${Math.round(Number(data.latency_ms || 0))} ms`;
      arch.textContent = data.architecture || "flat";
      renderBars(data.probabilities || {});
      summary.innerHTML = "";
      raw.textContent = JSON.stringify(data, null, 2);
    }

    function renderError(err) {
      label.textContent = "Request failed";
      setAction("review");
      hint.innerHTML = `<span class="error">${err.message || err}</span>`;
      raw.textContent = String(err.stack || err);
    }

    async function postJson(url, body) {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });
      const text = await res.text();
      let data;
      try {
        data = text ? JSON.parse(text) : {};
      } catch (err) {
        throw new Error(text || `${res.status} ${res.statusText}`);
      }
      if (!res.ok) {
        const detail = data.detail || data.message || res.statusText;
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail, null, 2));
      }
      return data;
    }

    async function predict() {
      const text = comment.value.trim();
      if (!text) {
        comment.focus();
        return;
      }
      setBusy(true);
      try {
        renderPrediction(await postJson("/predict", {
          text,
          architecture: archSelect.value,
          review_threshold: 0.70
        }));
      } catch (err) {
        renderError(err);
      } finally {
        setBusy(false);
      }
    }

    async function audit() {
      const texts = batchText.value
        .split(/\n+/)
        .map(s => s.trim())
        .filter(Boolean);
      if (!texts.length) {
        batchText.focus();
        return;
      }
      setBusy(true);
      try {
        const data = await postJson("/audit", {
          texts,
          architecture: archSelect.value,
          review_threshold: 0.70,
          include_predictions: false
        });
        label.textContent = "Audit Summary";
        setAction("summary");
        hint.textContent = `${data.summary.total} comments processed`;
        confidence.textContent = "-";
        latency.textContent = `${Math.round(data.summary.avg_latency_ms)} ms avg`;
        arch.textContent = archSelect.value;
        bars.innerHTML = "";
        summary.innerHTML = "";
        for (const [k, v] of Object.entries(data.summary.label_counts || {})) {
          const row = document.createElement("div");
          row.className = "summary-row";
          row.innerHTML = `<span>${labelNames[k] || k}</span><strong>${v}</strong>`;
          summary.appendChild(row);
        }
        for (const [k, v] of Object.entries(data.summary.action_counts || {})) {
          const row = document.createElement("div");
          row.className = "summary-row";
          row.innerHTML = `<span>Action: ${k}</span><strong>${v}</strong>`;
          summary.appendChild(row);
        }
        raw.textContent = JSON.stringify(data, null, 2);
      } catch (err) {
        renderError(err);
      } finally {
        setBusy(false);
      }
    }

    document.getElementById("examples").innerHTML = "";
    for (const [name, text] of examples) {
      const btn = document.createElement("button");
      btn.className = "secondary";
      btn.type = "button";
      btn.textContent = name;
      btn.addEventListener("click", () => {
        comment.value = text;
      });
      document.getElementById("examples").appendChild(btn);
    }

    batchText.value = examples.slice(0, 3).map(item => item[1]).join("\n");
    comment.value = examples[0][1];
    predictBtn.addEventListener("click", predict);
    auditBtn.addEventListener("click", audit);
    document.getElementById("clearBtn").addEventListener("click", () => {
      comment.value = "";
      comment.focus();
    });
  </script>
</body>
</html>"""


@app.on_event("startup")
def _preload_if_requested() -> None:
    preload = os.environ.get("API_PRELOAD_MODELS", "").strip().lower()
    if preload in {"1", "true", "yes", "flat"}:
        registry.get(ARCH_FLAT)
    elif preload in {"two_stage", "cascade"}:
        registry.get(ARCH_TWO_STAGE)
    elif preload in {"soft_two_stage", "soft_cascade"}:
        registry.get(ARCH_SOFT_TWO_STAGE)
    elif preload in {"strict_two_stage", "strict_router"}:
        registry.get(ARCH_STRICT_TWO_STAGE)
    elif preload == "all":
        registry.get(ARCH_FLAT)
        registry.get(ARCH_TWO_STAGE)


def _stage_to_dict(stage: Optional[StagePrediction]) -> Optional[dict]:
    if stage is None:
        return None
    return {
        "label": stage.label,
        "label_mn": config.LABEL_MN.get(stage.label, stage.label),
        "confidence": stage.confidence,
        "probabilities": stage.probs,
    }


def _four_class_probabilities(result: PipelineResult) -> Dict[str, float]:
    if result.flat_probs is not None:
        return {label: float(result.flat_probs.get(label, 0.0)) for label in config.FLAT_LABELS}

    if result.stage1 is None:
        return {label: 0.0 for label in config.FLAT_LABELS}

    stage1 = result.stage1.probs
    p_pos = float(stage1.get("POSITIVE", 0.0))
    p_neu = float(stage1.get("NEUTRAL", 0.0))
    p_neg = float(stage1.get("NEGATIVE", 0.0))
    if result.stage2 is not None:
        p_con = p_neg * float(result.stage2.probs.get("CONSTRUCTIVE", 0.0))
        p_tox = p_neg * float(result.stage2.probs.get("TOXIC", 0.0))
    else:
        p_con = p_neg / 2.0
        p_tox = p_neg / 2.0
    return {
        "POSITIVE": p_pos,
        "NEUTRAL": p_neu,
        "CONSTRUCTIVE": p_con,
        "TOXIC": p_tox,
    }


def _moderation_action(label: str, confidence: float, review_threshold: float) -> str:
    if confidence < review_threshold:
        return "review"
    if label == "TOXIC":
        return "flag"
    return "allow"


def _model_error(architecture: str, exc: Exception) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail={
            "message": (
                "Model inference failed before a prediction could be returned. "
                "Check that the checkpoint folders were copied and that PyTorch "
                "is installed for this machine."
            ),
            "architecture": architecture,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "model_paths": {
                ARCH_FLAT: str(config.FLAT_MODEL_DIR),
                "stage1": str(config.STAGE1_MODEL_DIR),
                "stage2": str(config.STAGE2_MODEL_DIR),
            },
            "model_paths_exist": {
                ARCH_FLAT: config.FLAT_MODEL_DIR.exists(),
                "stage1": config.STAGE1_MODEL_DIR.exists(),
                "stage2": config.STAGE2_MODEL_DIR.exists(),
            },
            "hint": (
                "For the RTX 3060 laptop, run setup_api_demo_nvidia.bat once, "
                "then run_api_demo_nvidia.bat. If CUDA still fails, update the "
                "NVIDIA driver or use the CPU fallback run_api_demo.bat."
            ),
        },
    )


def _predict_soft_two_stage(text: str, review_threshold: float) -> dict:
    start = time.perf_counter()
    try:
        model = registry.get(ARCH_SOFT_TWO_STAGE)
        stage1 = model.stage1.predict(text)
        stage2 = model.stage2.predict(text)
    except DeviceSuspendedError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "GPU device was suspended during inference. Restart the API "
                "or set API_FORCE_CPU=1 for a stable CPU fallback."
            ),
        ) from exc
    except Exception as exc:
        raise _model_error(ARCH_SOFT_TWO_STAGE, exc) from exc
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    p_negative = float(stage1.probs.get("NEGATIVE", 0.0))
    probabilities = {
        "POSITIVE": float(stage1.probs.get("POSITIVE", 0.0)),
        "NEUTRAL": float(stage1.probs.get("NEUTRAL", 0.0)),
        "CONSTRUCTIVE": p_negative * float(stage2.probs.get("CONSTRUCTIVE", 0.0)),
        "TOXIC": p_negative * float(stage2.probs.get("TOXIC", 0.0)),
    }
    label = max(probabilities, key=probabilities.get)
    confidence = float(probabilities[label])
    warning = language_warning(text) or None
    return {
        "label": label,
        "label_mn": config.LABEL_MN.get(label, label),
        "confidence": confidence,
        "probabilities": probabilities,
        "architecture": ARCH_SOFT_TWO_STAGE,
        "action": _moderation_action(label, confidence, review_threshold),
        "latency_ms": elapsed_ms,
        "cyrillic_ratio": cyrillic_ratio(text),
        "warning": warning,
        "branched_to_stage2": True,
        "stage1": _stage_to_dict(stage1),
        "stage2": _stage_to_dict(stage2),
        "note": (
            "No retraining: Stage 1 and Stage 2 weights are reused; only the "
            "hard cascade decision rule is replaced by probability fusion."
        ),
    }


def _predict_strict_two_stage(text: str, review_threshold: float) -> dict:
    start = time.perf_counter()
    try:
        model = registry.get(ARCH_STRICT_TWO_STAGE)
        stage1 = model.stage1.predict(text)
        p_negative = float(stage1.probs.get("NEGATIVE", 0.0))
        should_route = p_negative >= STRICT_NEGATIVE_THRESHOLD
        stage2 = model.stage2.predict(text) if should_route else None
    except DeviceSuspendedError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "GPU device was suspended during inference. Restart the API "
                "or set API_FORCE_CPU=1 for a stable CPU fallback."
            ),
        ) from exc
    except Exception as exc:
        raise _model_error(ARCH_STRICT_TWO_STAGE, exc) from exc
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    p_positive = float(stage1.probs.get("POSITIVE", 0.0))
    p_neutral = float(stage1.probs.get("NEUTRAL", 0.0))
    if stage2 is not None:
        probabilities = {
            "POSITIVE": p_positive,
            "NEUTRAL": p_neutral,
            "CONSTRUCTIVE": p_negative * float(stage2.probs.get("CONSTRUCTIVE", 0.0)),
            "TOXIC": p_negative * float(stage2.probs.get("TOXIC", 0.0)),
        }
        label = stage2.label
        confidence = p_negative * float(stage2.confidence)
    else:
        probabilities = {
            "POSITIVE": p_positive,
            "NEUTRAL": p_neutral,
            "CONSTRUCTIVE": 0.0,
            "TOXIC": 0.0,
        }
        label = "POSITIVE" if p_positive >= p_neutral else "NEUTRAL"
        confidence = float(probabilities[label])

    warning = language_warning(text) or None
    return {
        "label": label,
        "label_mn": config.LABEL_MN.get(label, label),
        "confidence": confidence,
        "probabilities": probabilities,
        "architecture": ARCH_STRICT_TWO_STAGE,
        "action": _moderation_action(label, confidence, review_threshold),
        "latency_ms": elapsed_ms,
        "cyrillic_ratio": cyrillic_ratio(text),
        "warning": warning,
        "branched_to_stage2": bool(stage2 is not None),
        "stage1": _stage_to_dict(stage1),
        "stage2": _stage_to_dict(stage2),
        "note": (
            "No retraining: Stage 2 is called only when Stage 1 negative "
            f"probability is at least {STRICT_NEGATIVE_THRESHOLD:.2f}. This "
            "reduces error propagation from borderline negative routing."
        ),
    }


def _predict_text(text: str, architecture: str, review_threshold: float) -> dict:
    clean = (text or "").strip()
    if not clean:
        raise HTTPException(status_code=400, detail="text must not be empty")

    architecture = _normalise_architecture(architecture)
    if architecture == ARCH_SOFT_TWO_STAGE:
        return _predict_soft_two_stage(clean, review_threshold)
    if architecture == ARCH_STRICT_TWO_STAGE:
        return _predict_strict_two_stage(clean, review_threshold)

    start = time.perf_counter()
    try:
        model = registry.get(architecture)
        result = model.predict(clean)
    except DeviceSuspendedError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "GPU device was suspended during inference. Restart the API "
                "or set API_FORCE_CPU=1 for a stable CPU fallback."
            ),
        ) from exc
    except Exception as exc:
        raise _model_error(architecture, exc) from exc
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    warning = language_warning(clean) or None
    label = result.final_label
    confidence = float(result.final_confidence)
    return {
        "label": label,
        "label_mn": config.LABEL_MN.get(label, label),
        "confidence": confidence,
        "probabilities": _four_class_probabilities(result),
        "architecture": architecture,
        "action": _moderation_action(label, confidence, review_threshold),
        "latency_ms": elapsed_ms,
        "cyrillic_ratio": cyrillic_ratio(clean),
        "warning": warning,
        "branched_to_stage2": bool(result.branched_to_stage2),
        "stage1": _stage_to_dict(result.stage1),
        "stage2": _stage_to_dict(result.stage2),
    }


def _summarise(predictions: List[dict]) -> dict:
    label_counts = Counter(pred["label"] for pred in predictions)
    action_counts = Counter(pred["action"] for pred in predictions)
    latencies = [float(pred["latency_ms"]) for pred in predictions]
    return {
        "total": len(predictions),
        "label_counts": dict(label_counts),
        "action_counts": dict(action_counts),
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "max_latency_ms": max(latencies) if latencies else 0.0,
    }


@app.get("/", response_class=HTMLResponse)
def demo_page() -> str:
    return DEMO_HTML


@app.get("/info")
def root() -> dict:
    return {
        "service": "MN-BERT Mongolian Comment Classifier API",
        "default_architecture": _normalise_architecture(DEFAULT_ARCH),
        "final_model": _flat_serving_metadata(),
        "docs": "/docs",
        "health": "/health",
        "endpoints": ["/predict", "/batch", "/audit"],
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "device": registry.device_label,
        "loaded_models": registry.loaded_models,
        "model_paths": {
            ARCH_FLAT: str(config.FLAT_MODEL_DIR),
            "stage1": str(config.STAGE1_MODEL_DIR),
            "stage2": str(config.STAGE2_MODEL_DIR),
        },
        "model_paths_exist": {
            ARCH_FLAT: config.FLAT_MODEL_DIR.exists(),
            "stage1": config.STAGE1_MODEL_DIR.exists(),
            "stage2": config.STAGE2_MODEL_DIR.exists(),
        },
        "final_model": _flat_serving_metadata(),
    }


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    return _predict_text(
        text=request.text,
        architecture=request.architecture,
        review_threshold=request.review_threshold,
    )


@app.post("/batch")
def batch(request: BatchRequest) -> dict:
    predictions = [
        _predict_text(text, request.architecture, request.review_threshold)
        for text in request.texts
    ]
    return {
        "summary": _summarise(predictions),
        "predictions": predictions,
    }


@app.post("/audit")
def audit(request: AuditRequest) -> dict:
    predictions = [
        _predict_text(text, request.architecture, request.review_threshold)
        for text in request.texts
    ]
    review_queue = [
        {
            "index": idx,
            "label": pred["label"],
            "confidence": pred["confidence"],
            "action": pred["action"],
        }
        for idx, pred in enumerate(predictions)
        if pred["action"] == "review"
    ]
    response = {
        "summary": _summarise(predictions),
        "review_queue": review_queue,
    }
    if request.include_predictions:
        response["predictions"] = predictions
    return response
