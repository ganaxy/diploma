"""Inference wrappers for the Flat 4-class and Two-stage MN-BERT pipelines.

Each model is fed EXACTLY the input format its training script used (this is
the train/serve-parity fix):

  Flat  (best_mnbert_sota_corrected_model, retrain_sota_corrected.py)
        input  = "[news.mn] " + normalize_flat(user_text)
        len    = 256
        — verified to score 0.8326/0.8062 on the corrected test set, i.e. the
          exact thesis number, vs only 0.8136 for the old bare-text@128 path.

  Stage 1 / Stage 2 (step2/step3 train scripts, now off-disk)
        input  = clean_text(user_text)   (≈ text_light_clean, no source tag)
        len    = 128                     (documented assumption)

Device order: torch-directml (AMD) -> CUDA -> CPU. Set GRADIO_FORCE_CPU=1 to
skip the GPU entirely (reliable fallback if DirectML is flaky on demo day).
"""

import os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from . import config
from .preprocess import clean_text, normalize_flat


# ── Device selection ────────────────────────────────────────────────────────

def pick_device() -> tuple[torch.device, str]:
    """Return (device, human-readable label).

    GRADIO_FORCE_CPU=1 forces CPU. Otherwise: DirectML (AMD) -> CUDA -> CPU.
    """
    if os.environ.get("GRADIO_FORCE_CPU", "").strip() in ("1", "true", "True"):
        return torch.device("cpu"), "CPU (GRADIO_FORCE_CPU=1)"
    try:
        import torch_directml  # noqa: WPS433
        dev = torch_directml.device()
        return dev, "DirectML (AMD GPU)"
    except Exception:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda"), f"CUDA ({torch.cuda.get_device_name(0)})"
    return torch.device("cpu"), "CPU"


class DeviceSuspendedError(RuntimeError):
    """Raised when the GPU device faults mid-inference (DirectML 'suspended')."""


# ── Result containers ───────────────────────────────────────────────────────

@dataclass
class StagePrediction:
    label: str
    confidence: float
    probs: dict          # {english_label: float}


@dataclass
class PipelineResult:
    final_label: str
    final_confidence: float
    flat_probs: Optional[dict] = None         # 4-class distribution in Flat mode
    stage1: Optional[StagePrediction] = None  # set only in two-stage mode
    stage2: Optional[StagePrediction] = None  # set only when Stage 1 = NEGATIVE
    branched_to_stage2: bool = False


# ── Single-model wrapper ────────────────────────────────────────────────────

class _SingleModel:
    """One MN-BERT classifier, fed its training-faithful input format."""

    def __init__(self, model_dir, device, expected_labels,
                 input_builder: Callable[[str], str], max_length: int):
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.input_builder = input_builder
        self.max_length = max_length
        # NOTE: do NOT requires_grad_(False) here — on torch-directml it
        # triggers "device suspended" on the next forward when a second model
        # shares the device. The explainer backward still works fine.

        cfg_id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        ordered = [cfg_id2label[i] for i in range(len(cfg_id2label))]

        # grid_ckpt_* / best_stage*_model were saved with generic LABEL_n ids;
        # the training scripts are the authority. Otherwise verify the order.
        if all(name.startswith("LABEL_") for name in ordered):
            if len(ordered) != len(expected_labels):
                raise ValueError(
                    f"Label-count mismatch for {model_dir}: model has "
                    f"{len(ordered)} classes, config.py expects "
                    f"{len(expected_labels)} ({expected_labels})"
                )
            self.id2label = {i: name for i, name in enumerate(expected_labels)}
            self.model.config.id2label = self.id2label
            self.model.config.label2id = {n: i for i, n in self.id2label.items()}
        else:
            if ordered != expected_labels:
                raise ValueError(
                    f"Label mismatch for {model_dir}: model says {ordered}, "
                    f"config.py expects {expected_labels}"
                )
            self.id2label = cfg_id2label
        self.num_labels = len(self.id2label)

    def build_input(self, raw_text: str) -> str:
        """The exact string fed to the tokenizer (training-faithful)."""
        return self.input_builder(raw_text)

    @torch.no_grad()
    def predict(self, raw_text: str) -> StagePrediction:
        model_input = self.build_input(raw_text)
        enc = self.tokenizer(
            model_input,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        try:
            logits = self.model(**enc).logits[0]
        except RuntimeError as exc:
            if "suspended" in str(exc).lower() or "device" in str(exc).lower():
                raise DeviceSuspendedError(str(exc)) from exc
            raise
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        idx = int(np.argmax(probs))
        return StagePrediction(
            label=self.id2label[idx],
            confidence=float(probs[idx]),
            probs={self.id2label[i]: float(probs[i]) for i in range(self.num_labels)},
        )


# ── Input builders (training-faithful, per model) ───────────────────────────

def _flat_input(raw_text: str) -> str:
    return config.FLAT_SOURCE_PREFIX + normalize_flat(raw_text)


def _two_stage_input(raw_text: str) -> str:
    return clean_text(raw_text)


# ── Public classifiers ──────────────────────────────────────────────────────

class FlatClassifier:
    """Single-pass 4-class MN-BERT (the headline 'SOTA' model)."""

    def __init__(self, device):
        self.model = _SingleModel(
            config.FLAT_MODEL_DIR, device, config.FLAT_LABELS,
            input_builder=_flat_input, max_length=config.FLAT_MAX_LENGTH,
        )
        self.device = device

    def predict(self, text: str) -> PipelineResult:
        pred = self.model.predict(text)
        return PipelineResult(
            final_label=pred.label,
            final_confidence=pred.confidence,
            flat_probs=pred.probs,
        )


class TwoStageClassifier:
    """Stage 1 (3-class) then conditional Stage 2 (binary). Chapter 3 design."""

    def __init__(self, device):
        self.stage1 = _SingleModel(
            config.STAGE1_MODEL_DIR, device, config.STAGE1_LABELS,
            input_builder=_two_stage_input, max_length=config.TWO_STAGE_MAX_LENGTH,
        )
        self.stage2 = _SingleModel(
            config.STAGE2_MODEL_DIR, device, config.STAGE2_LABELS,
            input_builder=_two_stage_input, max_length=config.TWO_STAGE_MAX_LENGTH,
        )
        self.device = device

    def predict(self, text: str) -> PipelineResult:
        s1 = self.stage1.predict(text)
        if s1.label != "NEGATIVE":
            return PipelineResult(
                final_label=s1.label,
                final_confidence=s1.confidence,
                stage1=s1,
                branched_to_stage2=False,
            )
        s2 = self.stage2.predict(text)
        # Joint confidence = P(NEGATIVE | s1) * P(final | s2). Thesis Ch. 6
        # discusses this multiplicative error compounding. It is NOT a
        # calibrated posterior — the UI says so explicitly.
        joint = s1.confidence * s2.confidence
        return PipelineResult(
            final_label=s2.label,
            final_confidence=joint,
            stage1=s1,
            stage2=s2,
            branched_to_stage2=True,
        )
