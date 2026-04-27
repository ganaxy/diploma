from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from . import config

def pick_device() -> tuple[torch.device, str]:
    try:
        dev = torch_directml.device()
        return dev, "DirectML (AMD GPU)"
    except Exception:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda"), f"CUDA ({torch.cuda.get_device_name(0)})"
    return torch.device("cpu"), "CPU"

@dataclass
class StagePrediction:
    label: str
    confidence: float

@dataclass
class PipelineResult:
    branched_to_stage2: bool = False

class _SingleModel:

    def __init__(self, model_dir, device: torch.device, expected_labels: list[str]):
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.to(device)
        self.model.eval()
        self.device = device

        cfg_id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        ordered = [cfg_id2label[i] for i in range(len(cfg_id2label))]

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

    @torch.no_grad()
    def predict(self, text: str) -> StagePrediction:
        enc = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=config.MAX_LENGTH,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(**enc).logits[0]
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        idx = int(np.argmax(probs))
        return StagePrediction(
            label=self.id2label[idx],
            confidence=float(probs[idx]),
            probs={self.id2label[i]: float(probs[i]) for i in range(self.num_labels)},
        )

class FlatClassifier:

    def __init__(self, device: torch.device):
        self.model = _SingleModel(config.FLAT_MODEL_DIR, device, config.FLAT_LABELS)
        self.device = device

    def predict(self, text: str) -> PipelineResult:
        pred = self.model.predict(text)
        return PipelineResult(
            final_label=pred.label,
            final_confidence=pred.confidence,
            flat_probs=pred.probs,
        )

class TwoStageClassifier:

    def __init__(self, device: torch.device):
        self.stage1 = _SingleModel(config.STAGE1_MODEL_DIR, device, config.STAGE1_LABELS)
        self.stage2 = _SingleModel(config.STAGE2_MODEL_DIR, device, config.STAGE2_LABELS)
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
        joint = s1.confidence * s2.confidence
        return PipelineResult(
            final_label=s2.label,
            final_confidence=joint,
            stage1=s1,
            stage2=s2,
            branched_to_stage2=True,
        )
