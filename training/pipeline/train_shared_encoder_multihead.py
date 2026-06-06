"""Train a shared-encoder two-head MN-BERT architecture.

This experiment tests the "one encoder, multiple classifier heads" alternative
to the original two-stage pipeline that used two separate BERT encoders.

Architecture:
    text -> shared MN-BERT encoder -> Stage 1 head (POSITIVE/NEUTRAL/NEGATIVE)
                                -> Stage 2 head (CONSTRUCTIVE/TOXIC)

Stage 2 loss is applied only to examples whose final label is CONSTRUCTIVE or
TOXIC. At inference time, Stage 1 routes POSITIVE/NEUTRAL directly and Stage 2
decides the final negative subtype.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_CSV = PROJECT_ROOT / "sample scores" / "relabeled_v7_corrected.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "sample scores" / "shared_encoder_multihead_model"
HF_NAME = "tugstugi/bert-base-mongolian-cased"

STAGE1_LABELS = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
STAGE2_LABELS = ["CONSTRUCTIVE", "TOXIC"]
FINAL_LABELS = ["POSITIVE", "NEUTRAL", "CONSTRUCTIVE", "TOXIC"]
STAGE1_MAP = {label: idx for idx, label in enumerate(STAGE1_LABELS)}
STAGE2_MAP = {label: idx for idx, label in enumerate(STAGE2_LABELS)}
FINAL_MAP = {label: idx for idx, label in enumerate(FINAL_LABELS)}
IGNORE_INDEX = -100


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name: str) -> tuple[torch.device, str]:
    name = name.lower()
    if name == "cpu":
        return torch.device("cpu"), "CPU"
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda"), f"CUDA ({torch.cuda.get_device_name(0)})"
    if name in {"auto", "directml"}:
        if name in {"auto", "directml"}:
            try:
                import torch_directml  # noqa: WPS433

                device = torch_directml.device()
                _ = torch.zeros(1, device=device)
                return device, "DirectML"
            except Exception:
                if name == "directml":
                    raise
        if torch.cuda.is_available():
            return torch.device("cuda"), f"CUDA ({torch.cuda.get_device_name(0)})"
    return torch.device("cpu"), "CPU"


def load_dataframe(path: Path, limit_train: int, limit_val: int, limit_test: int) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"id", "text_normalized", "label", "split", "source", "text_light_clean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df["split"] = df["split"].astype(str).str.lower()
    df["label"] = df["label"].astype(str).str.upper()
    df["input_text"] = df.apply(
        lambda row: f"[{row['source']}] {str(row['text_normalized'] or '')}",
        axis=1,
    )
    df["final_id"] = df["label"].map(FINAL_MAP)
    df["stage1_label"] = np.where(df["label"].isin(STAGE2_LABELS), "NEGATIVE", df["label"])
    df["stage1_id"] = df["stage1_label"].map(STAGE1_MAP)
    df["stage2_id"] = df["label"].map(STAGE2_MAP).fillna(IGNORE_INDEX).astype(int)

    limits = {"train": limit_train, "val": limit_val, "test": limit_test}
    parts = []
    for split, limit in limits.items():
        part = df[df["split"] == split].copy()
        if limit and limit > 0:
            part = part.head(limit).copy()
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


class EncodedDataset(Dataset):
    def __init__(
        self,
        encodings: dict[str, torch.Tensor],
        stage1_labels: list[int],
        stage2_labels: list[int],
        final_labels: list[int],
    ) -> None:
        self.encodings = encodings
        self.stage1_labels = torch.tensor(stage1_labels, dtype=torch.long)
        self.stage2_labels = torch.tensor(stage2_labels, dtype=torch.long)
        self.final_labels = torch.tensor(final_labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.final_labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "stage1_labels": self.stage1_labels[idx],
            "stage2_labels": self.stage2_labels[idx],
            "final_labels": self.final_labels[idx],
        }


def build_dataset(tokenizer, df: pd.DataFrame, max_length: int) -> EncodedDataset:
    enc = tokenizer(
        df["input_text"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return EncodedDataset(
        encodings={"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]},
        stage1_labels=df["stage1_id"].astype(int).tolist(),
        stage2_labels=df["stage2_id"].astype(int).tolist(),
        final_labels=df["final_id"].astype(int).tolist(),
    )


class SharedEncoderMultiHead(nn.Module):
    def __init__(self, model_name: str, dropout: float) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        config.hidden_dropout_prob = dropout
        config.attention_probs_dropout_prob = dropout
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        hidden_size = int(config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.stage1_head = nn.Linear(hidden_size, len(STAGE1_LABELS))
        self.stage2_head = nn.Linear(hidden_size, len(STAGE2_LABELS))
        self.config = config

    def resize_token_embeddings(self, size: int) -> None:
        self.encoder.resize_token_embeddings(size)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        return self.stage1_head(pooled), self.stage2_head(pooled)


def class_weights(values: list[int], num_labels: int) -> np.ndarray:
    counts = np.bincount(np.asarray(values, dtype=int), minlength=num_labels).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    total = counts.sum()
    return total / (num_labels * counts)


def make_loaders(args, tokenizer, df: pd.DataFrame):
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    train_ds = build_dataset(tokenizer, train_df, args.max_length)
    val_ds = build_dataset(tokenizer, val_df, args.max_length)
    test_ds = build_dataset(tokenizer, test_df, args.max_length)

    final_weights = class_weights(train_df["final_id"].astype(int).tolist(), len(FINAL_LABELS))
    sample_weights = np.asarray([final_weights[label] for label in train_df["final_id"].astype(int)], dtype=np.float64)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)
    return train_df, val_df, test_df, train_loader, val_loader, test_loader


def final_from_heads(stage1_logits: torch.Tensor, stage2_logits: torch.Tensor, mode: str, threshold: float) -> torch.Tensor:
    stage1_probs = torch.softmax(stage1_logits, dim=-1)
    stage2_probs = torch.softmax(stage2_logits, dim=-1)
    stage1_pred = torch.argmax(stage1_probs, dim=-1)
    stage2_pred = torch.argmax(stage2_probs, dim=-1) + 2

    if mode == "hard":
        return torch.where(stage1_pred == STAGE1_MAP["NEGATIVE"], stage2_pred, stage1_pred)
    if mode == "strict":
        route = stage1_probs[:, STAGE1_MAP["NEGATIVE"]] >= threshold
        non_negative = torch.where(
            stage1_probs[:, STAGE1_MAP["POSITIVE"]] >= stage1_probs[:, STAGE1_MAP["NEUTRAL"]],
            torch.zeros_like(stage1_pred),
            torch.ones_like(stage1_pred),
        )
        return torch.where(route, stage2_pred, non_negative)
    if mode == "soft":
        final_probs = torch.stack(
            [
                stage1_probs[:, STAGE1_MAP["POSITIVE"]],
                stage1_probs[:, STAGE1_MAP["NEUTRAL"]],
                stage1_probs[:, STAGE1_MAP["NEGATIVE"]] * stage2_probs[:, STAGE2_MAP["CONSTRUCTIVE"]],
                stage1_probs[:, STAGE1_MAP["NEGATIVE"]] * stage2_probs[:, STAGE2_MAP["TOXIC"]],
            ],
            dim=-1,
        )
        return torch.argmax(final_probs, dim=-1)
    raise ValueError(f"Unknown inference mode: {mode}")


@torch.no_grad()
def evaluate(model, loader, device, loss_stage1, loss_stage2, stage2_weight: float, threshold: float) -> dict:
    model.eval()
    losses: list[float] = []
    stage1_true: list[np.ndarray] = []
    stage1_pred: list[np.ndarray] = []
    stage2_true: list[np.ndarray] = []
    stage2_pred: list[np.ndarray] = []
    final_true: list[np.ndarray] = []
    final_hard: list[np.ndarray] = []
    final_soft: list[np.ndarray] = []
    final_strict: list[np.ndarray] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y1 = batch["stage1_labels"].to(device)
        y2 = batch["stage2_labels"].to(device)
        yf = batch["final_labels"].to(device)

        logits1, logits2 = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_stage1(logits1, y1) + stage2_weight * loss_stage2(logits2, y2)
        losses.append(float(loss.detach().cpu().item()))

        stage1_true.append(y1.cpu().numpy())
        stage1_pred.append(torch.argmax(logits1, dim=-1).cpu().numpy())
        neg_mask = y2 != IGNORE_INDEX
        if bool(neg_mask.any()):
            stage2_true.append(y2[neg_mask].cpu().numpy())
            stage2_pred.append(torch.argmax(logits2[neg_mask], dim=-1).cpu().numpy())

        final_true.append(yf.cpu().numpy())
        final_hard.append(final_from_heads(logits1, logits2, "hard", threshold).cpu().numpy())
        final_soft.append(final_from_heads(logits1, logits2, "soft", threshold).cpu().numpy())
        final_strict.append(final_from_heads(logits1, logits2, "strict", threshold).cpu().numpy())

    y_final = np.concatenate(final_true)
    pred_hard = np.concatenate(final_hard)
    pred_soft = np.concatenate(final_soft)
    pred_strict = np.concatenate(final_strict)
    y_stage1 = np.concatenate(stage1_true)
    pred_stage1 = np.concatenate(stage1_pred)
    y_stage2 = np.concatenate(stage2_true) if stage2_true else np.asarray([], dtype=int)
    pred_stage2 = np.concatenate(stage2_pred) if stage2_pred else np.asarray([], dtype=int)

    return {
        "loss": float(np.mean(losses)),
        "stage1_accuracy": accuracy_score(y_stage1, pred_stage1),
        "stage1_macro_f1": f1_score(y_stage1, pred_stage1, average="macro", zero_division=0),
        "stage2_accuracy": accuracy_score(y_stage2, pred_stage2) if len(y_stage2) else 0.0,
        "stage2_macro_f1": f1_score(y_stage2, pred_stage2, average="macro", zero_division=0) if len(y_stage2) else 0.0,
        "hard_accuracy": accuracy_score(y_final, pred_hard),
        "hard_macro_f1": f1_score(y_final, pred_hard, average="macro", zero_division=0),
        "soft_accuracy": accuracy_score(y_final, pred_soft),
        "soft_macro_f1": f1_score(y_final, pred_soft, average="macro", zero_division=0),
        "strict_accuracy": accuracy_score(y_final, pred_strict),
        "strict_macro_f1": f1_score(y_final, pred_strict, average="macro", zero_division=0),
        "y_true": y_final,
        "hard_pred": pred_hard,
        "soft_pred": pred_soft,
        "strict_pred": pred_strict,
    }


def save_checkpoint(model, tokenizer, output_dir: Path, metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    torch.save(state, output_dir / "pytorch_model.bin")
    tokenizer.save_pretrained(output_dir)
    (output_dir / "shared_encoder_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_test_outputs(output_dir: Path, test: dict) -> pd.DataFrame:
    rows = []
    y_true = test["y_true"]
    for mode, pred_key in [("hard", "hard_pred"), ("soft", "soft_pred"), ("strict", "strict_pred")]:
        pred = test[pred_key]
        rows.append(
            {
                "mode": mode,
                "accuracy": accuracy_score(y_true, pred),
                "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
                "weighted_f1": f1_score(y_true, pred, average="weighted", zero_division=0),
            }
        )
        report = classification_report(
            y_true,
            pred,
            labels=list(range(len(FINAL_LABELS))),
            target_names=FINAL_LABELS,
            digits=4,
            zero_division=0,
        )
        (output_dir / f"classification_report_{mode}.txt").write_text(report, encoding="utf-8")
        cm = confusion_matrix(y_true, pred, labels=list(range(len(FINAL_LABELS))))
        pd.DataFrame(cm, index=FINAL_LABELS, columns=FINAL_LABELS).to_csv(
            output_dir / f"confusion_matrix_{mode}.csv",
            encoding="utf-8-sig",
        )

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "test_results.csv", index=False, encoding="utf-8-sig")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-csv", default=str(DEFAULT_DATA_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="auto", choices=["auto", "directml", "cuda", "cpu"])
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--stage2-loss-weight", type=float, default=1.0)
    parser.add_argument("--strict-threshold", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite and not args.eval_only:
        raise SystemExit(f"Output directory exists and is non-empty: {output_dir}. Use --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    device, device_label = pick_device(args.device)
    print(f"Device: {device_label}", flush=True)
    print(f"Output: {output_dir}", flush=True)

    df = load_dataframe(Path(args.data_csv), args.limit_train, args.limit_val, args.limit_test)
    print(
        "Rows: "
        f"train={int((df.split == 'train').sum())} "
        f"val={int((df.split == 'val').sum())} "
        f"test={int((df.split == 'test').sum())}",
        flush=True,
    )
    print(f"Final train label counts: {df[df.split == 'train']['label'].value_counts().to_dict()}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(HF_NAME)
    train_df, val_df, test_df, train_loader, val_loader, test_loader = make_loaders(args, tokenizer, df)

    stage1_weights = class_weights(train_df["stage1_id"].astype(int).tolist(), len(STAGE1_LABELS))
    stage2_train = train_df[train_df["stage2_id"] != IGNORE_INDEX]
    stage2_weights = class_weights(stage2_train["stage2_id"].astype(int).tolist(), len(STAGE2_LABELS))
    print(f"Stage 1 weights: {dict(zip(STAGE1_LABELS, [round(float(x), 3) for x in stage1_weights]))}", flush=True)
    print(f"Stage 2 weights: {dict(zip(STAGE2_LABELS, [round(float(x), 3) for x in stage2_weights]))}", flush=True)

    model = SharedEncoderMultiHead(HF_NAME, dropout=args.dropout)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    loss_stage1 = nn.CrossEntropyLoss(
        weight=torch.tensor(stage1_weights, dtype=torch.float32).to(device),
    )
    loss_stage2 = nn.CrossEntropyLoss(
        weight=torch.tensor(stage2_weights, dtype=torch.float32).to(device),
        ignore_index=IGNORE_INDEX,
    )

    if args.eval_only:
        checkpoint = output_dir / "pytorch_model.bin"
        if not checkpoint.exists():
            raise SystemExit(f"Checkpoint not found: {checkpoint}")
        print(f"Evaluating checkpoint: {checkpoint}", flush=True)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False))
        model.to(device)
        test = evaluate(
            model,
            test_loader,
            device,
            loss_stage1,
            loss_stage2,
            args.stage2_loss_weight,
            args.strict_threshold,
        )
        summary = {
            "eval_only": True,
            "test": {k: float(v) for k, v in test.items() if isinstance(v, (int, float, np.floating))},
        }
        (output_dir / "test_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        results = write_test_outputs(output_dir, test)
        print("\nTest results", flush=True)
        print(results.to_string(index=False), flush=True)
        print(f"\nSaved outputs to {output_dir.resolve()}", flush=True)
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.max_epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(
        f"Training shared encoder: epochs={args.max_epochs}, batch={args.batch_size}, "
        f"lr={args.lr}, steps/epoch={len(train_loader)}, warmup={warmup_steps}",
        flush=True,
    )

    best_val = -1.0
    best_epoch = -1
    stale = 0
    start = time.perf_counter()
    history: list[dict] = []

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y1 = batch["stage1_labels"].to(device)
            y2 = batch["stage2_labels"].to(device)
            logits1, logits2 = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_stage1(logits1, y1) + args.stage2_loss_weight * loss_stage2(logits2, y2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu().item()))

        val = evaluate(
            model,
            val_loader,
            device,
            loss_stage1,
            loss_stage2,
            args.stage2_loss_weight,
            args.strict_threshold,
        )
        score = float(val["hard_macro_f1"])
        train_loss = float(np.mean(losses))
        if not np.isfinite(train_loss):
            print(f"Stopping: non-finite train loss at epoch {epoch}: {train_loss}", flush=True)
            break

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "val_hard_accuracy": val["hard_accuracy"],
            "val_hard_macro_f1": val["hard_macro_f1"],
            "val_soft_accuracy": val["soft_accuracy"],
            "val_soft_macro_f1": val["soft_macro_f1"],
            "val_strict_accuracy": val["strict_accuracy"],
            "val_strict_macro_f1": val["strict_macro_f1"],
            "val_stage1_macro_f1": val["stage1_macro_f1"],
            "val_stage2_macro_f1": val["stage2_macro_f1"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False, encoding="utf-8-sig")

        if not np.isfinite(score):
            print(f"Stopping: non-finite validation score at epoch {epoch}: {score}", flush=True)
            break

        improved = score > best_val
        tag = " [NEW BEST]" if improved else ""
        print(
            f"Epoch {epoch:02d} | train_loss={row['train_loss']:.4f} "
            f"val_hard_acc={val['hard_accuracy']:.4f} val_hard_f1={score:.4f} "
            f"val_stage1_f1={val['stage1_macro_f1']:.4f} "
            f"val_stage2_f1={val['stage2_macro_f1']:.4f}{tag}",
            flush=True,
        )

        if improved:
            best_val = score
            best_epoch = epoch
            stale = 0
            save_checkpoint(
                model,
                tokenizer,
                output_dir,
                {
                    "architecture": "shared_encoder_multihead",
                    "hf_name": HF_NAME,
                    "best_epoch": best_epoch,
                    "best_val_hard_macro_f1": best_val,
                    "stage1_labels": STAGE1_LABELS,
                    "stage2_labels": STAGE2_LABELS,
                    "final_labels": FINAL_LABELS,
                    "args": vars(args),
                },
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}", flush=True)
                break

    elapsed = time.perf_counter() - start
    print(f"Training finished in {elapsed / 60:.1f} min. Best epoch={best_epoch}", flush=True)

    # The model in memory is usually the final epoch. Reload the best checkpoint.
    if best_epoch < 0:
        raise SystemExit("No valid checkpoint was saved; cannot evaluate test split.")

    best_model = SharedEncoderMultiHead(HF_NAME, dropout=args.dropout)
    best_model.resize_token_embeddings(len(tokenizer))
    best_model.load_state_dict(torch.load(output_dir / "pytorch_model.bin", map_location="cpu", weights_only=False))
    best_model.to(device)

    test = evaluate(
        best_model,
        test_loader,
        device,
        loss_stage1,
        loss_stage2,
        args.stage2_loss_weight,
        args.strict_threshold,
    )
    summary = {
        "best_epoch": best_epoch,
        "best_val_hard_macro_f1": best_val,
        "test": {k: float(v) for k, v in test.items() if isinstance(v, (int, float, np.floating))},
    }
    (output_dir / "test_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    results = write_test_outputs(output_dir, test)
    print("\nTest results", flush=True)
    print(results.to_string(index=False), flush=True)
    print(f"\nSaved outputs to {output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
