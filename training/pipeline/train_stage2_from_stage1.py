"""
train_stage2_from_stage1.py
===========================

Stage 2 trained with SEQUENTIAL FINE-TUNING (STILTs-style):
the encoder is initialised from the best Stage 1 checkpoint instead of the
raw pretrained MN-BERT.  Motivation:

    Stage 1 has already learnt Mongolian sentiment representations
    (POSITIVE / NEUTRAL / NEGATIVE) on the full 10k corpus.
    Stage 2 only sees the NEGATIVE subset (~2-3k samples), where weight
    initialisation quality dominates final performance.

Method
------
1.  Build a fresh `BertForSequenceClassification(num_labels=2)`.
2.  Load Stage 1's `pytorch_model.bin` and copy every tensor whose key
    *and* shape match the new model.  The classifier head (768→3 vs 768→2)
    is therefore skipped and kept at its random initialisation.
3.  Train with discriminative learning rates:
        encoder / embeddings / pooler  →  ENCODER_LR = 1e-5  (low)
        classifier head                →  HEAD_LR    = 3e-5  (high)
    Weight decay excludes biases and LayerNorm parameters, per the
    standard BERT fine-tuning recipe.
4.  Everything else (data, splits, seed, sampler weights, class weights,
    max_length, batch_size, max_epochs, patience) is IDENTICAL to
    `train_stage2.py` so the difference in test-set metrics can be
    attributed solely to the initialisation strategy.

Outputs
-------
    best_stage2_from_stage1_model/
        pytorch_model.bin       — best validation checkpoint
        config.json             — HF config
        tokenizer*              — tokenizer files
        experiment_meta.json    — hyperparameters + initialisation manifest
        training_log.json       — per-epoch train/val metrics
        test_results.json       — final test-set metrics (for A/B compare)
        test_predictions.csv    — id, text, true, pred, conf  (error analysis)

References
----------
Phang, J., Févry, T., & Bowman, S. R. (2018). Sentence Encoders on STILTs:
    Supplementary Training on Intermediate Labeled-data Tasks.
    arXiv:1811.01088.
Howard, J., & Ruder, S. (2018). Universal Language Model Fine-tuning for
    Text Classification (ULMFiT). ACL 2018.
"""

import os
import sys
import gc
import json
import time
import random
import hashlib
import numpy as np
import pandas as pd
import torch
import torch_directml
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score,
    precision_recall_fscore_support,
)
import warnings
warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ════════════════════════════════════════════════════════════════════════
# 0. PATHS, CONSTANTS, HYPERPARAMETERS
# ════════════════════════════════════════════════════════════════════════
WORK_DIR   = os.path.dirname(os.path.abspath(__file__))
# Data and Stage 1 checkpoint live in "sample scores/" — same as the
# best two-stage model used.
REPO_ROOT  = os.path.abspath(os.path.join(WORK_DIR, "..", ".."))
ASSET_DIR  = os.path.join(REPO_ROOT, "sample scores")

HF_NAME    = "tugstugi/bert-base-mongolian-cased"
DATA_CSV   = os.path.join(ASSET_DIR, "stage2_dataset.csv")
SAVE_DIR   = os.path.join(ASSET_DIR, "best_stage2_from_stage1_model")
STAGE1_DIR = os.path.join(ASSET_DIR, "best_stage1_model")
STAGE1_BIN = os.path.join(STAGE1_DIR, "pytorch_model.bin")
BASE_RESULTS_JSON = os.path.join(ASSET_DIR, "best_stage2_model",
                                 "test_results.json")

LOG_JSON   = os.path.join(SAVE_DIR, "training_log.json")
META_JSON  = os.path.join(SAVE_DIR, "experiment_meta.json")
RES_JSON   = os.path.join(SAVE_DIR, "test_results.json")
PRED_CSV   = os.path.join(SAVE_DIR, "test_predictions.csv")

STAGE2_MAP   = {"CONSTRUCTIVE": 0, "TOXIC": 1}
STAGE2_ID2L  = {v: k for k, v in STAGE2_MAP.items()}
LABEL_NAMES  = ["CONSTRUCTIVE", "TOXIC"]
NUM_LABELS   = 2

# ── Hyperparameters: IDENTICAL to train_stage2.py (the best 2-stage run) ──
# Only the model INITIALISATION differs (Stage 1 transfer vs raw pretrained).
LR                 = 2e-5
WEIGHT_DECAY       = 0.01
WARMUP_RATIO       = 0.06
GRAD_CLIP          = 1.0
CLASSIFIER_DROPOUT = 0.15
MAX_EPOCHS         = 20
PATIENCE           = 5
VAL_TEST_BATCH     = 32
SEED               = 42

# ════════════════════════════════════════════════════════════════════════
# 1. REPRODUCIBILITY
# ════════════════════════════════════════════════════════════════════════
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
try:
    torch.use_deterministic_algorithms(False)  # DirectML lacks some det. kernels
except Exception:
    pass

print("─" * 72)
print(" Stage 2 — sequential fine-tuning experiment (init from Stage 1)")
print("─" * 72)
print(f"PyTorch       : {torch.__version__}")
print(f"NumPy         : {np.__version__}")
try:
    print(f"DirectML      : {torch_directml.__version__}")
except Exception:
    print(f"DirectML      : (version unknown)")
print(f"Seed          : {SEED}")

# ════════════════════════════════════════════════════════════════════════
# 2. SAFETY CHECKS
# ════════════════════════════════════════════════════════════════════════
if os.path.isdir(SAVE_DIR) and os.listdir(SAVE_DIR):
    print(f"ERROR: {SAVE_DIR} already exists and is non-empty. Stop.")
    raise SystemExit(1)
os.makedirs(SAVE_DIR, exist_ok=True)

if not os.path.exists(STAGE1_BIN):
    print(f"ERROR: Stage 1 checkpoint not found at\n  {STAGE1_BIN}")
    print("       Run training/pipeline/train_stage1.py first.")
    raise SystemExit(1)

if not os.path.exists(DATA_CSV):
    print(f"ERROR: {DATA_CSV} not found. Run prepare_data.py first.")
    raise SystemExit(1)

# SHA1 of the Stage 1 binary so that we can later verify which Stage 1
# weights produced these Stage 2 results.
def _sha1(path, chunk=1024 * 1024):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()

STAGE1_SHA1 = _sha1(STAGE1_BIN)
print(f"Stage 1 ckpt  : {STAGE1_BIN}")
print(f"Stage 1 SHA1  : {STAGE1_SHA1}")

try:
    device = torch_directml.device()
    _ = torch.zeros(1, device=device)
    print(f"Device        : {device}")
except Exception as e:
    print(f"ERROR: DirectML failed to initialise — {e}")
    raise SystemExit(1)

# ════════════════════════════════════════════════════════════════════════
# 3. DATA LOADING (identical to train_stage2.py)
# ════════════════════════════════════════════════════════════════════════
df = pd.read_csv(DATA_CSV, encoding="utf-8-sig")
required = {"id", "text_normalized", "label_stage2", "split", "source"}
missing = required - set(df.columns)
if missing:
    print(f"ERROR: missing cols in stage2_dataset.csv: {missing}")
    raise SystemExit(1)

df["input_text"] = df.apply(
    lambda r: f"[{r['source']}] {str(r['text_normalized'] or '')}", axis=1
)
df["label_id"] = df["label_stage2"].map(STAGE2_MAP)

print(f"\nStage 2 rows  : total={len(df)} "
      f"train={int((df.split=='train').sum())} "
      f"val={int((df.split=='val').sum())} "
      f"test={int((df.split=='test').sum())}")
print(f"Hyperparams   : lr={LR}  wd={WEIGHT_DECAY}  warmup={WARMUP_RATIO}  "
      f"patience={PATIENCE}  (identical to train_stage2.py)")
print("✅ [T.1] data loaded")

tokenizer = AutoTokenizer.from_pretrained(HF_NAME)
print("✅ [T.2] tokenizer loaded")

print("Computing p95 token length on Stage 2 train split ...")
train_texts = df[df["split"] == "train"]["input_text"].tolist()
tok_lens = []
for i in range(0, len(train_texts), 256):
    enc = tokenizer(train_texts[i:i+256], add_special_tokens=True,
                    truncation=False, padding=False)
    tok_lens.extend(len(x) for x in enc["input_ids"])
tok_lens = np.array(tok_lens, dtype=np.int64)
p50 = int(np.percentile(tok_lens, 50))
p95 = int(np.percentile(tok_lens, 95))
p99 = int(np.percentile(tok_lens, 99))
print(f"  token lengths: p50={p50} p95={p95} p99={p99} max={int(tok_lens.max())}")
if p95 <= 256:
    MAX_LENGTH = 256; BATCH_SIZE = 16
else:
    MAX_LENGTH = 512; BATCH_SIZE = 8
print(f"  → chosen max_length={MAX_LENGTH}, batch_size={BATCH_SIZE}")
print("✅ [T.3] max_length / batch_size decided")

def encode(texts, labels):
    enc = tokenizer(texts, padding="max_length", truncation=True,
                    max_length=MAX_LENGTH, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"], torch.tensor(labels, dtype=torch.long)

class DS(Dataset):
    def __init__(self, ids, mask, y):
        self.ids, self.mask, self.y = ids, mask, y
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "attention_mask": self.mask[i], "labels": self.y[i]}

print("Tokenizing Stage 2 splits ...")
tr_df = df[df.split == "train"].reset_index(drop=True)
va_df = df[df.split == "val"  ].reset_index(drop=True)
te_df = df[df.split == "test" ].reset_index(drop=True)
tr_ids, tr_mask, tr_y = encode(tr_df["input_text"].tolist(), tr_df["label_id"].tolist())
va_ids, va_mask, va_y = encode(va_df["input_text"].tolist(), va_df["label_id"].tolist())
te_ids, te_mask, te_y = encode(te_df["input_text"].tolist(), te_df["label_id"].tolist())
train_ds = DS(tr_ids, tr_mask, tr_y)
val_ds   = DS(va_ids, va_mask, va_y)
test_ds  = DS(te_ids, te_mask, te_y)
print(f"  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
print("✅ [T.4] tokenized")

train_labels_np = tr_y.numpy()
counts = np.bincount(train_labels_np, minlength=NUM_LABELS).astype(np.float64)
print(f"Stage 2 train label counts: "
      f"{dict(zip(LABEL_NAMES, counts.astype(int).tolist()))}")
total = counts.sum()
class_weights = (total / (NUM_LABELS * counts))
print(f"Class weights (CE): "
      f"{dict(zip(LABEL_NAMES, [round(w, 3) for w in class_weights.tolist()]))}")
sample_weights = np.array([class_weights[y] for y in train_labels_np], dtype=np.float64)
print("✅ [T.5] class weights computed")

# ════════════════════════════════════════════════════════════════════════
# 4. MODEL CONSTRUCTION — TRANSFER INIT FROM STAGE 1
# ════════════════════════════════════════════════════════════════════════
def transfer_init_model():
    """
    Build a 2-class BERT, then overwrite every parameter that exists in
    Stage 1 with the SAME key AND the SAME shape.  Shape-mismatched keys
    (i.e. the 3-way classifier head) are skipped, so the head stays at
    its random init.  Missing keys (any tensor newly introduced by the
    2-way head) are also kept at their default init.
    """
    config = AutoConfig.from_pretrained(HF_NAME, num_labels=NUM_LABELS)
    config.classifier_dropout = CLASSIFIER_DROPOUT
    m = AutoModelForSequenceClassification.from_pretrained(HF_NAME, config=config)
    m.resize_token_embeddings(len(tokenizer))

    stage1_state = torch.load(STAGE1_BIN, map_location="cpu")
    target_state = m.state_dict()

    loaded, skipped = [], []
    filtered = {}
    for k, v in stage1_state.items():
        if k not in target_state:
            skipped.append((k, "not present in 2-class model"))
            continue
        if target_state[k].shape != v.shape:
            skipped.append((k, f"shape mismatch "
                              f"{tuple(v.shape)} vs {tuple(target_state[k].shape)}"))
            continue
        filtered[k] = v
        loaded.append(k)

    missing, unexpected = m.load_state_dict(filtered, strict=False)
    print(f"\n  ✓ transferred {len(loaded)} tensors from Stage 1")
    if skipped:
        print("  ✗ skipped (kept at fresh init):")
        for k, why in skipped:
            print(f"      - {k:60s}  [{why}]")
    if unexpected:
        print(f"  unexpected keys in load_state_dict: {unexpected}")
    return m, loaded, skipped

def is_oom(err: BaseException) -> bool:
    s = str(err).lower()
    return any(k in s for k in [
        "out of memory", "oom", "allocation", "alloc failed",
        "cuda error: out", "directml", "hipout"
    ])

@torch.no_grad()
def evaluate(model, loader, return_probs=False):
    model.eval()
    losses, y_true, y_pred, all_probs = [], [], [], []
    ce = torch.nn.CrossEntropyLoss()
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attn).logits
        losses.append(ce(logits, labels).item())
        probs = torch.softmax(logits, dim=-1)
        y_true.append(labels.cpu().numpy())
        y_pred.append(torch.argmax(logits, dim=-1).cpu().numpy())
        if return_probs:
            all_probs.append(probs.cpu().numpy())
    yt = np.concatenate(y_true); yp = np.concatenate(y_pred)
    out = {"loss": float(np.mean(losses)),
           "macro_f1": f1_score(yt, yp, average="macro"),
           "accuracy": accuracy_score(yt, yp),
           "y_true": yt, "y_pred": yp}
    if return_probs:
        out["probs"] = np.concatenate(all_probs, axis=0)
    return out

# (Optimizer setup matches baseline train_stage2.py: single AdamW with one LR.)

# ════════════════════════════════════════════════════════════════════════
# 6. TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════
def do_full_training(batch_size):
    print(f"\nTraining Stage 2 (transfer init) — batch_size={batch_size}, "
          f"max_length={MAX_LENGTH}, lr={LR}, patience={PATIENCE}")

    # Match baseline: unseeded sampler (same as train_stage2.py).
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=VAL_TEST_BATCH,
                            shuffle=False, num_workers=0)

    model, loaded_keys, skipped_keys = transfer_init_model()
    model = model.to(device)

    # IDENTICAL optimizer to train_stage2.py:
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps  = len(train_loader) * MAX_EPOCHS
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    cw_tensor   = torch.tensor(class_weights, dtype=torch.float32).to(device)
    ce_weighted = torch.nn.CrossEntropyLoss(weight=cw_tensor)

    print(f"  steps/epoch={len(train_loader)}  total={total_steps}  "
          f"warmup={warmup_steps}")

    epoch_log = []
    best_val_f1 = -1.0
    best_epoch  = -1
    patience_counter = 0
    t_start = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tr_losses, tr_true, tr_pred = [], [], []
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attn      = batch["attention_mask"].to(device)
            labels    = batch["labels"].to(device)
            logits    = model(input_ids=input_ids, attention_mask=attn).logits
            loss      = ce_weighted(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            tr_losses.append(loss.item())
            tr_true.append(labels.detach().cpu().numpy())
            tr_pred.append(torch.argmax(logits, dim=-1).detach().cpu().numpy())

        tr_loss = float(np.mean(tr_losses))
        tr_f1   = f1_score(np.concatenate(tr_true),
                           np.concatenate(tr_pred), average="macro")
        val     = evaluate(model, val_loader)
        new_best = val["macro_f1"] > best_val_f1
        tag      = " [NEW BEST]" if new_best else ""

        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} "
              f"train_macro_f1={tr_f1:.4f} | "
              f"val_loss={val['loss']:.4f} val_macro_f1={val['macro_f1']:.4f}{tag}")

        epoch_log.append({
            "epoch":           epoch,
            "train_loss":      tr_loss,
            "train_macro_f1":  tr_f1,
            "val_loss":        val["loss"],
            "val_macro_f1":    val["macro_f1"],
            "val_accuracy":    val["accuracy"],
            "is_best":         bool(new_best),
            "elapsed_sec":     round(time.time() - t_start, 2),
        })

        if new_best:
            best_val_f1 = val["macro_f1"]
            best_epoch  = epoch
            patience_counter = 0
            cpu_sd = {k: v.detach().cpu().clone()
                      for k, v in model.state_dict().items()}
            torch.save(cpu_sd, os.path.join(SAVE_DIR, "pytorch_model.bin"))
            model.config.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
            print(f"  >> saved best checkpoint to {SAVE_DIR}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stop: patience {PATIENCE} hit at epoch {epoch}. "
                      f"Best epoch was {best_epoch} "
                      f"(val_macro_f1={best_val_f1:.4f}).")
                break

    with open(LOG_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "epochs":            epoch_log,
            "best_epoch":        best_epoch,
            "best_val_macro_f1": best_val_f1,
        }, f, ensure_ascii=False, indent=2)
    return best_val_f1, best_epoch

# ════════════════════════════════════════════════════════════════════════
# 7. WRITE EXPERIMENT MANIFEST BEFORE TRAINING (reproducibility)
# ════════════════════════════════════════════════════════════════════════
experiment_meta = {
    "experiment_name": "stage2_from_stage1_transfer",
    "description": (
        "STILTs-style sequential fine-tuning. Stage 2 BertForSequenceClassification "
        "(num_labels=2) is initialised by loading every key from the Stage 1 "
        "checkpoint whose shape matches (i.e. everything except the 3-way "
        "classifier head, which is kept at its random init). "
        "All hyperparameters and data are IDENTICAL to train_stage2.py; "
        "only the model initialisation differs."
    ),
    "hf_name":            HF_NAME,
    "stage1_checkpoint":  STAGE1_BIN,
    "stage1_sha1":        STAGE1_SHA1,
    "num_labels":         NUM_LABELS,
    "label_names":        LABEL_NAMES,
    "hyperparameters": {
        "lr":                  LR,
        "weight_decay":        WEIGHT_DECAY,
        "warmup_ratio":        WARMUP_RATIO,
        "grad_clip":           GRAD_CLIP,
        "classifier_dropout":  CLASSIFIER_DROPOUT,
        "max_epochs":          MAX_EPOCHS,
        "patience":            PATIENCE,
        "seed":                SEED,
        "max_length":          MAX_LENGTH,
        "batch_size":          BATCH_SIZE,
    },
    "data_csv":      DATA_CSV,
    "class_weights": dict(zip(LABEL_NAMES,
                              [float(w) for w in class_weights.tolist()])),
}
with open(META_JSON, "w", encoding="utf-8") as f:
    json.dump(experiment_meta, f, ensure_ascii=False, indent=2)

# ════════════════════════════════════════════════════════════════════════
# 8. RUN TRAINING (with OOM fallback to half batch size)
# ════════════════════════════════════════════════════════════════════════
current_batch = BATCH_SIZE
try:
    best_val_f1, best_epoch = do_full_training(current_batch)
except Exception as e:
    if is_oom(e):
        halved = max(1, current_batch // 2)
        print(f"\n⚠️  OOM at batch_size={current_batch}. Retrying at {halved}.")
        gc.collect()
        current_batch = halved
        try:
            best_val_f1, best_epoch = do_full_training(current_batch)
        except Exception as e2:
            if is_oom(e2):
                print(f"\nERROR: OOM persists at batch_size={current_batch}.")
                raise SystemExit(1)
            raise
    else:
        raise

print(f"\nTraining complete.  best_epoch={best_epoch}  "
      f"best_val_macro_f1={best_val_f1:.4f}")
print("✅ [T.6] training done")

# ════════════════════════════════════════════════════════════════════════
# 9. TEST-SET EVALUATION
# ════════════════════════════════════════════════════════════════════════
print("\nLoading best transfer-init checkpoint for test evaluation ...")
config = AutoConfig.from_pretrained(HF_NAME, num_labels=NUM_LABELS)
config.classifier_dropout = CLASSIFIER_DROPOUT
model = AutoModelForSequenceClassification.from_pretrained(HF_NAME, config=config)
model.resize_token_embeddings(len(tokenizer))
state = torch.load(os.path.join(SAVE_DIR, "pytorch_model.bin"),
                   map_location="cpu")
model.load_state_dict(state)
model.to(device); model.eval()

test_loader = DataLoader(test_ds, batch_size=VAL_TEST_BATCH,
                         shuffle=False, num_workers=0)
test = evaluate(model, test_loader, return_probs=True)
y_true = test["y_true"]; y_pred = test["y_pred"]; probs = test["probs"]

acc          = accuracy_score(y_true, y_pred)
macro_f1     = f1_score(y_true, y_pred, average="macro")
weighted_f1  = f1_score(y_true, y_pred, average="weighted")
p_c, r_c, f_c, n_c = precision_recall_fscore_support(
    y_true, y_pred, labels=list(range(NUM_LABELS)), zero_division=0)

print("\n" + "═" * 72)
print(" Stage 2 (TRANSFER from Stage 1) — test-set results")
print("═" * 72)
print(f"test_loss   : {test['loss']:.4f}")
print(f"Accuracy    : {acc:.4f}")
print(f"Macro F1    : {macro_f1:.4f}")
print(f"Weighted F1 : {weighted_f1:.4f}")
print("\nclassification_report:")
print(classification_report(y_true, y_pred,
      labels=list(range(NUM_LABELS)),
      target_names=LABEL_NAMES, digits=4))
print("Confusion matrix (rows=true, cols=pred):")
cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_LABELS)))
print("              " + " ".join(f"{n:>13s}" for n in LABEL_NAMES))
for i, row in enumerate(cm):
    cells = " ".join(f"{v:>13d}" for v in row)
    print(f"  {LABEL_NAMES[i]:<13s} {cells}")

# Save test results JSON
test_results = {
    "test_loss":     float(test["loss"]),
    "accuracy":      float(acc),
    "macro_f1":      float(macro_f1),
    "weighted_f1":   float(weighted_f1),
    "per_class": {
        LABEL_NAMES[i]: {
            "precision": float(p_c[i]),
            "recall":    float(r_c[i]),
            "f1":        float(f_c[i]),
            "support":   int(n_c[i]),
        } for i in range(NUM_LABELS)
    },
    "confusion_matrix":  cm.tolist(),
    "best_epoch":        best_epoch,
    "best_val_macro_f1": float(best_val_f1),
    "init_strategy":     "stage1_transfer",
}
with open(RES_JSON, "w", encoding="utf-8") as f:
    json.dump(test_results, f, ensure_ascii=False, indent=2)

# Save predictions for downstream error analysis
pred_df = te_df[["id", "input_text", "label_stage2"]].copy()
pred_df["true_id"]    = y_true
pred_df["pred_id"]    = y_pred
pred_df["pred_label"] = [STAGE2_ID2L[int(i)] for i in y_pred]
pred_df["confidence"] = probs.max(axis=1)
pred_df["p_constructive"] = probs[:, STAGE2_MAP["CONSTRUCTIVE"]]
pred_df["p_toxic"]        = probs[:, STAGE2_MAP["TOXIC"]]
pred_df["correct"]    = (pred_df["true_id"] == pred_df["pred_id"])
pred_df.to_csv(PRED_CSV, index=False, encoding="utf-8-sig")
print(f"\n  predictions saved: {PRED_CSV}")
print(f"  results JSON     : {RES_JSON}")

# ════════════════════════════════════════════════════════════════════════
# 10. A/B COMPARISON vs BASELINE (if baseline test_results.json exists)
# ════════════════════════════════════════════════════════════════════════
if os.path.exists(BASE_RESULTS_JSON):
    with open(BASE_RESULTS_JSON, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    print("\n" + "═" * 72)
    print(" A/B Comparison — baseline (raw init) vs transfer (Stage 1 init)")
    print("═" * 72)
    def _d(a, b): return float("nan") if a is None else (b - a)
    print(f"{'metric':<22} {'baseline':>12} {'transfer':>12} {'Δ':>10}")
    print("-" * 60)
    for key in ["accuracy", "macro_f1", "weighted_f1"]:
        bv = baseline.get(key)
        tv = test_results[key]
        delta = "        n/a" if bv is None else f"{_d(bv, tv):>+10.4f}"
        bv_s  = "n/a" if bv is None else f"{bv:.4f}"
        print(f"{key:<22} {bv_s:>12} {tv:>12.4f} {delta}")
    print()
    for cls in LABEL_NAMES:
        bf = baseline.get("per_class", {}).get(cls, {}).get("f1")
        tf = test_results["per_class"][cls]["f1"]
        delta = "        n/a" if bf is None else f"{_d(bf, tf):>+10.4f}"
        bf_s  = "n/a" if bf is None else f"{bf:.4f}"
        print(f"  {cls+' F1':<20} {bf_s:>12} {tf:>12.4f} {delta}")
else:
    print(f"\nℹ Baseline results not found at\n  {BASE_RESULTS_JSON}")
    print("  (Re-)run train_stage2.py — it will save test_results.json — to "
          "enable the automatic A/B comparison.")

print("\n✅ Stage 2 transfer-init experiment complete.")
print(f"   Checkpoint : {SAVE_DIR}")
print(f"   Log        : {LOG_JSON}")
print(f"   Manifest   : {META_JSON}")
