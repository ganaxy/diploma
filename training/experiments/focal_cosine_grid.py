import os
import re
import sys
import gc
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_directml
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
)
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score
)
import warnings
warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORK_DIR)

HF_NAME      = "tugstugi/bert-base-mongolian-cased"
MERGED_CSV   = os.path.join(WORK_DIR, "relabeled_v7_normalized.csv")
GRID_CSV     = os.path.join(WORK_DIR, "grid_search_results.csv")
CONFIG_TXT   = os.path.join(WORK_DIR, "process of training.txt")

LABEL_MAP  = {"POSITIVE": 0, "NEUTRAL": 1, "CONSTRUCTIVE": 2, "TOXIC": 3}
ID2LABEL   = {v: k for k, v in LABEL_MAP.items()}
LABEL_NAMES= ["POSITIVE", "NEUTRAL", "CONSTRUCTIVE", "TOXIC"]

DEFAULT_EPOCHS     = 20

WEIGHT_DECAY       = 0.01
GRAD_CLIP          = 1.0
CLASSIFIER_DROPOUT = 0.15
VAL_TEST_BATCH     = 32
MAX_LENGTH         = 256
SEED               = 42

FOCAL_GAMMA        = 2.0

BATCH_GRID = [16, 32]
LR_GRID    = [2e-5, 3e-5, 5e-5]

torch.manual_seed(SEED)
np.random.seed(SEED)

def load_txt_config(path, defaults):
    cfg = dict(defaults)
    if not os.path.exists(path):
        return cfg
    try:
        with open(path, "r", encoding="utf-8") as fh:
            txt = fh.read()
    except Exception:
        return cfg

    def grab(keys, cast):
        for k in keys:
            m = re.search(rf"(?mi)^\s*{k}\s*[:=]\s*([0-9eE\.\-\+]+)", txt)
            if m:
                try:
                    return cast(m.group(1))
                except Exception:
                    pass
        return None

    lr_val  = grab(["lr", "learning_rate"], float)
    bs_val  = grab(["batch_size", "bs"], int)
    ep_val  = grab(["epochs", "max_epochs"], int)

    if lr_val is not None: cfg["lr"] = lr_val
    if bs_val is not None: cfg["batch_size"] = bs_val
    if ep_val is not None: cfg["epochs"] = ep_val
    return cfg

cfg = load_txt_config(CONFIG_TXT, {
    "lr":         DEFAULT_LR,
    "batch_size": DEFAULT_BATCH_SIZE,
    "epochs":     DEFAULT_EPOCHS,
})
MAX_EPOCHS = cfg["epochs"]
print(f"Config: lr_default={cfg['lr']} bs_default={cfg['batch_size']} "
      f"epochs={MAX_EPOCHS}  (grid overrides lr & batch_size)")

try:
    device = torch_directml.device()
    _ = torch.zeros(1, device=device)
    print(f"DirectML device initialized: {device}")
except Exception as e:
    print(f"ERROR: DirectML failed to initialize — {e}")
    raise SystemExit(1)

if not os.path.exists(MERGED_CSV):
    print(f"ERROR: {MERGED_CSV} not found"); raise SystemExit(1)

df = pd.read_csv(MERGED_CSV, encoding="utf-8-sig")
required = {"id", "text_light_clean", "text_normalized", "label", "split", "source"}
missing = required - set(df.columns)
if missing:
    print(f"ERROR: missing columns: {missing}"); raise SystemExit(1)

df["input_text"] = df.apply(
    lambda r: f"[{r['source']}] {str(r['text_normalized'] or '')}", axis=1
)
df["label_id"] = df["label"].map(LABEL_MAP)
print(f"Rows: total={len(df)} "
      f"train={int((df.split=='train').sum())} "
      f"val={int((df.split=='val').sum())} "
      f"test={int((df.split=='test').sum())}")

tokenizer = AutoTokenizer.from_pretrained(HF_NAME)

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

print("Tokenizing splits ...")
tr_ids, tr_mask, tr_y = encode(df[df.split=="train"]["input_text"].tolist(),
                               df[df.split=="train"]["label_id"].tolist())
va_ids, va_mask, va_y = encode(df[df.split=="val"]["input_text"].tolist(),
                               df[df.split=="val"]["label_id"].tolist())
te_ids, te_mask, te_y = encode(df[df.split=="test"]["input_text"].tolist(),
                               df[df.split=="test"]["label_id"].tolist())
train_ds = DS(tr_ids, tr_mask, tr_y)
val_ds   = DS(va_ids, va_mask, va_y)
test_ds  = DS(te_ids, te_mask, te_y)
print(f"  train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

train_labels_np = tr_y.numpy()
counts = np.bincount(train_labels_np, minlength=4).astype(np.float64)
print(f"Train label counts: {dict(zip(LABEL_NAMES, counts.astype(int).tolist()))}")
total = counts.sum()
print(f"Class weights (alpha for focal): "
      f"{dict(zip(LABEL_NAMES, [round(w,3) for w in class_weights.tolist()]))}")
sample_weights = np.array([class_weights[y] for y in train_labels_np], dtype=np.float64)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", alpha if alpha is not None else torch.tensor([]),
                             persistent=False)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.alpha.numel() > 0:
            loss = -alpha_t * focal_term * log_pt
        else:
            loss = -focal_term * log_pt
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

def fresh_model():
    config = AutoConfig.from_pretrained(HF_NAME, num_labels=4)
    config.classifier_dropout = CLASSIFIER_DROPOUT
    m = AutoModelForSequenceClassification.from_pretrained(HF_NAME, config=config)
    m.resize_token_embeddings(len(tokenizer))
    return m

def is_oom(err: BaseException) -> bool:
    s = str(err).lower()
    return any(k in s for k in [
        "out of memory",
        "oom",
        "alloc failed",
        "cuda error: out of memory",
        "hip out of memory",
    ])

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    losses, y_true, y_pred = [], [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attn).logits
        losses.append(ce(logits, labels).item())
        y_true.append(labels.cpu().numpy())
        y_pred.append(torch.argmax(logits, dim=-1).cpu().numpy())
    yt = np.concatenate(y_true); yp = np.concatenate(y_pred)
    return {"loss": float(np.mean(losses)),
            "macro_f1": f1_score(yt, yp, average="macro"),
            "accuracy": accuracy_score(yt, yp),
            "y_true": yt, "y_pred": yp}

def do_full_training(batch_size: int, lr: float, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n── Training combo: batch_size={batch_size}, lr={lr:.0e}, "
          f"epochs={MAX_EPOCHS}, patience={PATIENCE}")

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=VAL_TEST_BATCH, shuffle=False, num_workers=0)

    model = fresh_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    total_steps  = len(train_loader) * MAX_EPOCHS
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(f"  Scheduler: linear warmup ({warmup_steps} steps, "
          f"{100*WARMUP_RATIO:.0f}%) → cosine decay to 0 over {total_steps} steps")

    focal_fn = FocalLoss(
        alpha=torch.tensor(class_weights, dtype=torch.float32),
        gamma=FOCAL_GAMMA,
        reduction="mean",
    )
    print(f"  Loss: FocalLoss(gamma={FOCAL_GAMMA}, alpha=class_weights)")

    best_val_f1 = -1.0; best_epoch = -1; patience_counter = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tr_losses = []; tr_true, tr_pred = [], []
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            tr_losses.append(loss.item())
            tr_true.append(labels.detach().cpu().numpy())
            tr_pred.append(torch.argmax(logits, dim=-1).detach().cpu().numpy())

        tr_loss = float(np.mean(tr_losses))
        tr_f1 = f1_score(np.concatenate(tr_true), np.concatenate(tr_pred), average="macro")
        val = evaluate(model, val_loader)
        val_loss = val["loss"]; val_f1 = val["macro_f1"]

        new_best = val_f1 > best_val_f1
        tag = " [NEW BEST]" if new_best else ""
        print(f"  Epoch {epoch:02d} | train_loss={tr_loss:.4f} train_macro_f1={tr_f1:.4f} | "
              f"val_loss={val_loss:.4f} val_macro_f1={val_f1:.4f}{tag}")

        if new_best:
            best_val_f1 = val_f1; best_epoch = epoch; patience_counter = 0
            cpu_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(cpu_state_dict, os.path.join(save_dir, "pytorch_model.bin"))
            model.config.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stop: patience {PATIENCE} hit at epoch {epoch}. "
                      f"Best epoch was {best_epoch} (val_macro_f1={best_val_f1:.4f}).")
                break
    return best_val_f1, best_epoch

def reload_and_eval(save_dir: str):
    config = AutoConfig.from_pretrained(HF_NAME, num_labels=4)
    config.classifier_dropout = CLASSIFIER_DROPOUT
    m = AutoModelForSequenceClassification.from_pretrained(HF_NAME, config=config)
    m.resize_token_embeddings(len(tokenizer))
    state = torch.load(os.path.join(save_dir, "pytorch_model.bin"), map_location="cpu")
    m.load_state_dict(state)
    m.to(device); m.eval()
    vl = DataLoader(val_ds,  batch_size=VAL_TEST_BATCH, shuffle=False, num_workers=0)
    tl = DataLoader(test_ds, batch_size=VAL_TEST_BATCH, shuffle=False, num_workers=0)
    val_m  = evaluate(m, vl)
    test_m = evaluate(m, tl)
    del m; gc.collect()
    return val_m, test_m

combinations = [
    {"combo_id": i + 1, "batch_size": bs, "lr": lr}
    for i, (bs, lr) in enumerate(itertools.product(BATCH_GRID, LR_GRID))
]
print(f"\nGrid search: {len(combinations)} combinations "
      f"({len(BATCH_GRID)}×{len(LR_GRID)})")
for c in combinations:
    print(f"  combo {c['combo_id']:>2d}: bs={c['batch_size']:>3d}  lr={c['lr']:.0e}")

results = []
for c in combinations:
    bs, lr, cid = c["batch_size"], c["lr"], c["combo_id"]
    save_dir = os.path.join(
        WORK_DIR, f"grid_ckpt_bs{bs}_lr{lr:.0e}".replace("+", "")
    )
    if os.path.isdir(save_dir) and os.listdir(save_dir):
        print(f"\n⚠️  {save_dir} already exists and non-empty. "
              f"SKIPPING combo {cid} to avoid overwrite. "
              f"Delete the dir or rename to re-run.")
        continue

    print(f"\n{'='*70}\n GRID COMBO {cid}/{len(combinations)}  "
          f"bs={bs}  lr={lr:.0e}  dir={os.path.basename(save_dir)}\n{'='*70}")

    current_bs = bs
    try:
        best_val_f1, best_epoch = do_full_training(current_bs, lr, save_dir)
    except Exception as e:
        if is_oom(e):
            halved = max(1, current_bs // 2)
            already = any(
                r["batch_size"] == halved and r["learning_rate"] == lr
                and r["status"] == "ok"
                for r in results
            )
            if already:
                print(f"\n⚠️  OOM at bs={current_bs}. "
                      f"Halved (bs={halved}, lr={lr:.0e}) already recorded. "
                      f"Skipping combo.")
                results.append({
                    "combo_id": cid, "batch_size": bs, "learning_rate": lr,
                    "val_accuracy": None, "val_macro_f1": None,
                    "test_accuracy": None, "test_macro_f1": None,
                    "best_epoch": None,
                    "status": f"OOM @ bs={bs} (dup of bs={halved})",
                    "checkpoint_dir": os.path.basename(save_dir),
                })
                pd.DataFrame(results).to_csv(GRID_CSV, index=False, encoding="utf-8-sig")
                continue

            print(f"\n⚠️  OOM at bs={current_bs}. Retrying at {halved}.")
            gc.collect()
            current_bs = halved
            try:
                best_val_f1, best_epoch = do_full_training(current_bs, lr, save_dir)
            except Exception as e2:
                if is_oom(e2):
                    print(f"\nERROR: OOM persists at bs={current_bs}. Skipping combo.")
                    results.append({
                        "combo_id": cid, "batch_size": bs, "learning_rate": lr,
                        "val_accuracy": None, "val_macro_f1": None,
                        "test_accuracy": None, "test_macro_f1": None,
                        "best_epoch": None, "status": f"OOM @ bs={current_bs}",
                        "checkpoint_dir": os.path.basename(save_dir),
                    })
                    pd.DataFrame(results).to_csv(GRID_CSV, index=False, encoding="utf-8-sig")
                    continue
                raise
        else:
            raise

    val_m, test_m = reload_and_eval(save_dir)
    row = {
        "combo_id":       cid,
        "batch_size":     bs,
        "learning_rate":  lr,
        "val_accuracy":   round(val_m["accuracy"], 4),
        "val_macro_f1":   round(val_m["macro_f1"], 4),
        "test_accuracy":  round(test_m["accuracy"], 4),
        "test_macro_f1":  round(test_m["macro_f1"], 4),
        "best_epoch":     best_epoch,
        "status":         "ok",
        "checkpoint_dir": os.path.basename(save_dir),
    }
    results.append(row)
    pd.DataFrame(results).to_csv(GRID_CSV, index=False, encoding="utf-8-sig")
    print(f"  → val_acc={row['val_accuracy']:.4f} val_macro_f1={row['val_macro_f1']:.4f} | "
          f"test_acc={row['test_accuracy']:.4f} test_macro_f1={row['test_macro_f1']:.4f}")

if not results:
    print("\nNo grid results recorded. Aborting summary.")
    raise SystemExit(0)

res_df = pd.DataFrame(results)
res_df.to_csv(GRID_CSV, index=False, encoding="utf-8-sig")

ok_df = res_df[res_df["status"] == "ok"].copy()
print("\n" + "═"*72)
print(" GRID SEARCH SUMMARY")
print("═"*72)
print(f"{'combo':>5s} {'bs':>4s} {'lr':>7s} "
      f"{'val_acc':>8s} {'val_f1':>8s} {'test_acc':>9s} {'test_f1':>8s} "
      f"{'epoch':>6s} {'status':<16s}")
def _fmt(v, spec, width):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "--".rjust(width)
    return format(v, spec).rjust(width)

for _, r in res_df.iterrows():
    va = _fmt(r["val_accuracy"],  ".4f", 8)
    vf = _fmt(r["val_macro_f1"],  ".4f", 8)
    ta = _fmt(r["test_accuracy"], ".4f", 9)
    tf = _fmt(r["test_macro_f1"], ".4f", 8)
    be = "--".rjust(6) if (r["best_epoch"] is None or
                            (isinstance(r["best_epoch"], float) and pd.isna(r["best_epoch"]))) \
         else f"{int(r['best_epoch']):>6d}"
    print(f"{int(r['combo_id']):>5d} {int(r['batch_size']):>4d} "
          f"{r['learning_rate']:>7.0e} "
          f"{va} {vf} {ta} {tf} {be} {r['status']:<24s}")

if len(ok_df):
    best = ok_df.sort_values("val_macro_f1", ascending=False).iloc[0]
    print("\n── Best combination (by val_macro_f1) ──")
    print(f"  combo_id      : {int(best['combo_id'])}")
    print(f"  batch_size    : {int(best['batch_size'])}")
    print(f"  learning_rate : {best['learning_rate']:.0e}")
    print(f"  val_accuracy  : {best['val_accuracy']:.4f}")
    print(f"  val_macro_f1  : {best['val_macro_f1']:.4f}")
    print(f"  test_accuracy : {best['test_accuracy']:.4f}")
    print(f"  test_macro_f1 : {best['test_macro_f1']:.4f}")
    print(f"  checkpoint    : {best['checkpoint_dir']}")
else:
    print("\n  All combos failed — no best combination to report.")

print(f"\n✅ Grid search complete. {len(ok_df)}/{len(results)} combos succeeded. "
      f"Full results → {GRID_CSV}")
