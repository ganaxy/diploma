import os
import sys
import numpy as np
import pandas as pd
import torch
import torch_directml
from torch.utils.data import Dataset, DataLoader
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

HF_NAME       = "tugstugi/bert-base-mongolian-cased"
STAGE1_CKPT   = os.path.join(WORK_DIR, "best_stage1_model")
STAGE2_CKPT   = os.path.join(WORK_DIR, "best_stage2_model")
DATA_CSV      = os.path.join(WORK_DIR, "relabeled_v9_normalized.csv")
OUT_CSV       = os.path.join(WORK_DIR, "test_predictions_2stage.csv")

STAGE1_NAMES = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
STAGE1_MAP   = {"POSITIVE": 0, "NEUTRAL": 1, "NEGATIVE": 2}
STAGE1_ID2L  = {v: k for k, v in STAGE1_MAP.items()}

STAGE2_NAMES = ["CONSTRUCTIVE", "TOXIC"]
STAGE2_MAP   = {"CONSTRUCTIVE": 0, "TOXIC": 1}
STAGE2_ID2L  = {v: k for k, v in STAGE2_MAP.items()}

FINAL_NAMES = ["POSITIVE", "NEUTRAL", "CONSTRUCTIVE", "TOXIC"]
FINAL_MAP   = {"POSITIVE": 0, "NEUTRAL": 1, "CONSTRUCTIVE": 2, "TOXIC": 3}
FINAL_ID2L  = {v: k for k, v in FINAL_MAP.items()}

MAX_LENGTH = 256
BATCH      = 32
CLASSIFIER_DROPOUT = 0.15

A_METRICS = {
    "accuracy":     0.8064,
    "macro_f1":     0.7698,
    "weighted_f1":  0.8052,
    "POSITIVE":     0.6593,
    "NEUTRAL":      0.7697,
    "CONSTRUCTIVE": 0.8539,
    "TOXIC":        0.7964,
}

def fail(msg: str):
    print(f"ERROR: {msg}")
    raise SystemExit(1)

try:
    device = torch_directml.device()
    _ = torch.zeros(1, device=device)
    print(f"DirectML device initialized: {device}")
except Exception as e:
    fail(f"DirectML failed to initialize — {e}")
print("✅ [Step 4.1 — device ready]")

if not os.path.isdir(STAGE1_CKPT):
    fail(f"{STAGE1_CKPT} not found — run STEP 2 first")
if not os.path.isdir(STAGE2_CKPT):
    fail(f"{STAGE2_CKPT} not found — run STEP 3 first")
print("✅ [Step 4.2 — both stage checkpoints present]")

if not os.path.exists(DATA_CSV):
    fail(f"{DATA_CSV} not found")
df = pd.read_csv(DATA_CSV, encoding="utf-8-sig")
unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
if unnamed:
    df = df.drop(columns=unnamed)
test_df = df[df["split"] == "test"].copy().reset_index(drop=True)
test_df["input_text"] = test_df.apply(
    lambda r: f"[{r['source']}] {str(r['text_normalized'] or '')}", axis=1
)
test_df["true_label"] = test_df["label"].astype(str).str.upper()
test_df["true_id"]    = test_df["true_label"].map(FINAL_MAP)
print(f"Loaded test split: {len(test_df)} rows")
print("✅ [Step 4.3 — test data loaded]")

def load_ckpt(ckpt_dir, num_labels):
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    config = AutoConfig.from_pretrained(HF_NAME, num_labels=num_labels)
    config.classifier_dropout = CLASSIFIER_DROPOUT
    m = AutoModelForSequenceClassification.from_pretrained(HF_NAME, config=config)
    m.resize_token_embeddings(len(tokenizer))
    state = torch.load(os.path.join(ckpt_dir, "pytorch_model.bin"),
                       map_location="cpu")
    m.load_state_dict(state)
    m.to(device); m.eval()
    return tokenizer, m

tok1, model1 = load_ckpt(STAGE1_CKPT, num_labels=3)
print("✅ [Step 4.4 — Stage 1 checkpoint loaded]")
tok2, model2 = load_ckpt(STAGE2_CKPT, num_labels=2)
print("✅ [Step 4.5 — Stage 2 checkpoint loaded]")

class InferDS(Dataset):
    def __init__(self, ids, mask):
        self.ids, self.mask = ids, mask
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "attention_mask": self.mask[i]}

@torch.no_grad()
def batch_predict(model, tokenizer, texts):
    enc = tokenizer(texts, padding="max_length", truncation=True,
                    max_length=MAX_LENGTH, return_tensors="pt")
    ds = InferDS(enc["input_ids"], enc["attention_mask"])
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)
    preds = []
    for batch in loader:
        iid = batch["input_ids"].to(device)
        att = batch["attention_mask"].to(device)
        logits = model(input_ids=iid, attention_mask=att).logits
        preds.append(torch.argmax(logits, dim=-1).detach().cpu().numpy())
    return np.concatenate(preds).astype(int)

print(f"\nRunning Stage 1 inference on {len(test_df)} test rows ...")
stage1_pred_id = batch_predict(model1, tok1, test_df["input_text"].tolist())
test_df["stage1_pred"] = [STAGE1_ID2L[i] for i in stage1_pred_id]
print("✅ [Step 4.6 — Stage 1 predictions done]")

neg_mask = test_df["stage1_pred"] == "NEGATIVE"
n_to_stage2 = int(neg_mask.sum())
print(f"\nRouting {n_to_stage2} rows (Stage 1 said NEGATIVE) to Stage 2 ...")
stage2_pred_label = np.full(len(test_df), "", dtype=object)
if n_to_stage2:
    sub_texts = test_df.loc[neg_mask, "input_text"].tolist()
    s2_ids = batch_predict(model2, tok2, sub_texts)
    s2_labels = np.array([STAGE2_ID2L[i] for i in s2_ids], dtype=object)
    stage2_pred_label[neg_mask.values] = s2_labels
test_df["stage2_pred"] = stage2_pred_label
print("✅ [Step 4.7 — Stage 2 predictions done]")

def final_from_row(s1, s2):
    if s1 == "POSITIVE":
        return "POSITIVE"
    if s1 == "NEUTRAL":
        return "NEUTRAL"
    return s2 if s2 in ("CONSTRUCTIVE", "TOXIC") else "CONSTRUCTIVE"

test_df["final_pred"] = [
    final_from_row(s1, s2) for s1, s2 in
    zip(test_df["stage1_pred"], test_df["stage2_pred"])
]
test_df["final_pred_id"] = test_df["final_pred"].map(FINAL_MAP)
print("✅ [Step 4.8 — final 4-class predictions composed]")

y_true = test_df["true_id"].values
y_pred = test_df["final_pred_id"].values

acc         = accuracy_score(y_true, y_pred)
macro_f1    = f1_score(y_true, y_pred, average="macro")
weighted_f1 = f1_score(y_true, y_pred, average="weighted")
per_class_f1 = f1_score(y_true, y_pred, average=None,
                        labels=list(range(4)))

print("\n" + "═"*64)
print(" STEP 4 — 2-stage pipeline results on full 4-class test set")
print("═"*64)
print(f"Accuracy        : {acc:.4f}")
print(f"Macro F1        : {macro_f1:.4f}")
print(f"Weighted F1     : {weighted_f1:.4f}")

print("\nclassification_report:")
print(classification_report(y_true, y_pred, labels=list(range(4)),
                            target_names=FINAL_NAMES, digits=4))

print("Confusion matrix (rows=true, cols=pred):")
cm = confusion_matrix(y_true, y_pred, labels=list(range(4)))
print("               " + " ".join(f"{n:>13s}" for n in FINAL_NAMES))
for i, row in enumerate(cm):
    cells = " ".join(f"{v:>13d}" for v in row)
    print(f"  {FINAL_NAMES[i]:<12s} {cells}")
print("✅ [Step 4.9 — metrics computed]")

true_is_negsub = test_df["true_label"].isin(["CONSTRUCTIVE", "TOXIC"])
s1_misrouted = true_is_negsub & (test_df["stage1_pred"] != "NEGATIVE")
n_con_to_pos = int(((test_df["true_label"] == "CONSTRUCTIVE") &
                    (test_df["stage1_pred"] == "POSITIVE")).sum())
n_con_to_neu = int(((test_df["true_label"] == "CONSTRUCTIVE") &
                    (test_df["stage1_pred"] == "NEUTRAL")).sum())
n_tox_to_pos = int(((test_df["true_label"] == "TOXIC") &
                    (test_df["stage1_pred"] == "POSITIVE")).sum())
n_tox_to_neu = int(((test_df["true_label"] == "TOXIC") &
                    (test_df["stage1_pred"] == "NEUTRAL")).sum())
n_total_negsub = int(true_is_negsub.sum())
n_s1_mis     = int(s1_misrouted.sum())

true_not_negsub = ~true_is_negsub
s1_falsely_neg = true_not_negsub & (test_df["stage1_pred"] == "NEGATIVE")
n_pos_to_neg = int(((test_df["true_label"] == "POSITIVE") &
                    (test_df["stage1_pred"] == "NEGATIVE")).sum())
n_neu_to_neg = int(((test_df["true_label"] == "NEUTRAL") &
                    (test_df["stage1_pred"] == "NEGATIVE")).sum())
n_s1_falseneg = int(s1_falsely_neg.sum())

print("\n── Stage 1 error propagation analysis ───────────────────")
print(f"  True CON/TOX rows  : {n_total_negsub}")
print(f"  Misrouted by S1    : {n_s1_mis}  "
      f"({100.0*n_s1_mis/n_total_negsub:.2f}% of CON/TOX)")
print(f"    CON → POSITIVE   : {n_con_to_pos}")
print(f"    CON → NEUTRAL    : {n_con_to_neu}")
print(f"    TOX → POSITIVE   : {n_tox_to_pos}")
print(f"    TOX → NEUTRAL    : {n_tox_to_neu}")
print(f"  These {n_s1_mis} rows never reach Stage 2 — "
      f"guaranteed wrong in final output.")
print()
print(f"  True POS/NEU rows falsely routed to Stage 2 : {n_s1_falseneg}")
print(f"    POS → NEGATIVE (will be tagged CON/TOX) : {n_pos_to_neg}")
print(f"    NEU → NEGATIVE (will be tagged CON/TOX) : {n_neu_to_neg}")
print(f"  These {n_s1_falseneg} rows reach Stage 2 but are guaranteed wrong.")
print(f"✅ [Step 4.10 — error propagation analysed]")

def pct(prev, new):
    return f"{(new - prev) * 100:+.2f}%"

B_METRICS = {
    "accuracy":     acc,
    "macro_f1":     macro_f1,
    "weighted_f1":  weighted_f1,
    "POSITIVE":     per_class_f1[0],
    "NEUTRAL":      per_class_f1[1],
    "CONSTRUCTIVE": per_class_f1[2],
    "TOXIC":        per_class_f1[3],
}

print("\n── Comparison: Approach A (flat v7) | Approach B (2-stage) ──")
print(f"| {'Metric':<15s} | {'A (flat v7)':>12s} | {'B (2-stage)':>12s} | {'Δ A→B':>8s} |")
print(f"|{'-'*17}|{'-'*14}|{'-'*14}|{'-'*10}|")
for key, label in [
    ("accuracy",     "Accuracy"),
    ("macro_f1",     "Macro F1"),
    ("weighted_f1",  "Weighted F1"),
    ("POSITIVE",     "POSITIVE F1"),
    ("NEUTRAL",      "NEUTRAL F1"),
    ("CONSTRUCTIVE", "CONSTRUCTIVE F1"),
    ("TOXIC",        "TOXIC F1"),
]:
    a = A_METRICS[key]; b = B_METRICS[key]
    print(f"| {label:<15s} | {a:>12.4f} | {b:>12.4f} | {pct(a, b):>8s} |")

macro_delta_pct = (macro_f1 - A_METRICS["macro_f1"]) * 100

print("\n── Verdict ─────────────────────────────────────────")
if macro_delta_pct > 0.5:
    print(f"  Approach B macro F1 is {macro_delta_pct:+.2f} pct-pts above A.")
    print("  2-stage improved — hierarchical architecture is beneficial for this task.")
elif macro_delta_pct < -0.5:
    print(f"  Approach B macro F1 is {macro_delta_pct:+.2f} pct-pts below A.")
    print("  2-stage regressed — Stage 1 error propagation is the likely cause.")
else:
    print(f"  Approach B macro F1 moved {macro_delta_pct:+.2f} pct-pts vs A "
          f"(within ±0.5 pct-pt band).")
    print("  2-stage is on par — architectural complexity adds no significant gain.")
print("✅ [Step 4.11 — A vs B comparison done]")

out = pd.DataFrame({
    "id":               test_df["id"].values,
    "text_light_clean": test_df["text_light_clean"].values,
    "true_label":       test_df["true_label"].values,
    "stage1_pred":      test_df["stage1_pred"].values,
    "stage2_pred":      test_df["stage2_pred"].values,
    "final_pred":       test_df["final_pred"].values,
})
out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n✅ [Step 4.12 — saved {len(out)} test predictions to {OUT_CSV}]")
print("\n✅ STEP 4 (2-stage pipeline) complete.")
