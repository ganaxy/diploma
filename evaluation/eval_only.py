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

HF_NAME = "tugstugi/bert-base-mongolian-cased"
NORMALIZED_CSV = os.path.join(WORK_DIR, "10k_fully_labeled_normalized.csv")
SAVE_DIR = os.path.join(WORK_DIR, "best_mnbert_v2_model")
PRED_CSV = os.path.join(WORK_DIR, "test_predictions_v2.csv")

LABEL_MAP = {"POSITIVE": 0, "NEUTRAL": 1, "CONSTRUCTIVE": 2, "TOXIC": 3}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
LABEL_NAMES = ["POSITIVE", "NEUTRAL", "CONSTRUCTIVE", "TOXIC"]
MAX_LENGTH = 256
VAL_TEST_BATCH = 32
CLASSIFIER_DROPOUT = 0.15

PREV_METRICS = {
    "accuracy":     0.7656,
    "macro_f1":     0.7235,
    "weighted_f1":  0.7644,
    "POSITIVE":     0.6358,
    "NEUTRAL":      0.7295,
    "CONSTRUCTIVE": 0.8242,
    "TOXIC":        0.7045,
}

device = torch_directml.device()
_ = torch.zeros(1, device=device)
print(f"DirectML device: {device}")

df = pd.read_csv(NORMALIZED_CSV, encoding="utf-8-sig")
df["input_text"] = df.apply(
    lambda r: f"[{r['source']}] {str(r['text_normalized'] or '')}", axis=1
)
df["label_id"] = df["label"].map(LABEL_MAP)
test_df = df[df["split"] == "test"].reset_index(drop=True)
print(f"Test rows: {len(test_df)}")

tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
enc = tokenizer(
    test_df["input_text"].tolist(),
    padding="max_length", truncation=True,
    max_length=MAX_LENGTH, return_tensors="pt",
)

class DS(Dataset):
    def __init__(self, ids, mask, y):
        self.ids, self.mask, self.y = ids, mask, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"input_ids": self.ids[i],
                "attention_mask": self.mask[i],
                "labels": self.y[i]}

test_ds = DS(
    enc["input_ids"], enc["attention_mask"],
    torch.tensor(test_df["label_id"].tolist(), dtype=torch.long),
)
test_loader = DataLoader(test_ds, batch_size=VAL_TEST_BATCH, shuffle=False)

print("Building fresh model from HF base → resize → load state_dict ...")
config = AutoConfig.from_pretrained(HF_NAME, num_labels=4)
config.classifier_dropout = CLASSIFIER_DROPOUT
model = AutoModelForSequenceClassification.from_pretrained(HF_NAME, config=config)
model.resize_token_embeddings(len(tokenizer))
state = torch.load(os.path.join(SAVE_DIR, "pytorch_model.bin"),
                   map_location="cpu")
model.load_state_dict(state)
model.to(device)
model.eval()
print(f"Model loaded: vocab={model.get_input_embeddings().num_embeddings}")

@torch.no_grad()
def evaluate(model, loader):
    y_true = []
    y_pred = []
    losses = []
    ce = torch.nn.CrossEntropyLoss()
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attn).logits
        losses.append(ce(logits, labels).item())
        y_true.append(labels.cpu().numpy())
        y_pred.append(torch.argmax(logits, dim=-1).cpu().numpy())
    return (np.concatenate(y_true), np.concatenate(y_pred),
            float(np.mean(losses)))

y_true, y_pred, test_loss = evaluate(model, test_loader)

acc = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro")
weighted_f1 = f1_score(y_true, y_pred, average="weighted")
per_class_f1 = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3])

print("\n" + "═" * 64)
print(" TASK 4 — Test-set results (MN-BERT v2 with text_normalized)")
print("═" * 64)
print(f"test_loss            : {test_loss:.4f}")
print(f"Overall accuracy     : {acc:.4f}")
print(f"Macro F1             : {macro_f1:.4f}")
print(f"Weighted F1          : {weighted_f1:.4f}")

print("\nclassification_report:")
print(classification_report(y_true, y_pred, labels=[0, 1, 2, 3],
                            target_names=LABEL_NAMES, digits=4))

print("Confusion matrix (rows=true, cols=pred):")
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
print("            " + " ".join(f"{n:>13s}" for n in LABEL_NAMES))
for i, row in enumerate(cm):
    cells = " ".join(f"{v:>13d}" for v in row)
    print(f"  {LABEL_NAMES[i]:<10s}{cells}")

def delta(prev, new):
    return f"{(new - prev) * 100:+.2f}%"

new_metrics = {
    "accuracy":     acc,
    "macro_f1":     macro_f1,
    "weighted_f1":  weighted_f1,
    "POSITIVE":     per_class_f1[0],
    "NEUTRAL":      per_class_f1[1],
    "CONSTRUCTIVE": per_class_f1[2],
    "TOXIC":        per_class_f1[3],
}

print("\n── Comparison vs previous best ─────────────────")
print(f"| {'Metric':<15s} | {'Previous':>8s} | {'New':>8s} | {'Delta':>7s} |")
print(f"|{'-'*17}|{'-'*10}|{'-'*10}|{'-'*9}|")
for key, label in [
    ("accuracy",     "Accuracy"),
    ("macro_f1",     "Macro F1"),
    ("weighted_f1",  "Weighted F1"),
    ("POSITIVE",     "POSITIVE F1"),
    ("NEUTRAL",      "NEUTRAL F1"),
    ("CONSTRUCTIVE", "CONSTRUCTIVE F1"),
    ("TOXIC",        "TOXIC F1"),
]:
    prev = PREV_METRICS[key]
    new = new_metrics[key]
    print(f"| {label:<15s} | {prev:>8.4f} | {new:>8.4f} | {delta(prev,new):>7s} |")

out = pd.DataFrame({
    "id": test_df["id"].values,
    "text_light_clean": test_df["text_light_clean"].values,
    "true_label": [ID2LABEL[int(v)] for v in y_true],
    "predicted_label": [ID2LABEL[int(v)] for v in y_pred],
})
out.to_csv(PRED_CSV, index=False, encoding="utf-8-sig")
print(f"\n✅ Saved test predictions to {PRED_CSV}")
print("✅ TASK 4 complete.")
