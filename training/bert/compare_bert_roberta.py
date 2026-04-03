import os
import json
import math
import copy
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORK_DIR)

LABEL_MAP = {"POSITIVE": 0, "NEUTRAL": 1, "CONSTRUCTIVE": 2, "TOXIC": 3}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
LABEL_NAMES = ["POSITIVE", "NEUTRAL", "CONSTRUCTIVE", "TOXIC"]

with open(os.path.join(WORK_DIR, "label_map.json"), "w", encoding="utf-8") as f:
    json.dump(LABEL_MAP, f, ensure_ascii=False)

LR = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
GRAD_CLIP = 1.0
CLASSIFIER_DROPOUT = 0.15
MAX_EPOCHS = 15
PATIENCE = 4
VAL_TEST_BATCH = 32

try:
    device = torch_directml.device()
    _ = torch.zeros(1, device=device)
    print(f"DirectML device initialized: {device}")
except Exception as e:
    print(f"ERROR: DirectML failed to initialize — {e}")
    print("Stopping. Do NOT fall back to CPU.")
    raise SystemExit(1)

CSV_PATH = os.path.join(WORK_DIR, "10k_fully_labeled.csv")
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

required_cols = {"id", "text_light_clean", "label", "split", "source"}
missing = required_cols - set(df.columns)
if missing:
    raise RuntimeError(f"Missing columns: {missing}")

valid_labels = {"POSITIVE", "NEUTRAL", "CONSTRUCTIVE", "TOXIC"}
invalid_mask = ~df["label"].isin(valid_labels)
n_invalid = int(invalid_mask.sum())
if n_invalid > 0:
    print(f"Dropping {n_invalid} rows with invalid labels (annotation artifacts)")
    df = df[~invalid_mask].reset_index(drop=True)

assert set(df["label"].unique()) == valid_labels, f"Unexpected labels: {df['label'].unique()}"
assert set(df["split"].unique()) == {"train", "val", "test"}, f"Unexpected splits: {df['split'].unique()}"

df["label_id"] = df["label"].map(LABEL_MAP)
df["input_text"] = df.apply(
    lambda r: f"[{r['source']}] {str(r['text_light_clean'] or '')}", axis=1
)

split_counts = df["split"].value_counts().to_dict()
print(f"Dataset after filter: {len(df)} rows | splits: {split_counts}")
print("Per-split label counts:")
print(df.groupby(['split','label']).size().unstack(fill_value=0))

class TextDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }

def is_oom_error(err: BaseException) -> bool:
    s = str(err).lower()
    keywords = ["out of memory", "oom", "allocation", "alloc failed",
                "cuda error: out", "directml", "hipout"]
    return any(k in s for k in keywords)

def run_one_model(shortname: str, display_name: str, hf_name: str, save_dir: str,
                  pred_csv: str, df: pd.DataFrame):
    print("\n" + "═" * 72)
    print(f" RUN — {display_name}  ({hf_name})")
    print("═" * 72)

    print(f"Rows: train={int((df['split']=='train').sum())} "
          f"val={int((df['split']=='val').sum())} "
          f"test={int((df['split']=='test').sum())}")
    print(f"✅ [Step 1 — load+filter — {display_name} — done]")

    print(f"✅ [Step 2 — source-prepended input text — {display_name} — done]")

    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_name)
    except Exception as e:
        print(f"ERROR: Failed to load tokenizer from '{hf_name}': {e}")
        raise SystemExit(1)

    train_texts_all = df.loc[df["split"] == "train", "input_text"].tolist()
    lens = [len(tokenizer.encode(t, add_special_tokens=True, truncation=False))
            for t in train_texts_all]
    p95 = int(np.percentile(lens, 95))
    if p95 <= 256:
        max_length = 256
        train_batch = 16
    else:
        max_length = 512
        train_batch = 8
    print(f"Train p95 token length: {p95} -> max_length={max_length}, train batch_size={train_batch}")
    print(f"✅ [Step 3 — tokenizer + max_length policy — {display_name} — done]")

    def fresh_model():
        config = AutoConfig.from_pretrained(hf_name, num_labels=4)
        config.classifier_dropout = CLASSIFIER_DROPOUT
        try:
            m = AutoModelForSequenceClassification.from_pretrained(hf_name, config=config)
        except Exception as e:
            print(f"ERROR: Failed to load model from '{hf_name}': {e}")
            raise SystemExit(1)
        m.resize_token_embeddings(len(tokenizer))
        return m

    def tokenize_split(split_name):
        texts = df.loc[df["split"] == split_name, "input_text"].tolist()
        enc = tokenizer(texts, max_length=max_length, truncation=True,
                        padding="max_length", return_tensors="pt")
        labels = torch.tensor(df.loc[df["split"] == split_name, "label_id"].values,
                              dtype=torch.long)
        return TextDataset(enc["input_ids"], enc["attention_mask"], labels)

    train_ds = tokenize_split("train")
    val_ds = tokenize_split("val")
    test_ds = tokenize_split("test")
    print(f"Tokenized — train:{len(train_ds)} val:{len(val_ds)} test:{len(test_ds)}")
    print(f"✅ [Step 4 — tokenize (max_length={max_length}) — {display_name} — done]")

    train_label_ids = df.loc[df["split"] == "train", "label_id"].values
    class_counts = np.bincount(train_label_ids, minlength=4).astype(float)

    def build_loaders(batch_size):
        sample_weights = torch.tensor([1.0 / class_counts[l] for l in train_label_ids],
                                      dtype=torch.float64)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds),
                                        replacement=True)
        train_ld = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
        val_ld = DataLoader(val_ds, batch_size=VAL_TEST_BATCH, shuffle=False)
        test_ld = DataLoader(test_ds, batch_size=VAL_TEST_BATCH, shuffle=False)
        return train_ld, val_ld, test_ld

    print(f"Class counts (train): { {LABEL_NAMES[i]: int(class_counts[i]) for i in range(4)} }")
    print(f"✅ [Step 5 — datasets + WeightedRandomSampler — {display_name} — done]")

    class_weights = (1.0 / class_counts) * class_counts.sum() / len(class_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Class weights: { {LABEL_NAMES[i]: round(class_weights[i], 4) for i in range(4)} }")
    print(f"✅ [Step 6 — class-weighted loss — {display_name} — done]")

    def do_full_training(batch_size):
        train_ld, val_ld, test_ld = build_loaders(batch_size)
        model = fresh_model().to(device)

        criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        total_steps = len(train_ld) * MAX_EPOCHS
        warmup_steps = int(total_steps * WARMUP_RATIO)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        print(f"\nTraining config: lr={LR} wd={WEIGHT_DECAY} warmup_steps={warmup_steps} "
              f"total_steps={total_steps} batch_size={batch_size} max_epochs={MAX_EPOCHS} patience={PATIENCE}")

        best_val_f1 = -1.0
        best_epoch = -1
        patience_counter = 0
        os.makedirs(save_dir, exist_ok=True)

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            total_loss = 0.0
            train_preds, train_true = [], []

            for batch_idx, batch in enumerate(train_ld):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                train_preds.extend(preds)
                train_true.extend(labels.cpu().numpy())

                if (batch_idx + 1) % 100 == 0:
                    print(f"  Epoch {epoch} batch {batch_idx+1}/{len(train_ld)} loss={loss.item():.4f}")

            train_loss = total_loss / len(train_ld)
            train_f1 = f1_score(train_true, train_preds, average="macro")

            model.eval()
            val_preds, val_true = [], []
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_ld:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels)
                    val_loss += loss.item()
                    preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                    val_preds.extend(preds)
                    val_true.extend(labels.cpu().numpy())
            avg_val_loss = val_loss / len(val_ld)
            val_f1 = f1_score(val_true, val_preds, average="macro")

            new_best = val_f1 > best_val_f1
            tag = " [NEW BEST]" if new_best else ""
            print(f"Epoch {epoch} | train_loss={train_loss:.4f} | train_macro_f1={train_f1:.4f}"
                  f" | val_loss={avg_val_loss:.4f} | val_macro_f1={val_f1:.4f}{tag}")

            if new_best:
                best_val_f1 = val_f1
                best_epoch = epoch
                patience_counter = 0
                cpu_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(cpu_state_dict, os.path.join(save_dir, "pytorch_model.bin"))
                model.config.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                print(f"  >> saved checkpoint to {save_dir}")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch} (patience={PATIENCE} exhausted)")
                    break

        return best_val_f1, best_epoch, test_ld

    retry_used = False
    current_batch = None
    while True:
        try:
            if current_batch is None:
                current_batch = train_batch
            print(f"\n[Training attempt] batch_size={current_batch}")
            best_val_f1, best_epoch, test_ld = do_full_training(current_batch)
            break
        except RuntimeError as e:
            if is_oom_error(e) and not retry_used:
                retry_used = True
                new_batch = max(1, current_batch // 2)
                print(f"⚠  OOM detected: {e}")
                print(f"⚠  Halving batch_size: {current_batch} -> {new_batch} and retrying.")
                current_batch = new_batch
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                continue
            print(f"ERROR during training: {e}")
            if is_oom_error(e):
                print("OOM persists after one halving — stopping.")
            raise

    print(f"✅ [Step 7 — training — {display_name} — best val macro F1={best_val_f1:.4f} @ epoch {best_epoch}]")

    print(f"✅ [Step 8 — checkpoint saved — {display_name} — {save_dir}/]")

    config = AutoConfig.from_pretrained(hf_name, num_labels=4)
    config.classifier_dropout = CLASSIFIER_DROPOUT
    best_model = AutoModelForSequenceClassification.from_pretrained(hf_name, config=config)
    best_model.resize_token_embeddings(len(tokenizer))
    state_dict = torch.load(os.path.join(save_dir, "pytorch_model.bin"), map_location="cpu")
    best_model.load_state_dict(state_dict)
    best_model.to(device)
    best_model.eval()

    test_preds, test_true = [], []
    with torch.no_grad():
        for batch in test_ld:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            outputs = best_model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            test_preds.extend(preds)
            test_true.extend(labels.numpy())
    test_preds = np.array(test_preds)
    test_true = np.array(test_true)
    print(f"✅ [Step 9 — test inference — {display_name} — done]")

    macro_f1 = f1_score(test_true, test_preds, average="macro")
    weighted_f1 = f1_score(test_true, test_preds, average="weighted")
    acc = accuracy_score(test_true, test_preds)
    per_class_f1 = f1_score(test_true, test_preds, average=None, labels=[0, 1, 2, 3])

    print("\n" + "─" * 72)
    print(f"TEST RESULTS — {display_name}")
    print("─" * 72)
    print(f"Accuracy:     {acc:.4f}")
    print(f"Macro F1:     {macro_f1:.4f}")
    print(f"Weighted F1:  {weighted_f1:.4f}")
    print("\n--- sklearn classification_report ---")
    print(classification_report(test_true, test_preds, target_names=LABEL_NAMES, digits=4))
    print("--- Confusion Matrix (rows=true, cols=pred) ---")
    cm = confusion_matrix(test_true, test_preds, labels=[0, 1, 2, 3])
    header = " " * 16 + "  ".join(f"{n:>13s}" for n in LABEL_NAMES)
    print(f"{'Predicted →':>16s}")
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>13d}" for v in row)
        print(f"{LABEL_NAMES[i]:>16s}  {row_str}")

    test_df = df.loc[df["split"] == "test"].copy().reset_index(drop=True)
    test_df["predicted_label"] = [ID2LABEL[int(p)] for p in test_preds]
    test_df["true_label"] = test_df["label"]
    out_cols = ["id", "text_light_clean", "true_label", "predicted_label"]
    test_df[out_cols].to_csv(os.path.join(WORK_DIR, pred_csv), index=False, encoding="utf-8-sig")
    print(f"\nPredictions written to ./{pred_csv}")
    print(f"✅ [Step 10 — evaluation + predictions — {display_name} — done]")

    best_model.cpu()
    del best_model
    import gc; gc.collect()

    return {
        "shortname": shortname,
        "display_name": display_name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": {LABEL_NAMES[i]: float(per_class_f1[i]) for i in range(4)},
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
    }

RUNS = [
    ("bert", "MN-BERT",     "tugstugi/bert-base-mongolian-cased",
     os.path.join(WORK_DIR, "best_mnbert_model"),  "test_predictions_bert.csv"),
    ("roberta", "MN-RoBERTa", "bayartsogt/mongolian-roberta-base",
     os.path.join(WORK_DIR, "best_roberta_model"), "test_predictions_roberta.csv"),
]

results = []
for shortname, display_name, hf_name, save_dir, pred_csv in RUNS:
    r = run_one_model(shortname, display_name, hf_name, save_dir, pred_csv, df)
    results.append(r)

def fmt(v, int_ok=False):
    if int_ok:
        return f"{int(v)}"
    return f"{v:.4f}"

bert_r = next(r for r in results if r["shortname"] == "bert")
rob_r = next(r for r in results if r["shortname"] == "roberta")

print("\n" + "═" * 72)
print(" FINAL COMPARISON — MN-BERT vs MN-RoBERTa")
print("═" * 72)
rows = [
    ("Accuracy",        fmt(bert_r["accuracy"]),            fmt(rob_r["accuracy"])),
    ("Macro F1",        fmt(bert_r["macro_f1"]),            fmt(rob_r["macro_f1"])),
    ("Weighted F1",     fmt(bert_r["weighted_f1"]),         fmt(rob_r["weighted_f1"])),
    ("POSITIVE F1",     fmt(bert_r["per_class_f1"]["POSITIVE"]),     fmt(rob_r["per_class_f1"]["POSITIVE"])),
    ("NEUTRAL F1",      fmt(bert_r["per_class_f1"]["NEUTRAL"]),      fmt(rob_r["per_class_f1"]["NEUTRAL"])),
    ("CONSTRUCTIVE F1", fmt(bert_r["per_class_f1"]["CONSTRUCTIVE"]), fmt(rob_r["per_class_f1"]["CONSTRUCTIVE"])),
    ("TOXIC F1",        fmt(bert_r["per_class_f1"]["TOXIC"]),        fmt(rob_r["per_class_f1"]["TOXIC"])),
    ("Best epoch",      fmt(bert_r["best_epoch"], int_ok=True),       fmt(rob_r["best_epoch"], int_ok=True)),
]
print(f"| {'Metric':<16s} | {'MN-BERT':>8s} | {'MN-RoBERTa':>10s} |")
print(f"|{'-'*18}|{'-'*10}|{'-'*12}|")
for name, a, b in rows:
    print(f"| {name:<16s} | {a:>8s} | {b:>10s} |")

print("\nAll runs complete.")
