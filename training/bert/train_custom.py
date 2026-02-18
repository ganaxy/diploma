import os
import json
import math
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score

from src.config import (
    LABEL_NAMES, RANDOM_SEED, MODELS_DIR, TEXT_COLUMN,
    CUSTOM_MAX_SEQ_LEN, CUSTOM_MAX_WORD_LEN,
    CUSTOM_WORD_VOCAB_SIZE, CUSTOM_CHAR_VOCAB_SIZE,
    CUSTOM_CHAR_EMBED_DIM, CUSTOM_CHAR_CNN_FILTERS, CUSTOM_CHAR_CNN_KERNELS,
    CUSTOM_WORD_EMBED_DIM, CUSTOM_FUSION_DIM,
    CUSTOM_LSTM_HIDDEN, CUSTOM_LSTM_LAYERS, CUSTOM_LSTM_DROPOUT,
    CUSTOM_ATTN_HEADS, CUSTOM_ATTN_DROPOUT,
    CUSTOM_CLASSIFIER_HIDDEN, CUSTOM_CLASSIFIER_DROPOUT,
    CUSTOM_EMBED_SPATIAL_DROPOUT, CUSTOM_TOKEN_DROPOUT_RATE,
    CUSTOM_BATCH_SIZE, CUSTOM_EPOCHS, CUSTOM_LR,
    CUSTOM_WEIGHT_DECAY, CUSTOM_LABEL_SMOOTHING,
    CUSTOM_GRAD_CLIP, CUSTOM_PATIENCE,
)
from src.data_prep import load_and_prepare
from src.evaluate import full_evaluation

LABEL2ID: Dict[str, int] = {name: i for i, name in enumerate(LABEL_NAMES)}
ID2LABEL: Dict[int, str] = {i: name for i, name in enumerate(LABEL_NAMES)}

def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("[custom] Using CUDA GPU")
        return torch.device("cuda")

    try:
        import torch_directml
        dml_device = torch_directml.device()
        print("[custom] Using DirectML (AMD GPU)")
        return dml_device
    except ImportError:
        pass

    print("[custom] WARNING: No GPU detected — training will be slow.")
    print("[custom] Install torch-directml for AMD GPU support: pip install torch-directml")
    return torch.device("cpu")

def build_word_vocab(texts: List[str], max_size: int) -> Dict[str, int]:
    counter: Counter = Counter()
    for text in texts:
        for word in text.split():
            counter[word] += 1
    vocab: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(max_size - 2):
        vocab[word] = len(vocab)
    return vocab

def build_char_vocab(texts: List[str], max_size: int) -> Dict[str, int]:
    chars: Counter = Counter()
    for text in texts:
        for ch in text:
            chars[ch] += 1
    vocab: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for ch, _ in chars.most_common(max_size - 2):
        vocab[ch] = len(vocab)
    return vocab

class MongolianTextDataset(Dataset):

    def __init__(
        self,
        texts: np.ndarray,
        labels: np.ndarray,
        word_vocab: Dict[str, int],
        char_vocab: Dict[str, int],
        max_seq_len: int,
        max_word_len: int,
    ) -> None:
        self.texts = texts
        self.labels = np.array([LABEL2ID[l] for l in labels])
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.max_seq_len = max_seq_len
        self.max_word_len = max_word_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        text = str(self.texts[idx])
        words = text.split()[: self.max_seq_len]

        word_ids: List[int] = []
        char_ids: List[List[int]] = []

        for word in words:
            word_ids.append(self.word_vocab.get(word, 1))
            chars = [self.char_vocab.get(c, 1) for c in word[: self.max_word_len]]
            pad_len = self.max_word_len - len(chars)
            chars = chars + [0] * pad_len
            char_ids.append(chars)

        actual_len = len(word_ids)
        pad_len = self.max_seq_len - actual_len

        word_ids = word_ids + [0] * pad_len
        char_ids = char_ids + [[0] * self.max_word_len] * pad_len
        mask = [1.0] * actual_len + [0.0] * pad_len

        return (
            torch.tensor(word_ids, dtype=torch.long),
            torch.tensor(char_ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )

class SpatialDropout1d(nn.Module):

    def __init__(self, p: float = 0.2) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x
        keep = 1.0 - self.p
        mask = torch.bernoulli(torch.full((x.size(0), 1, x.size(2)), keep))
        mask = mask.to(x.device) / keep
        return x * mask

class CharCNNEncoder(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_filters: int,
        kernel_sizes: List[int],
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.output_dim = num_filters * len(kernel_sizes)

        nn.init.xavier_uniform_(self.embedding.weight)
        with torch.no_grad():
            self.embedding.weight[0].fill_(0)
        for conv in self.convs:
            nn.init.kaiming_uniform_(conv.weight, nonlinearity="relu")

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, max_word_len = char_ids.size()

        x = char_ids.view(batch_size * seq_len, max_word_len)

        features: List[torch.Tensor] = []
        for conv in self.convs:
            features.append(c)

        out = out.view(batch_size, seq_len, self.output_dim)
        return out

class MultiHeadAttentionPooling(nn.Module):

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_model = d_model

        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.queries = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.02)
        self.attn_dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.permute(0, 2, 1, 3).contiguous().view(
            batch_size * self.num_heads, seq_len, self.head_dim
        )
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.permute(0, 2, 1, 3).contiguous().view(
            batch_size * self.num_heads, seq_len, self.head_dim
        )

        Q = Q.contiguous().view(batch_size * self.num_heads, 1, self.head_dim)

        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.head_dim)

        if mask is not None:
            mask_exp = mask_exp.contiguous().view(batch_size * self.num_heads, 1, seq_len)
            additive_mask = (1.0 - mask_exp) * -1e9
            scores = scores + additive_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        pooled = torch.bmm(attn_weights, V)

        pooled = pooled.view(batch_size, self.num_heads, self.head_dim)
        pooled = pooled.contiguous().view(batch_size, self.d_model)

        output = self.W_o(pooled)
        return output

class ManualLSTMCell(nn.Module):

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.W_ih = nn.Linear(input_size, 4 * hidden_size)
        self.W_hh = nn.Linear(hidden_size, 4 * hidden_size)

        nn.init.xavier_uniform_(self.W_ih.weight)
        nn.init.xavier_uniform_(self.W_hh.weight)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gates = self.W_ih(x) + self.W_hh(h)
        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

class ManualBiLSTM(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.forward_cells = nn.ModuleList()
        self.backward_cells = nn.ModuleList()

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * 2
            self.forward_cells.append(ManualLSTMCell(layer_input_size, hidden_size))
            self.backward_cells.append(ManualLSTMCell(layer_input_size, hidden_size))

        self.dropout = nn.Dropout(dropout) if dropout > 0 and num_layers > 1 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        batch_size, seq_len, _ = x.size()
        current_input = x

        for layer in range(self.num_layers):
            h_f = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c_f = torch.zeros(batch_size, self.hidden_size, device=x.device)
            forward_outputs: List[torch.Tensor] = []
            for t in range(seq_len):
                h_f, c_f = self.forward_cells[layer](current_input[:, t, :], h_f, c_f)
                forward_outputs.append(h_f)

            h_b = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c_b = torch.zeros(batch_size, self.hidden_size, device=x.device)
            backward_outputs: List[torch.Tensor] = []
            for t in range(seq_len - 1, -1, -1):
                h_b, c_b = self.backward_cells[layer](current_input[:, t, :], h_b, c_b)
                backward_outputs.append(h_b)
            backward_outputs.reverse()

            forward_out = torch.stack(forward_outputs, dim=1)
            backward_out = torch.stack(backward_outputs, dim=1)
            current_input = torch.cat([forward_out, backward_out], dim=2)

            if self.dropout is not None and layer < self.num_layers - 1:
                current_input = self.dropout(current_input)

        return current_input, None

class MongolianClassifier(nn.Module):

    def __init__(
        self,
        char_vocab_size: int,
        word_vocab_size: int,
        num_classes: int,
        char_embed_dim: int,
        char_cnn_filters: int,
        char_cnn_kernels: List[int],
        word_embed_dim: int,
        fusion_dim: int,
        lstm_hidden: int,
        lstm_layers: int,
        lstm_dropout: float,
        attn_heads: int,
        attn_dropout: float,
        classifier_hidden: int,
        classifier_dropout: float,
        spatial_dropout: float,
        token_dropout_rate: float,
    ) -> None:
        super().__init__()
        self.token_dropout_rate = token_dropout_rate

        self.char_cnn = CharCNNEncoder(
            vocab_size=char_vocab_size,
            embed_dim=char_embed_dim,
            num_filters=char_cnn_filters,
            kernel_sizes=char_cnn_kernels,
        )

        self.word_embed = nn.Embedding(word_vocab_size, word_embed_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.word_embed.weight)
        with torch.no_grad():
            self.word_embed.weight[0].fill_(0)

        self.fusion = nn.Linear(char_feat_dim + word_embed_dim, fusion_dim)
        nn.init.xavier_uniform_(self.fusion.weight)

        self.spatial_drop = SpatialDropout1d(spatial_dropout)

        self.lstm = ManualBiLSTM(
            input_size=fusion_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )

        self.attention_pool = MultiHeadAttentionPooling(
            d_model=lstm_out_dim,
            num_heads=attn_heads,
            dropout=attn_dropout,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Dropout(classifier_dropout),
            nn.Linear(lstm_out_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden, num_classes),
        )
        nn.init.xavier_uniform_(self.classifier[2].weight)
        nn.init.xavier_uniform_(self.classifier[5].weight)

    def forward(
        self,
        word_ids: torch.Tensor,
        char_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = word_ids.size()

        fused = self.spatial_drop(fused)

        if self.training and self.token_dropout_rate > 0:
            keep_prob = 1.0 - self.token_dropout_rate
            token_mask = torch.bernoulli(
                torch.full((batch_size, seq_len, 1), keep_prob)
            ).to(fused.device)
            fused = fused * token_mask

        return logits

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for batch in dataloader:
            word_ids, char_ids, mask, labels = batch
            word_ids = word_ids.to(device)
            char_ids = char_ids.to(device)
            mask = mask.to(device)

            logits = model(word_ids, char_ids, mask)
            preds = torch.argmax(logits.cpu(), dim=-1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    all_preds_arr = np.array(all_preds)
    all_labels_arr = np.array(all_labels)
    return {
        "preds": all_preds_arr,
        "labels": all_labels_arr,
        "accuracy": accuracy_score(all_labels_arr, all_preds_arr),
        "f1_macro": f1_score(all_labels_arr, all_preds_arr, average="macro", zero_division=0),
        "f1_weighted": f1_score(all_labels_arr, all_preds_arr, average="weighted", zero_division=0),
    }

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for step, batch in enumerate(dataloader):
        word_ids, char_ids, mask, labels = batch
        word_ids = word_ids.to(device)
        char_ids = char_ids.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        logits = model(word_ids, char_ids, mask)
        loss = loss_fn(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CUSTOM_GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1

        if (step + 1) % 50 == 0:
            avg = total_loss / num_batches
            print(f"    Step {step + 1}/{len(dataloader)}, Loss: {avg:.4f}")

    return total_loss / max(num_batches, 1)

def main() -> None:
    print("=" * 60)
    print("  Custom Mongolian Classifier — CharCNN + BiLSTM + Attention")
    print("=" * 60)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = get_device()

    print(f"\n[custom] Using text column: {TEXT_COLUMN}")
    splits = load_and_prepare()

    X_train = splits["train"]["X"]
    y_train = splits["train"]["y"]
    X_val = splits["val"]["X"]
    y_val = splits["val"]["y"]
    X_test = splits["test"]["X"]
    y_test = splits["test"]["y"]

    unique_labels = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=unique_labels, y=y_train)
    label_weight_map = dict(zip(unique_labels, weights))
    class_weights = [label_weight_map[name] for name in LABEL_NAMES]
    print(f"[custom] Class weights: {dict(zip(LABEL_NAMES, [f'{w:.3f}' for w in class_weights]))}")

    print("[custom] Building vocabularies from training data ...")
    word_vocab = build_word_vocab(list(X_train), CUSTOM_WORD_VOCAB_SIZE)
    char_vocab = build_char_vocab(list(X_train), CUSTOM_CHAR_VOCAB_SIZE)
    print(f"[custom] Word vocab size: {len(word_vocab)}")
    print(f"[custom] Char vocab size: {len(char_vocab)}")

    best_model_dir = os.path.join(MODELS_DIR, "custom_model_best")
    os.makedirs(best_model_dir, exist_ok=True)
    with open(os.path.join(best_model_dir, "word_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(word_vocab, f, ensure_ascii=False, indent=1)
    with open(os.path.join(best_model_dir, "char_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(char_vocab, f, ensure_ascii=False, indent=1)
    print(f"[custom] Vocabularies saved to {best_model_dir}")

    train_dataset = MongolianTextDataset(
        X_train, y_train, word_vocab, char_vocab,
        CUSTOM_MAX_SEQ_LEN, CUSTOM_MAX_WORD_LEN,
    )
    val_dataset = MongolianTextDataset(
        X_val, y_val, word_vocab, char_vocab,
        CUSTOM_MAX_SEQ_LEN, CUSTOM_MAX_WORD_LEN,
    )
    test_dataset = MongolianTextDataset(
        X_test, y_test, word_vocab, char_vocab,
        CUSTOM_MAX_SEQ_LEN, CUSTOM_MAX_WORD_LEN,
    )

    train_loader = DataLoader(train_dataset, batch_size=CUSTOM_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CUSTOM_BATCH_SIZE * 2)
    test_loader = DataLoader(test_dataset, batch_size=CUSTOM_BATCH_SIZE * 2)

    model = MongolianClassifier(
        char_vocab_size=len(char_vocab),
        word_vocab_size=len(word_vocab),
        num_classes=len(LABEL_NAMES),
        char_embed_dim=CUSTOM_CHAR_EMBED_DIM,
        char_cnn_filters=CUSTOM_CHAR_CNN_FILTERS,
        char_cnn_kernels=CUSTOM_CHAR_CNN_KERNELS,
        word_embed_dim=CUSTOM_WORD_EMBED_DIM,
        fusion_dim=CUSTOM_FUSION_DIM,
        lstm_hidden=CUSTOM_LSTM_HIDDEN,
        lstm_layers=CUSTOM_LSTM_LAYERS,
        lstm_dropout=CUSTOM_LSTM_DROPOUT,
        attn_heads=CUSTOM_ATTN_HEADS,
        attn_dropout=CUSTOM_ATTN_DROPOUT,
        classifier_hidden=CUSTOM_CLASSIFIER_HIDDEN,
        classifier_dropout=CUSTOM_CLASSIFIER_DROPOUT,
        spatial_dropout=CUSTOM_EMBED_SPATIAL_DROPOUT,
        token_dropout_rate=CUSTOM_TOKEN_DROPOUT_RATE,
    )
    model.to(device)

    num_params = count_parameters(model)
    print(f"[custom] Total trainable parameters: {num_params:,}")
    assert 5_000_000 <= num_params <= 25_000_000, (
        f"Parameter count {num_params:,} outside target range [5M, 25M]"
    )
    print(f"[custom] Model moved to {device}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CUSTOM_LR, weight_decay=CUSTOM_WEIGHT_DECAY,
    )

    total_steps = len(train_loader) * CUSTOM_EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CUSTOM_LR,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy="cos",
    )
    print(f"[custom] Steps/epoch: {len(train_loader)}, total: {total_steps}")

    class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weight_tensor,
        label_smoothing=CUSTOM_LABEL_SMOOTHING,
    )

    best_f1 = 0.0
    patience_counter = 0

    print("\n[custom] Starting training ...")
    for epoch in range(CUSTOM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{CUSTOM_EPOCHS} ---")

        avg_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, device,
        )
        print(f"  Epoch {epoch + 1} avg loss: {avg_loss:.4f}")

        val_metrics = evaluate_model(model, val_loader, device)
        print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Val Macro-F1: {val_metrics['f1_macro']:.4f}")
        print(f"  Val Weighted-F1: {val_metrics['f1_weighted']:.4f}")

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            patience_counter = 0

            cpu_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(cpu_state, os.path.join(best_model_dir, "model.pt"))

            model_meta = {
                "char_vocab_size": len(char_vocab),
                "word_vocab_size": len(word_vocab),
                "num_classes": len(LABEL_NAMES),
            }
            with open(os.path.join(best_model_dir, "model_meta.json"), "w") as f:
                json.dump(model_meta, f, indent=2)

            print(f"  >> New best model saved (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{CUSTOM_PATIENCE})")
            if patience_counter >= CUSTOM_PATIENCE:
                print(f"  Early stopping triggered at epoch {epoch + 1}")
                break

    print("\n[custom] Training complete.")

    print(f"[custom] Loading best model from {best_model_dir}")
    cpu_state = torch.load(
        os.path.join(best_model_dir, "model.pt"),
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(cpu_state)
    model.to(device)

    print("\n" + "=" * 60)
    print("  CUSTOM MODEL VALIDATION EVALUATION")
    print("=" * 60)
    val_raw = evaluate_model(model, val_loader, device)
    val_pred_labels = np.array([ID2LABEL[p] for p in val_raw["preds"]])
    val_result = full_evaluation(X_val, y_val, val_pred_labels, "Custom_CharCNN_BiLSTM_Attn", "val")

    print("\n" + "=" * 60)
    print("  CUSTOM MODEL TEST EVALUATION")
    print("=" * 60)
    test_raw = evaluate_model(model, test_loader, device)
    test_pred_labels = np.array([ID2LABEL[p] for p in test_raw["preds"]])
    test_result = full_evaluation(X_test, y_test, test_pred_labels, "Custom_CharCNN_BiLSTM_Attn", "test")

    summary = f"""
{'=' * 60}
  CUSTOM MODEL EXPERIMENT COMPLETE
{'=' * 60}
  Architecture:  CharCNN + BiLSTM + MultiHead Attention Pooling
  Text column:   {TEXT_COLUMN}
  Device:        {device}
  Parameters:    {num_params:,}
  Val Macro-F1:  {val_result['f1_macro']:.4f}
  Test Macro-F1: {test_result['f1_macro']:.4f}
  Test Accuracy: {test_result['accuracy']:.4f}
  Saved model:   {best_model_dir}
{'=' * 60}
"""
    try:
        print(summary)
    except UnicodeEncodeError:
        print(summary.encode("ascii", errors="replace").decode("ascii"))

if __name__ == "__main__":
    main()
