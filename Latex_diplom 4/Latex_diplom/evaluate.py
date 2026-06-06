"""
Violence Detection — Evaluation Script
Qwen2.5-VL-7B + QLoRA adapter
RunPod дээр ажиллуулах

Ашиглалт:
    python evaluate.py --adapter your-hf-user/your-adapter --test val.jsonl
"""

import os, io, re, json, base64, argparse
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, classification_report,
)
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel


# ── Label parsing (таны датасетийн формат) ──────────────────────────────────

# val.jsonl-ээс олдсон жинхэнэ label-ууд:
# "хүчирхийлэл"  →  violence
# "хүчирхийлэлгүй" →  non-violence

LABEL_MAP = {
    "хүчирхийлэл":    "violence",
    "хүчирхийлэлгүй": "non-violence",
    # fallback variants
    "хүчирхийллийн бус": "non-violence",
    "violence":          "violence",
    "non-violence":      "non-violence",
    "non_violence":      "non-violence",
}

def parse_label(text: str) -> str:
    t = text.lower().strip()
    # "ШОШГО: ХҮЧИРХИЙЛЭЛ" эсвэл "ШОШГО: ХҮЧИРХИЙЛЭЛГҮЙ"
    m = re.search(r"шошго\s*[:\-]\s*(.+)", t)
    if m:
        raw = m.group(1).strip().rstrip(".,;\n")
        return LABEL_MAP.get(raw, raw)
    # direct match
    for key, val in LABEL_MAP.items():
        if key in t:
            return val
    return "unknown"

def get_ground_truth(sample: dict) -> str:
    for msg in reversed(sample.get("messages", [])):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "text":
                        return parse_label(b["text"])
            elif isinstance(content, str):
                return parse_label(content)
    return "unknown"


# ── Image decode ─────────────────────────────────────────────────────────────

def decode_image(data_uri: str) -> Image.Image:
    if "base64," in data_uri:
        data_uri = data_uri.split("base64,", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(data_uri))).convert("RGB")

def extract_frames(messages, max_frames=8):
    """Assistant turn-г хасаж, зургуудыг задлах."""
    images, prompt = [], []
    for msg in messages:
        if msg.get("role") == "assistant":
            break
        role = msg["role"]
        content = msg.get("content", "")
        if isinstance(content, list):
            blocks = []
            for b in content:
                if isinstance(b, dict) and b.get("type") == "image":
                    raw = b.get("image", "")
                    if raw and len(images) < max_frames:
                        try:
                            images.append(decode_image(raw))
                            blocks.append({"type": "image"})
                        except Exception:
                            pass  # зураг унших алдаа → алгасах
                    # max_frames хүрсэн бол зургийн block-г орхих
                    elif not raw:
                        blocks.append(b)
                else:
                    blocks.append(b)
            prompt.append({"role": role, "content": blocks})
        else:
            prompt.append({"role": role, "content": content})
    return images, prompt


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(model, processor, sample, device, max_new_tokens=48):
    images, prompt_msgs = extract_frames(sample["messages"], max_frames=8)

    text = processor.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(
        text=text,
        images=images if images else None,
        return_tensors="pt",
        max_length=4096,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    generated = out[0][inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(generated, skip_special_tokens=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    print("\n" + "="*60)
    print("  Violence Detection — Evaluation")
    print(f"  Adapter : {args.adapter}")
    print(f"  Test    : {args.test}")
    print("="*60)

    # 4-bit load
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    BASE = "Qwen/Qwen2.5-VL-7B-Instruct"

    print(f"\n  [1/3] Loading base model ({BASE}) ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE, quantization_config=bnb, device_map="auto", trust_remote_code=True,
    )
    print(f"  [2/3] Loading LoRA adapter ({args.adapter}) ...")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    print(f"  [3/3] Loading processor ...")
    processor = AutoProcessor.from_pretrained(
        BASE, trust_remote_code=True, max_pixels=512*28*28,
    )
    device = next(model.parameters()).device
    print(f"  Device: {device}\n")

    # Data
    samples = []
    with open(args.test, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    print(f"  Samples: {len(samples)}\n" + "-"*60)

    y_true, y_pred, details = [], [], []
    n_unknown = 0

    for i, sample in enumerate(tqdm(samples, desc="Evaluating")):
        gt = get_ground_truth(sample)
        try:
            raw = predict(model, processor, sample, device)
            pred = parse_label(raw)
        except Exception as e:
            print(f"\n  [!] Sample {i} error: {e}")
            raw, pred = "", "unknown"

        if pred == "unknown":
            n_unknown += 1
            print(f"\n  [!] Unknown output ({i}): {raw[:100]}")

        y_true.append(gt)
        y_pred.append(pred)
        details.append({
            "idx": i, "gt": gt, "pred": pred,
            "correct": gt == pred, "raw": raw[:150],
        })

        if (i + 1) % 50 == 0:
            acc = sum(d["correct"] for d in details) / len(details)
            print(f"\n  Progress [{i+1}/{len(samples)}]  acc={acc:.3f}")

    # ── Metrics ──────────────────────────────────────────────────────────────
    valid = [(t, p) for t, p in zip(y_true, y_pred)
             if t != "unknown" and p != "unknown"]

    if not valid:
        print("\nERROR: 0 valid predictions. Label parsing-г шалгана уу.")
        return

    vt, vp = zip(*valid)
    labs = ["non-violence", "violence"]

    acc   = accuracy_score(vt, vp)
    f1w   = f1_score(vt, vp, average="weighted",  labels=labs, zero_division=0)
    f1m   = f1_score(vt, vp, average="macro",     labels=labs, zero_division=0)
    prec  = precision_score(vt, vp, average="weighted", labels=labs, zero_division=0)
    rec   = recall_score(vt, vp, average="weighted",    labels=labs, zero_division=0)
    cm    = confusion_matrix(vt, vp, labels=labs)

    print("\n" + "="*60)
    print("  RESULTS")
    print("="*60)
    print(f"  Evaluated  : {len(valid)} / {len(samples)}")
    print(f"  Unknown    : {n_unknown}")
    print(f"\n  Accuracy   : {acc:.4f}")
    print(f"  F1 weighted: {f1w:.4f}   ← энэ л тезисийн үр дүн")
    print(f"  F1 macro   : {f1m:.4f}")
    print(f"  Precision  : {prec:.4f}")
    print(f"  Recall     : {rec:.4f}")
    print(f"\n  Per-class:")
    print(classification_report(vt, vp, labels=labs, zero_division=0))
    print(f"  Confusion matrix  [non-viol | violence]:")
    print(f"    non-violence  {cm[0]}")
    print(f"    violence      {cm[1]}")
    print("="*60)

    # ── Save ─────────────────────────────────────────────────────────────────
    out = {
        "adapter":  args.adapter,
        "test":     args.test,
        "n_total":  len(samples),
        "n_valid":  len(valid),
        "n_unknown": n_unknown,
        "metrics": {
            "accuracy":    round(acc,  4),
            "f1_weighted": round(f1w,  4),
            "f1_macro":    round(f1m,  4),
            "precision":   round(prec, 4),
            "recall":      round(rec,  4),
        },
        "confusion_matrix": {"labels": labs, "matrix": cm.tolist()},
        "details": details,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved → {args.output}")
    print(f"\n  ✓  F1 (weighted) = {f1w:.4f}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter",     required=True,  help="HF adapter repo (user/name)")
    ap.add_argument("--test",        default="val.jsonl")
    ap.add_argument("--output",      default="results.json")
    ap.add_argument("--max_samples", type=int, default=0, help="0=бүгд")
    main(ap.parse_args())