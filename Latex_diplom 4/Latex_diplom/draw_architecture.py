#!/usr/bin/env python3
"""
Violence Detection System — Architecture Diagram  (v2 clean)
Publication-quality figure for academic thesis.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

# ── Canvas ────────────────────────────────────────────────────────────────────
FW, FH = 22, 14
fig = plt.figure(figsize=(FW, FH), dpi=200)
ax  = fig.add_subplot(111)
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis("off")
fig.patch.set_facecolor("white")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

# ── Colour palette  (face, edge) ─────────────────────────────────────────────
CLR = dict(
    sky    = ("#D6EAF8", "#2471A3"),
    mint   = ("#D5F5E3", "#1E8449"),
    amber  = ("#FAE5D3", "#CA6F1E"),
    plum   = ("#E8DAEF", "#7D3C98"),
    gold   = ("#FEF9E7", "#B7950B"),
    coral  = ("#FADBD8", "#C0392B"),
    silver = ("#F2F3F4", "#7F8C8D"),
)
DARK, GRAY = "#1B2631", "#626567"
A_MAIN = "#2C3E50"
A_KD   = "#C0392B"
A_PRPL = "#7D3C98"

# ── Helpers ───────────────────────────────────────────────────────────────────
def box(cx, cy, w, h, label, sub=None, clr="sky",
        bold=False, fs=8.8, lw=1.5, z=3):
    fc, ec = CLR[clr]
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.12,rounding_size=0.20",
        fc=fc, ec=ec, lw=lw, zorder=z, clip_on=False))
    fw = "bold" if bold else "normal"
    if sub:
        ax.text(cx, cy+0.14, label, ha="center", va="center",
                fontsize=fs, fontweight=fw, color=DARK, zorder=z+1)
        ax.text(cx, cy-0.19, sub, ha="center", va="center",
                fontsize=fs-1.8, color=GRAY, zorder=z+1)
    else:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fs, fontweight=fw, color=DARK, zorder=z+1)

def arr(x0, y0, x1, y1, color=A_MAIN, lw=1.6, rad=0.0, label=None, lx=0.12):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                        connectionstyle=f"arc3,rad={rad}",
                        mutation_scale=14), zorder=7)
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx+lx, my, label, fontsize=7.5, color=color,
                ha="left", va="center", style="italic", zorder=8)

def phase(x, y, w, h, title):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.2,rounding_size=0.35",
        fc="#F8F9FA", ec="#BFC9CA", lw=1.0, ls="--", zorder=1))
    ax.text(x+w/2, y+h+0.13, title, ha="center", va="bottom",
            fontsize=10.5, fontweight="bold", color=DARK, zorder=8)

# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
# Phase 1  x: 0.3 – 6.0   center 3.15
# Phase 2  x: 6.3 – 16.0  center 11.15
# Phase 3  x: 16.3 – 21.7 center 19.0

P1x, P1w = 0.30, 5.70
P2x, P2w = 6.30, 9.70
P3x, P3w = 16.30, 5.40

C1 = P1x + P1w/2          # 3.15
C2 = P2x + P2w/2          # 11.15
C3 = P3x + P3w/2          # 19.00

BW2 = 8.4    # main box width in phase 2  → x extent: 6.95 – 15.35

YTOP  = 11.7
YBOT  = 1.5

# ══════════════════════════════════════════════════════════════════════════════
# PHASE BACKGROUNDS
# ══════════════════════════════════════════════════════════════════════════════
phase(P1x, 1.0, P1w, 11.3, "Phase 1 — Data Preparation")
phase(P2x, 1.0, P2w, 11.3, "Phase 2 — Knowledge Distillation")
phase(P3x, 1.0, P3w, 11.3, "Phase 3 — Inference")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1
# ══════════════════════════════════════════════════════════════════════════════
box(C1, YTOP, P1w-0.5, 0.70, "Surveillance Video Datasets",
    clr="sky", bold=True, fs=9.5)

# Three dataset chips
for i, (name, sub) in enumerate([
        ("RWF-2000",    "2,000 clips"),
        ("UCF-Crime",   "850 clips"),
        ("Hockey Fight","1,000 clips")]):
    box(C1 - 1.6 + i*1.6, 10.55, 1.45, 0.58, name, sub, clr="sky", fs=7.8)

box(C1, 9.35, P1w-0.5, 0.65, "TXT Surveillance Reports",
    "2,269 annotation files", clr="sky", fs=9)

box(C1, 8.15, P1w-0.5, 0.70, "8-Frame Uniform Sampling",
    "fₖ = ⌊kF/N⌋  ·  uniform temporal stride", clr="mint", fs=8.8)

box(C1, 6.90, P1w-0.5, 0.70, "Preprocessing",
    "Resize 224×224  ·  ImageNet normalization", clr="mint", fs=8.8)

# Arrows
arr(C1, 11.35, C1, 10.87)
arr(C1-1.6, 10.25, C1, 9.68);  arr(C1, 10.25, C1, 9.68);  arr(C1+1.6, 10.25, C1, 9.68)
arr(C1, 9.02, C1, 8.50)
arr(C1, 7.80, C1, 7.25)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2
# ══════════════════════════════════════════════════════════════════════════════

# ── Teacher ──────────────────────────────────────────────────────────────────
box(C2, YTOP, BW2, 0.85, "Teacher Model — Qwen3-VL-30B-A3B",
    "MoE · 3.7B active / 30B total params · bfloat16 · A100 80 GB",
    clr="amber", bold=True, fs=10)

for i, (lbl, sub) in enumerate([
        ("Vision Encoder", "ViT + 2D-RoPE"),
        ("VL Merger",      "MLP projection"),
        ("LLM (MoE)",      "Qwen3 backbone")]):
    bx = C2 - 2.65 + i*2.65
    box(bx, 10.50, 2.20, 0.58, lbl, sub, clr="amber", fs=7.8)

arr(C2-2.65, 10.20, C2,      10.20, lw=1.2)
arr(C2,      10.20, C2+2.65, 10.20, lw=1.2)

# ── Q&A Generation ───────────────────────────────────────────────────────────
box(C2, 9.30, BW2, 0.65, "Q&A Prompt Generation",
    "4 types: event_type · subjects · chronology · conclusion",
    clr="gold", fs=9)

# ── Q&A Dataset ──────────────────────────────────────────────────────────────
box(C2, 8.10, BW2, 0.78, "Q&A Training Dataset — ~9,040 pairs",
    "train.jsonl (8,136)  +  val.jsonl (904)  ·  ~34 h generation time",
    clr="gold", bold=True, fs=9.3)

# ── QLoRA Training ───────────────────────────────────────────────────────────
box(C2, 6.80, BW2, 0.78, "QLoRA Fine-Tuning",
    "LoRA (r=16, α=32) · NF4 4-bit quant. · 5 epochs · cosine LR · warmup 5% · ~40 h",
    clr="mint", bold=True, fs=8.9)

# monitoring side panel (inside phase 2, right side, above training box)
mx = C2 + BW2/2 - 0.65      # 14.55
box(mx, 7.65, 2.0, 0.52, "WandB",      "real-time monitoring",  clr="silver", fs=7.8)
box(mx, 7.00, 2.0, 0.52, "Checkpoint", "every 500 steps",       clr="silver", fs=7.8)
arr(mx - 1.0, 6.80, mx - 1.0, 7.38, color=CLR["silver"][1], lw=1.1)
arr(mx - 1.0, 6.80, mx - 1.0, 7.00, color=CLR["silver"][1], lw=1.1)
# small arrows from training right edge to monitor boxes
tx_right = C2 + BW2/2   # 15.35
arr(tx_right, 6.90, mx+0.0, 7.65, color=CLR["silver"][1], lw=1.2, rad=-0.25)
arr(tx_right, 6.70, mx+0.0, 7.00, color=CLR["silver"][1], lw=1.2, rad= 0.15)

# ── Student ───────────────────────────────────────────────────────────────────
box(C2, 5.55, BW2, 0.85, "Student Model — Qwen2.5-VL-7B",
    "LoRA adapter (r=16, α=32) · ~85M trainable params (1.2% of 7B)",
    clr="plum", bold=True, fs=10)

for i, (lbl, sub) in enumerate([
        ("Vision Encoder", "ViT + MRoPE"),
        ("LLM Backbone",   "Qwen2.5 · 7B"),
        ("LoRA Adapter",   "r=16,  α=32")]):
    bx = C2 - 2.65 + i*2.65
    box(bx, 4.55, 2.20, 0.58, lbl, sub, clr="plum", fs=7.8)

arr(C2-2.65, 4.25, C2,      4.25, lw=1.2)
arr(C2,      4.25, C2+2.65, 4.25, lw=1.2)

# KD double arrow (left margin of phase 2)
kd_x = P2x + 0.55
ax.annotate("", xy=(kd_x, 5.55), xytext=(kd_x, YTOP),
    arrowprops=dict(arrowstyle="<->", color=A_KD, lw=1.8,
                    connectionstyle="arc3,rad=0"), zorder=7)
ax.text(kd_x - 0.35, 8.60, "Knowledge\nDistillation",
        ha="center", va="center", fontsize=8.5, fontweight="bold",
        color=A_KD, rotation=90, zorder=8)

# Phase-2 vertical flow arrows
arr(C2, 11.28, C2, 10.82)   # teacher → internal row
arr(C2, 10.20, C2,  9.63)   # internal → Q&A gen
arr(C2,  8.97, C2,  8.50)   # Q&A gen → dataset
arr(C2,  7.72, C2,  7.20)   # dataset → training
arr(C2,  6.43, C2,  5.98)   # training → student

# Phase 1 → Phase 2
arr(P1x+P1w-0.15, 6.90, P2x+0.75, 9.30,
    color="#5D6D7E", lw=2.0, label="frames + reports", lx=0.10)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3
# ══════════════════════════════════════════════════════════════════════════════
BW3 = P3w - 0.6   # 4.8

box(C3, YTOP, BW3, 0.65, "Video Input",
    "Surveillance camera stream", clr="sky", fs=9)

box(C3, 10.45, BW3, 0.68, "8-Frame Sampling",
    "224×224  ·  ImageNet norm", clr="mint", fs=9)

box(C3,  9.15, BW3, 1.00, "Fine-tuned Qwen2.5-VL-7B",
    "LoRA merged  ·  NF4 quantization\nInference: ~200 ms  (4× faster than teacher)",
    clr="plum", bold=True, fs=9.3)

box(C3,  7.65, BW3, 0.72, "Structured Mongolian Report",
    "8-section output format", clr="coral", bold=True, fs=9.3)

# 8 output sections in 2×4 grid
sections = [
    ("ШОШГО (Label)",    "ИТГЭЛЦЭЛ (Conf.)"),
    ("СУБЪЕКТҮҮД",       "ҮЙЛДЛҮҮД"),
    ("ЯВЦ (T0 – T3)",    "НОТЛОХ БАРИМТ"),
    ("ДҮГНЭЛТ",          "ЗӨВЛӨМЖ (Alert)"),
]
for i, (sl, sr) in enumerate(sections):
    yy = 6.75 - i * 0.60
    box(C3 - 1.15, yy, 2.05, 0.48, sl, clr="coral", fs=7.8)
    box(C3 + 1.15, yy, 2.05, 0.48, sr, clr="coral", fs=7.8)

# Phase-3 arrows
arr(C3, 11.38, C3, 10.78)
arr(C3, 10.12, C3,  9.65)
arr(C3,  8.65, C3,  8.02)
arr(C3,  7.30, C3,  7.00)

# Phase 2 → Phase 3 (LoRA merge)
arr(C2 + BW2/2, 5.55, P3x+0.55, 9.15,
    color=A_PRPL, lw=2.0, label="LoRA merge →", lx=0.10)

# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
ax.text(FW/2, 13.38,
    "Violence Detection System — Architecture Overview",
    ha="center", va="center",
    fontsize=15, fontweight="bold", color=DARK, zorder=9)
ax.text(FW/2, 12.88,
    "Teacher–Student Knowledge Distillation  ·  "
    "QLoRA Fine-Tuning  ·  Mongolian Structured Report",
    ha="center", va="center",
    fontsize=10, color=GRAY, style="italic", zorder=9)

# ══════════════════════════════════════════════════════════════════════════════
# LEGEND
# ══════════════════════════════════════════════════════════════════════════════
legend = [("sky","Data / Input"),("mint","Processing"),
          ("amber","Teacher Model"),("plum","Student Model"),
          ("gold","Q&A Dataset"),("coral","Output")]
gap = (FW - 1.4) / len(legend)
for i, (ck, lbl) in enumerate(legend):
    fc, ec = CLR[ck]
    xr = 0.7 + i*gap
    ax.add_patch(FancyBboxPatch((xr, 0.22), 0.52, 0.38,
        boxstyle="round,pad=0.05,rounding_size=0.07",
        fc=fc, ec=ec, lw=1.2, zorder=6))
    ax.text(xr+0.62, 0.41, lbl, va="center",
            fontsize=8.5, color=DARK, zorder=7)

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
plt.tight_layout(pad=0.4)
for fn in ("Figures/architecture_diagram.pdf",
           "Figures/architecture_diagram.png"):
    plt.savefig(fn, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"  ✓  {fn}")
