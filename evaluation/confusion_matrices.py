import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT_DIR = r'C:\Users\M Tech\Desktop\diplom\latex file\Figures\CM'
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'figure.facecolor': 'white',
})

def plot_cm(cm, class_names, title, filename):
    cm = np.array(cm, dtype=int)
    n = len(class_names)
    row_sums = cm.sum(axis=1, keepdims=True)

    fig_size = max(4.5, n * 1.8)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size - 0.3))

    cmap = plt.cm.Blues
    im = ax.imshow(cm_pct, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Row proportion', fontsize=10)

    ticks = np.arange(n)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=25, ha='right', rotation_mode='anchor')

    thresh = 0.5
    for i in range(n):
        for j in range(n):
            count = cm[i, j]
            pct = cm_pct[i, j] * 100
            color = 'white' if cm_pct[i, j] > thresh else '#1a1a2e'
            ax.text(j, i,
                    f'{count}\n({pct:.1f}%)',
                    ha='center', va='center',
                    color=color, fontsize=10, fontweight='bold',
                    linespacing=1.4)

    ax.set_ylabel('Жинхэнэ шошго (Actual)', fontsize=12, labelpad=8)
    ax.set_xlabel('Таамагласан шошго (Predicted)', fontsize=12, labelpad=8)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=14)

    for k in range(n + 1):
        ax.axhline(k - 0.5, color='white', linewidth=1.5)
        ax.axvline(k - 0.5, color='white', linewidth=1.5)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {filename}')

cm_s1 = [
]
labels_s1 = ['Эерэг', 'Саармаг', 'Сөрөг']
plot_cm(cm_s1, labels_s1,
        'Stage 1 — MN-BERT (3-ангилал)',
        os.path.join(OUT_DIR, 'cm_stage1.png'))

cm_s2 = [
]
labels_s2 = ['Хортой\nсөрөг', 'Бүтээлч\nшүүмжлэл']
plot_cm(cm_s2, labels_s2,
        'Stage 2 — MN-BERT (2-ангилал)',
        os.path.join(OUT_DIR, 'cm_stage2.png'))

cm_e2e = [
]
labels_e2e = ['Эерэг', 'Саармаг', 'Хортой\nсөрөг', 'Бүтээлч\nшүүмжлэл']
plot_cm(cm_e2e, labels_e2e,
        'End-to-End — Flat MN-BERT (4-ангилал)',
        os.path.join(OUT_DIR, 'cm_e2e.png'))

print('\nDone. All 3 confusion matrices saved to:', OUT_DIR)
