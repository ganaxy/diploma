import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

LABELED_PATH = r'C:\Users\M Tech\Desktop\diplom\sample scores\10k_fully_labeled.csv'
CORPUS_PATH  = r'C:\Users\M Tech\Desktop\diplom\cleaned_data_v1\merged_cleaned_comments.csv'
OUT_DIR      = r'C:\Users\M Tech\Desktop\diplom\latex file\Figures'

os.makedirs(OUT_DIR, exist_ok=True)

LABEL_MN = {
    'CONSTRUCTIVE': 'Бүтээлч шүүмжлэл',
    'NEUTRAL':      'Саармаг',
    'TOXIC':        'Хортой сөрөг',
    'POSITIVE':     'Эерэг',
}

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

print("=" * 60)
print("STEP 1: Loading labeled dataset")
print("=" * 60)
df = pd.read_csv(LABELED_PATH)
print(f"Columns: {df.columns.tolist()}")
print(f"Shape:   {df.shape}")

label_col = 'label'
counts_raw = df[label_col].value_counts()
total = len(df)
print(f"\nLabel column: '{label_col}'")
print(f"Total rows: {total}")
print("Label counts:")
print(counts_raw.to_string())

counts_mn = counts_raw.rename(index=LABEL_MN)
pct_mn    = (counts_raw / total * 100).rename(index=LABEL_MN)

display_order = ['Бүтээлч шүүмжлэл', 'Саармаг', 'Хортой сөрөг', 'Эерэг']
counts_ordered = counts_mn.reindex(display_order)
pct_ordered    = pct_mn.reindex(display_order)

print("\n" + "=" * 60)
print("SUPPLEMENT 6A: Terminal summary")
print("=" * 60)
print(f"Total labeled records : {total}")
print(f"Unique categories     : {counts_raw.nunique()}")
largest  = counts_mn.idxmax()
smallest = counts_mn.idxmin()
print(f"Largest  category     : {largest} ({counts_mn[largest]})")
print(f"Smallest category     : {smallest} ({counts_mn[smallest]})")
print(f"Imbalance ratio (L/S) : {counts_mn[largest] / counts_mn[smallest]:.2f}x")

print("\nGenerating Figure 1 (horizontal bar chart) ...")
fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.barh(display_order, counts_ordered.values,
               color=COLORS, edgecolor='white', height=0.6)

for bar, cnt, pct in zip(bars, counts_ordered.values, pct_ordered.values):
    ax.text(bar.get_width() + 35, bar.get_y() + bar.get_height() / 2,
            f'{cnt:,}  ({pct:.1f}%)', va='center', ha='left', fontsize=10)

ax.set_xlabel('Сэтгэгдлийн тоо', fontsize=11)
ax.set_title('Ангилал тус бүрийн шошголсон өгөгдлийн тоо (n=10,000)',
             fontsize=12, fontweight='bold', pad=14)
ax.set_xlim(0, counts_ordered.max() * 1.25)
ax.invert_yaxis()
ax.spines[['top', 'right']].set_visible(False)
ax.tick_params(axis='y', labelsize=11)
plt.tight_layout()
path1 = os.path.join(OUT_DIR, 'data_categories_bar.png')
fig.savefig(path1, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {path1}")

print("Generating Figure 2 (pie chart) ...")
fig, ax = plt.subplots(figsize=(8, 7))

wedges, texts, autotexts = ax.pie(
    counts_ordered.values,
    labels=display_order,
    autopct='%1.1f%%',
    colors=COLORS,
    startangle=140,
    pctdistance=0.78,
    wedgeprops=dict(edgecolor='white', linewidth=1.5),
    textprops=dict(fontsize=11),
)
for at in autotexts:
    at.set_fontsize(10)
    at.set_fontweight('bold')

ax.set_title('Шошголсон өгөгдлийн ангиллын харьцаа (n=10,000)',
             fontsize=12, fontweight='bold', pad=18)
plt.tight_layout()
path2 = os.path.join(OUT_DIR, 'data_categories_pie.png')
fig.savefig(path2, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {path2}")

print("Generating Figure 3 (summary table) ...")
cum_pct = pct_ordered.cumsum()
table_data = []
for cat in display_order:
    table_data.append([cat,
                        f'{counts_ordered[cat]:,}',
                        f'{pct_ordered[cat]:.1f}%',
                        f'{cum_pct[cat]:.1f}%'])
table_data.append(['Нийт  (Total)',

col_labels = ['Ангилал', 'Тоо', 'Хувь (%)', 'Хуримтлагдсан (%)']

fig, ax = plt.subplots(figsize=(10, 3.5))
ax.axis('off')

tbl = ax.table(cellText=table_data,
               colLabels=col_labels,
               cellLoc='center',
               loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.0, 2.0)

for j in range(len(col_labels)):
    cell = tbl[(0, j)]
    cell.set_facecolor('#2c3e50')
    cell.set_text_props(color='white', fontweight='bold')

row_colors = ['#f2f2f2', '#ffffff']
for i in range(1, len(table_data) + 1):
    shade = row_colors[(i - 1) % 2]
    for j in range(len(col_labels)):
        cell = tbl[(i, j)]
        cell.set_facecolor(shade)
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor('#dce8f5')

ax.set_title('Шошголсон өгөгдлийн бүтэц — дэлгэрэнгүй (n=10,000)',
             fontsize=12, fontweight='bold', pad=18)
plt.tight_layout()
path3 = os.path.join(OUT_DIR, 'data_summary_table.png')
fig.savefig(path3, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {path3}")

print("\nGenerating Supplement 6B (30k source chart) ...")
df30 = pd.read_csv(CORPUS_PATH)
print(f"30k corpus shape: {df30.shape}")

df30['source_norm'] = df30['SOURCE'].str.strip().str.lower()
src_map = {
    'news.mn': 'news.mn', 'gogo.mn': 'gogo.mn',
    'ikon.mn': 'IKON.mn', 'e-mongolia': 'E-Mongolia',
    'medee.mn': 'Medee.mn', 'zaluusinfo': 'Zaluusinfo',
    'eagle news': 'Eagle News',
    'мэдэхгүй зүйлээ асуу': 'Мэдэхгүй зүйлээ асуу',
}
df30['source_label'] = df30['source_norm'].map(lambda x: src_map.get(x, x))
src_counts = df30['source_label'].value_counts().sort_values(ascending=False)
print("Source counts (30k):")
print(src_counts.to_string())

palette_30k = plt.cm.tab10(np.linspace(0, 0.9, len(src_counts)))
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(src_counts.index.tolist(), src_counts.values,
               color=palette_30k, edgecolor='white', height=0.6)
for bar, cnt in zip(bars, src_counts.values):
    pct = cnt / len(df30) * 100
    ax.text(bar.get_width() + 60, bar.get_y() + bar.get_height() / 2,
            f'{cnt:,}  ({pct:.1f}%)', va='center', fontsize=9)

ax.set_xlabel('Сэтгэгдлийн тоо', fontsize=11)
ax.set_title(f'Нийт цуглуулсан өгөгдлийн эх үүсвэр (n={len(df30):,})',
             fontsize=12, fontweight='bold', pad=14)
ax.set_xlim(0, src_counts.max() * 1.28)
ax.invert_yaxis()
ax.spines[['top', 'right']].set_visible(False)
ax.tick_params(axis='y', labelsize=10)
plt.tight_layout()
path4 = os.path.join(OUT_DIR, 'data_by_source_raw.png')
fig.savefig(path4, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {path4}")

print("\n" + "=" * 60)
print("ALL FIGURES GENERATED SUCCESSFULLY")
print("=" * 60)
print(f"  Figure 1 (bar):    {path1}")
print(f"  Figure 2 (pie):    {path2}")
print(f"  Figure 3 (table):  {path3}")
print(f"  Figure 4 (source): {path4}")

print("\n--- EXACT COUNTS FOR LATEX TABLES ---")
for cat in display_order:
    raw_key = {v: k for k, v in LABEL_MN.items()}[cat]
    print(f"  {cat:30s} {counts_ordered[cat]:5d}  {pct_ordered[cat]:.1f}%")
print(f"  {'НИЙТ':30s} {total:5d}  100.0%")
print(f"\n30k total rows: {len(df30)}")
for src, cnt in src_counts.items():
    pct = cnt / len(df30) * 100
    print(f"  {src:30s} {cnt:6d}  ({pct:.1f}%)")
