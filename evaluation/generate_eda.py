import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter
import re

sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'Arial'

BASE_DIR = r"C:\Users\M Tech\Desktop\diplom\sample scores"
DATA_PATH = os.path.join(BASE_DIR, "relabeled_v7_normalized.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "eda_plots")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

df = pd.read_csv(DATA_PATH)

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='label', palette='viridis', order=['POSITIVE', 'NEUTRAL', 'CONSTRUCTIVE', 'TOXIC'])
plt.title('Label Distribution in 10,000 Comment Dataset', fontsize=15)
plt.xlabel('Category', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.savefig(os.path.join(OUTPUT_DIR, "label_distribution.png"), dpi=300, bbox_inches='tight')
plt.close()

df['text_len'] = df['text_normalized'].astype(str).str.len()
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='label', y='text_len', palette='magma', order=['POSITIVE', 'NEUTRAL', 'CONSTRUCTIVE', 'TOXIC'])
plt.yscale('log')
plt.title('Comment Length (Characters) by Label', fontsize=15)
plt.xlabel('Category', fontsize=12)
plt.ylabel('Length (Log Scale)', fontsize=12)
plt.savefig(os.path.join(OUTPUT_DIR, "length_distribution.png"), dpi=300, bbox_inches='tight')
plt.close()

def get_top_ngrams(corpus, n=None, top_k=20):

    words = []
    for text in corpus:

        text = re.sub(r'[^\w\s]', '', str(text).lower())
        tokens = text.split()
        if n == 1:
            words.extend(tokens)
        else:
            words.extend([" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)])

    stopwords = {'нь', 'бөгөөд', 'байна', 'байгаа', 'гэж', 'бол', 'байсан', 'болсон', 'бүх', 'тэр', 'энэ', 'нэг', 'юм', 'байх', 'гэх', 'уу', 'үү', 'аа', 'ээ'}

    filtered_words = [w for w in words if w not in stopwords]
    if n > 1:
        filtered_words = [w for w in words if not any(sw in w.split() for sw in stopwords)]

    return Counter(filtered_words).most_common(top_k)

def plot_ngrams(label_name, n=1, color='blue'):
    corpus = df[df['label'] == label_name]['text_normalized']
    top_words = get_top_ngrams(corpus, n=n)

    if not top_words:
        return

    words, counts = zip(*top_words)
    plt.figure(figsize=(10, 8))
    sns.barplot(x=list(counts), y=list(words), color=color)
    plt.title(f'Top {n}-grams in {label_name} Comments', fontsize=15)
    plt.xlabel('Frequency', fontsize=12)
    plt.savefig(os.path.join(OUTPUT_DIR, f"top_{n}gram_{label_name.lower()}.png"), dpi=300, bbox_inches='tight')
    plt.close()

plot_ngrams('TOXIC', n=1, color='salmon')
plot_ngrams('TOXIC', n=2, color='darkred')
plot_ngrams('CONSTRUCTIVE', n=1, color='lightgreen')
plot_ngrams('CONSTRUCTIVE', n=2, color='darkgreen')

print(f"EDA plots saved to {OUTPUT_DIR}")
