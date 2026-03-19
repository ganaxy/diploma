import pandas as pd, re, sys
sys.stdout.reconfigure(encoding='utf-8')

mg = pd.read_csv(r'C:\Users\M Tech\Desktop\diplom\sample scores\relabeled_v7_aug_v2_normalized.csv', encoding='utf-8-sig')
base = mg[mg['source'] != 'augmented']
aug  = mg[mg['source'] == 'augmented']

emoji_re = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F1FF]"
)
url_re   = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
html_re  = re.compile(r'<[^>]+>')
digit_re = re.compile(r'\d')

PUNCT = set('.,;:!?(){}[]"\'-+_*&^%$#@~/<>=`«»–—…\\')

def cyr_ratio(s):
    s = str(s)
    meaningful = [c for c in s if (not c.isspace()) and (not c.isdigit()) and (c not in PUNCT)]
    if not meaningful:
        return 0.0
    cyr = sum(1 for c in meaningful if 'Ѐ' <= c <= 'ӿ')
    return cyr / len(meaningful)

for name, df in [('BASE v7 (10000)', base), ('AUG v2 (1020)', aug)]:
    txt = df['text_light_clean'].astype(str)
    print(f'--- {name} ---')
    print(f'  has emoji   : {int(txt.str.contains(emoji_re).sum())}')
    print(f'  has URL     : {int(txt.str.contains(url_re).sum())}')
    print(f'  has HTML tag: {int(txt.str.contains(html_re).sum())}')
    print(f'  has digits  : {int(txt.str.contains(digit_re).sum())}')
    cratio = txt.apply(cyr_ratio)
    print(f'  cyr<85%     : {int((cratio<0.85).sum())}')
    print(f'  median cyr% : {cratio.median()*100:.1f}')
    print()
