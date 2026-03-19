import pandas as pd

main = pd.read_excel('10k_fully_labeled_relabeled _v6.xlsx')
corr = pd.read_csv('mislabeled_rows_v6.csv', sep=',', encoding='utf-8-sig')

print('Label value_counts BEFORE patching:')
print(main['label'].value_counts())
print()

corr_map = corr.set_index('id')['relabaled_labels']
mask = main['id'].isin(corr_map.index)
patched = mask.sum()
main.loc[mask, 'label'] = main.loc[mask, 'id'].map(corr_map)

print(f'Total rows patched: {patched}')
print()
print('Label value_counts AFTER patching:')
print(main['label'].value_counts())

main[['id', 'label']].to_csv(
    '10k_fully_labeled_relabeled_v7_labels.csv',
    sep='\t', encoding='utf-8-sig', index=False,
)
