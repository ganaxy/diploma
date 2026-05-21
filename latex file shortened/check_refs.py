import re, glob
refs = set()
labels = set()
ref_pat = re.compile(r'\\ref\{([^}]+)\}')
label_pat = re.compile(r'\\label\{([^}]+)\}')
for f in glob.glob('Chapters/*.tex') + glob.glob('FrontBackMatter/*.tex'):
    content = open(f, encoding='utf-8', errors='replace').read()
    refs.update(ref_pat.findall(content))
    labels.update(label_pat.findall(content))
undefined = refs - labels
print('UNDEFINED REFS:')
for r in sorted(undefined):
    print(' ', r)
print()
print('Total refs: %d, Total labels: %d, Undefined: %d' % (len(refs), len(labels), len(undefined)))
