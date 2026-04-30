#!/usr/bin/env python3
"""Static check: cite ↔ bibitem, ref/eqref ↔ label, label uniqueness."""
import re, glob, os
from collections import Counter

tex_files = sorted(glob.glob('sections/*.tex') + glob.glob('appendix/*.tex') + ['main.tex'])

all_cites, all_refs, all_labels = set(), set(), []
for f in tex_files:
    with open(f, encoding='utf-8') as fh:
        s = fh.read()
    # \cite{a,b,c} or \cite[opt]{a,b}
    for m in re.finditer(r'\\cite(?:\[[^\]]*\])?\{([^}]+)\}', s):
        for k in m.group(1).split(','):
            all_cites.add(k.strip())
    for m in re.finditer(r'\\(?:ref|eqref)\{([^}]+)\}', s):
        all_refs.add(m.group(1).strip())
    for m in re.finditer(r'\\label\{([^}]+)\}', s):
        all_labels.append((m.group(1).strip(), f))

bibitems = set()
with open('sections/bibliography.tex', encoding='utf-8') as fh:
    s = fh.read()
for m in re.finditer(r'\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}', s):
    bibitems.add(m.group(1).strip())

label_set = set(l[0] for l in all_labels)
print(f'cites:   {len(all_cites):>3}   bibitems: {len(bibitems):>3}')
print(f'refs:    {len(all_refs):>3}   labels:   {len(all_labels):>3}   unique: {len(label_set)}')
print('missing_cites:    ', sorted(all_cites - bibitems))
print('unused_bibitems:  ', sorted(bibitems - all_cites))
print('missing_refs:     ', sorted(all_refs - label_set))

c = Counter(x[0] for x in all_labels)
dups = [(k,v) for k,v in c.items() if v>1]
print('duplicate_labels: ', dups)
