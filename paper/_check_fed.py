# -*- coding: utf-8 -*-
"""FED-compliance checks for manuscript_fed.tex (temp tool, not part of submission)."""
import re
import sys
sys.stdout.reconfigure(encoding='utf-8')

tex = open('manuscript_fed.tex', encoding='utf-8').read()

# 1. Abstract word count (FED limit: 250 words)
m = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', tex, re.S)
abs_text = m.group(1)
abs_text = re.sub(r'\\small\s*|\\textbf|\\textit|\\texttt|[{}]|\\textsuperscript\{[^}]*\}', ' ', abs_text)
abs_text = re.sub(r'\$[^$]*\$', '0', abs_text)
words = [w for w in abs_text.split() if re.search(r'[A-Za-z0-9]', w)]
print('Abstract word count:', len(words), '(FED limit: 250)')

# 2. Highlights char counts from paper/highlights.txt (Elsevier limit: 85)
print('--- Highlights (from highlights.txt) ---')
for line in open('highlights.txt', encoding='utf-8'):
    s = line.strip()
    if s and s[0].isdigit():
        body = s.split('. ', 1)[1]
        print(len(body), 'chars |', body)

# 3. Citation order check
bibitems = re.findall(r'\\bibitem\{([^}]+)\}', tex)
first_cite = []
for cite in re.finditer(r'\\cite\{([^}]+)\}', tex):
    for key in cite.group(1).split(','):
        key = key.strip()
        if key not in first_cite:
            first_cite.append(key)
print('--- References ---')
print('bibitem order == first-citation order:', bibitems == first_cite)
print('bibitems:', len(bibitems), 'cited keys:', len(first_cite))

# 4. Includegraphics files
print('--- Figures ---')
for mm in re.finditer(r'\\includegraphics.*?\{([^}]+)\}', tex):
    import os
    p = mm.group(1)
    print(p, 'exists' if os.path.exists(p) else 'MISSING')
