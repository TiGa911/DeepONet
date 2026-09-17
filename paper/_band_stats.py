# -*- coding: utf-8 -*-
"""临时：全局验收区间统计（可并入 make_latex_tables.py）。"""
import sys
import numpy as np
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

from generate_paper_figures import (
    load_npz_metadata, build_acceptance_band, in_band_fraction,
    NN_METHODS, METHOD_SPECS, DIAGNOSTICS)

metas = load_npz_metadata()
res = {d: {m: [] for m in NN_METHODS} for d in DIAGNOSTICS}
for (shot, td) in metas:
    for diag in DIAGNOSTICS:
        band = build_acceptance_band(shot, td, diag)
        if band is None:
            continue
        for m in NN_METHODS:
            f = in_band_fraction(shot, td, diag, m, band=band)
            if f is not None:
                res[diag][m].append(f)

print('BAND TABLE (mean in-band % ± std, n)')
for diag in DIAGNOSTICS:
    cells = []
    for m in NN_METHODS:
        v = res[diag][m]
        cells.append(f'{np.mean(v)*100:.1f}$\\pm${np.std(v)*100:.1f}')
    n = len(res[diag]['nn'])
    print(f'{diag} (n={n}): ' + ' & '.join(cells))
