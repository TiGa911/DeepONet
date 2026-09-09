# -*- coding: utf-8 -*-
"""
significance_tests.py — Wilcoxon signed-rank 配对检验（审稿意见 M2）

对每个（诊断, 模式）组内的时间点级 MAE 做配对 Wilcoxon 检验，
判断各方法对之间的均值差异是否统计显著。

用法:
  python significance_tests.py [--csv paper_results/all_metrics.csv]
"""
import argparse
import csv
import itertools
import sys

import numpy as np
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')

parser = argparse.ArgumentParser()
parser.add_argument('--csv', default='paper_results/all_metrics.csv')
args = parser.parse_args()

METHOD_ORDER = ['nn', 'lstm', 'cnn', 'cnn_deeponet', 'transformer']
LABELS = {'nn': 'ProfileNet', 'lstm': 'LSTM', 'cnn': 'CNN-1D',
          'cnn_deeponet': 'CNN-DeepONet', 'transformer': 'Transformer'}
DIAGS = ['Te', 'ne', 'Ti']

rows = list(csv.DictReader(open(args.csv, encoding='utf-8')))


def get_mae(diag, mode, method):
    out = {}
    for r in rows:
        if r['diagnostic'] != diag or r['method'] != method:
            continue
        if mode == 'All' or r['plasma_mode'] == mode:
            out[(r['shot'], r['time_dir'])] = float(r['MAE'])
    return out


for diag in DIAGS:
    for mode in ['All', 'H', 'L']:
        maes = {m: get_mae(diag, mode, m) for m in METHOD_ORDER}
        common = set.intersection(*(set(v) for v in maes.values() if v))
        if len(common) < 2:
            continue
        print(f'\n=== {diag}  {mode}  (n={len(common)} time points) ===')
        print(f'{"pair":36s} {"A mean":>7s} {"B mean":>7s} {"p":>8s}  verdict')
        for a, b in itertools.combinations(METHOD_ORDER, 2):
            A = np.array([maes[a][k] for k in sorted(common)])
            B = np.array([maes[b][k] for k in sorted(common)])
            if np.allclose(A, B):
                continue
            w = stats.wilcoxon(A, B)
            sig = ('***' if w.pvalue < 0.001 else
                   '**' if w.pvalue < 0.01 else
                   '*' if w.pvalue < 0.05 else 'n.s.')
            print(f'{LABELS[a]:>14s}  vs {LABELS[b]:<14s} '
                  f'{A.mean():7.3f} {B.mean():7.3f} {w.pvalue:8.4f}  {sig}')
