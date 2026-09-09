# -*- coding: utf-8 -*-
"""
make_latex_tables.py — 从 paper_results/all_metrics.csv 生成论文表格数据

用法：
  python make_latex_tables.py [--csv paper_results/all_metrics.csv]

输出（stdout）：
  1. Table 1（测试集特征）逐炮统计
  2. Table 2（MAE ± std）LaTeX 行
  3. Table 3（MeanRel% ± std）LaTeX 行
  4. 摘要/正文需要的关键统计（时间点数、模式分布、MAE 区间）

用于投稿前数据核对；当前值应与 manuscript_fed.tex 中表格一致。
"""
import argparse
import csv
import json
import os
import statistics as st
import sys

sys.stdout.reconfigure(encoding='utf-8')

parser = argparse.ArgumentParser()
parser.add_argument('--csv', default='paper_results/all_metrics.csv')
parser.add_argument('--base', default='paper_results',
                    help='paper_results 根目录（用于读 metadata.json）')
args = parser.parse_args()

METHOD_ORDER = [
    ('nn', 'ProfileNet'),
    ('lstm', 'LSTM'),
    ('cnn', 'CNN-1D'),
    ('cnn_deeponet', 'CNN-DeepONet'),
    ('transformer', 'Transformer'),
]
DIAGS = ['Te', 'ne', 'Ti']
DIAG_UNIT = {'Te': 'keV', 'ne': '10$^{19}$', 'Ti': 'keV'}

rows = list(csv.DictReader(open(args.csv, encoding='utf-8')))


def fnum(x, nd=3):
    return f'{x:.{nd}f}'


# =============================================================================
# Table 1: 逐炮统计
# =============================================================================
print('=' * 70)
print('TABLE 1 — 测试集特征（逐炮）')
print('=' * 70)

shot_stats = {}
for r in rows:
    s = r['shot']
    if s not in shot_stats:
        shot_stats[s] = {'times': set(), 'te_cores': [], 'h98': [],
                         'mode': {}, 'diags': set(), 'time_modes': {}}
    ss = shot_stats[s]
    ss['times'].add(r['time_dir'])
    if r['diagnostic'] == 'Te' and r['method'] == 'nn':
        ss['te_cores'].append(float(r['mtanh_core']))
    if r['h98'] not in ('', None):
        ss['h98'].append(float(r['h98']))
    ss['diags'].add(r['diagnostic'])
    # 模式按唯一时间点计数（避免 5 方法 × 3 诊断 重复累计）
    if r['time_dir'] not in ss['time_modes']:
        ss['time_modes'][r['time_dir']] = r['plasma_mode']

total_tp = 0
mode_tot = {}
for s in sorted(shot_stats, key=int):
    ss = shot_stats[s]
    n_tp = len(ss['times'])
    total_tp += n_tp
    te_range = (min(ss['te_cores']), max(ss['te_cores'])) if ss['te_cores'] else (None, None)
    h98_range = (min(ss['h98']), max(ss['h98']))
    modes = {}
    for m in ss['time_modes'].values():
        modes[m] = modes.get(m, 0) + 1
    for m, c in modes.items():
        mode_tot[m] = mode_tot.get(m, 0) + c
    print(f"shot {s}: time_points={n_tp}  Te_core_range="
          f"{te_range[0]:.2f}-{te_range[1]:.2f} keV  "
          f"modes={modes}  h98_range=({h98_range[0]:.2f}, {h98_range[1]:.2f})  "
          f"diags={sorted(ss['diags'])}")

print(f"TOTAL: time_points={total_tp}  mode_counts={mode_tot}")
print()

# =============================================================================
# Table 2/3: 按 (诊断, 模式) × 方法聚合
# =============================================================================
print('=' * 70)
print('TABLE 2 — MAE ± std（每格: mean ± std，跨该组时间点）')
print('=' * 70)


def agg(metric, diag, mode):
    out = []
    for key, label in METHOD_ORDER:
        vals = [float(r[metric]) for r in rows
                if r['diagnostic'] == diag and r['method'] == key
                and (mode == 'All' or r['plasma_mode'] == mode)]
        out.append((label, st.mean(vals), st.stdev(vals) if len(vals) > 1 else 0.0, len(vals)))
    return out


for mode in ['All', 'H', 'L']:
    for diag in DIAGS:
        cells = agg('MAE', diag, mode)
        n = cells[0][3]
        if n == 0:
            continue
        line = ' & '.join(f'{m:.3f}$\\pm${s:.3f}' for _, m, s, _ in cells)
        print(f"{diag} {mode:>3s} (n={n:2d}):  {line}")

print()
print('=' * 70)
print('TABLE 3 — MeanRel% ± std（H/L）')
print('=' * 70)
for mode in ['H', 'L']:
    for diag in DIAGS:
        cells = agg('MeanRel%', diag, mode)
        n = cells[0][3]
        if n == 0:
            continue
        line = ' & '.join(f'{m:.1f}$\\pm${s:.1f}' for _, m, s, _ in cells)
        print(f"{diag} {mode:>3s} (n={n:2d}):  {line}")

# =============================================================================
# 摘要关键统计
# =============================================================================
print()
print('=' * 70)
print('ABSTRACT STATS')
print('=' * 70)
for diag in DIAGS:
    all_mae = [st.mean([float(r['MAE']) for r in rows
                        if r['diagnostic'] == diag and r['method'] == key])
               for key, _ in METHOD_ORDER]
    n_prof = len([r for r in rows if r['diagnostic'] == diag and r['method'] == 'nn'])
    print(f"{diag}: MAE range {min(all_mae):.3f}-{max(all_mae):.3f} "
          f"({DIAG_UNIT[diag]}), unique profiles={n_prof}")
n_unique = len({(r['shot'], r['time_dir'], r['diagnostic']) for r in rows
                if r['method'] == 'nn'})
n_pred = len(rows)
print(f"unique test profiles (shot×time×diag): {n_unique}")
print(f"total NN profile predictions (rows): {n_pred} = {n_unique} × 5 methods")
