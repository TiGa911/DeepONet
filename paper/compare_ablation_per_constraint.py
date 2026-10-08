# -*- coding: utf-8 -*-
"""
compare_ablation_per_constraint.py — 逐约束消融评估（审稿意见 M4）

对训练完成的 profile_nn_models_ablation_pc/{config}/{Arch}_Te.pt 做 33 点测试集推理，
计算各配置相对 physics-on 基线的 Te MAE 与 Δ%，输出汇总表、CSV 和对比图。

用法:
  python compare_ablation_per_constraint.py   # 需先完成 train_per_constraint_ablation.py

输出:
  paper_results/per_constraint_ablation.csv
  paper_results/figures/fig_ablation_per_constraint.png
"""
import csv
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from compare_ablation import (
    discover_test_points, run_inference_phase, TMP_DIR,
    NN_METHODS, METHOD_SPECS, compute_metrics,
)

ROOT = os.path.dirname(SCRIPT_DIR)
PC_BASE = os.path.join(ROOT, 'profile_nn_models_ablation_pc')
ORIG_DIR = os.path.join(SCRIPT_DIR, 'profile_nn_models')
ALLOFF_DIR = os.path.join(SCRIPT_DIR, 'profile_nn_models_ablation')
RESULT_BASE = os.path.join(SCRIPT_DIR, 'paper_results')
FIG_DIR = os.path.join(RESULT_BASE, 'figures')

CONFIGS = ['mono_off', 'bdy_off', 'smooth_off', 'log_off']
REQUIRED_FILES = ['ProfileNet_Te.pt', 'LSTM_Te.pt', 'CNN_Te.pt', 'Transformer_Te.pt']


def main():
    entries = discover_test_points()
    print(f'{len(entries)} Te test points')

    dirs = {'on': ORIG_DIR, 'all_off': ALLOFF_DIR}
    for c in CONFIGS:
        dirs[c] = os.path.join(PC_BASE, c)

    saved = {}
    for label, d in dirs.items():
        if not all(os.path.exists(os.path.join(d, f)) for f in REQUIRED_FILES):
            print(f'  skip {label}: models missing in {d}')
            continue
        saved[label] = run_inference_phase(d, label, entries)

    # ---- 逐点 MAE vs mtanh ----
    rho_201 = np.linspace(0, 1, 201)
    rows = []
    for label, save_dir in saved.items():
        for entry in entries:
            key = f'{entry["shot"]}_{entry["time_dir"]}'
            mt = np.load(entry['mtanh_npz'])['y']
            for method in NN_METHODS:
                p = os.path.join(save_dir, f'{key}_{method}_Te.npz')
                if not os.path.exists(p):
                    continue
                y = np.load(p)['y']
                mae = compute_metrics(mt, y, rho_201, rho_201)['MAE']
                rows.append({'config': label, 'method': method, 'key': key, 'MAE': mae})

    csv_path = os.path.join(RESULT_BASE, 'per_constraint_ablation.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['config', 'method', 'key', 'MAE'])
        w.writeheader()
        w.writerows(rows)
    print(f'CSV: {csv_path} ({len(rows)} rows)')

    # ---- 汇总 ----
    def mean_of(label, m):
        vs = [r['MAE'] for r in rows if r['config'] == label and r['method'] == m]
        return np.mean(vs) if vs else None

    on = {m: mean_of('on', m) for m in NN_METHODS}
    labels = ['on'] + CONFIGS + ['all_off']
    print(f'\n{"config":>10s} | ' + ' | '.join(f'{METHOD_SPECS[m]["label"]:>14s}' for m in NN_METHODS))
    print('-' * 100)
    for label in labels:
        if label not in saved:
            continue
        cells = []
        for m in NN_METHODS:
            mu = mean_of(label, m)
            if mu is None:
                cells.append(f'{"--":>14s}')
            else:
                d = (mu - on[m]) / on[m] * 100
                cells.append(f'{mu:>8.3f}{d:>+6.0f}%')
        print(f'{label:>10s} | ' + ' | '.join(cells))

    # ---- 图：各约束的 Δ% 按架构分组（仅含 4 个有对照模型的架构）----
    plot_methods = [m for m in NN_METHODS if on[m] is not None]
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    x = np.arange(len(plot_methods))
    width = 0.18
    colors = {'mono_off': '#E63946', 'bdy_off': '#2A9D8F',
              'smooth_off': '#457B9D', 'log_off': '#F4A261', 'all_off': '#555555'}
    for i, label in enumerate(CONFIGS + ['all_off']):
        deltas = [((mean_of(label, m) or 0) - on[m]) / on[m] * 100 for m in plot_methods]
        ax.bar(x + (i - 2) * width, deltas, width, label=label,
               color=colors[label], alpha=0.9)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_SPECS[m]['label'] for m in plot_methods])
    ax.set_ylabel('Te MAE change vs physics-on (%)')
    ax.set_title('Per-constraint ablation (Te): removing one physics term at a time')
    ax.legend(ncol=5, fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig_ablation_per_constraint.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Figure: {fig_path}')


if __name__ == '__main__':
    main()
