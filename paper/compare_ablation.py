# -*- coding: utf-8 -*-
"""
compare_ablation.py — 物理约束消融对比

分两阶段：先用消融模型推理，再用原始模型推理，最后对比 MAE vs mtanh。

用法:
  pytorch_run paper/compare_ablation.py          # 服务器

输出:
  paper_results/all_ablation_metrics.csv
  paper_results/figures/fig7_ablation.png
"""

import numpy as np
import os
import sys
import json
import csv
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

MODEL_DIR_ORIG = os.path.join(SCRIPT_DIR, 'profile_nn_models')
MODEL_DIR_ABL = os.path.join(SCRIPT_DIR, 'profile_nn_models_ablation')
RESULT_BASE = os.path.join(SCRIPT_DIR, 'paper_results')
FIG_DIR = os.path.join(RESULT_BASE, 'figures')
TMP_DIR = os.path.join(RESULT_BASE, '_ablation_tmp')
os.makedirs(FIG_DIR, exist_ok=True)

from metrics_utils import (
    METHOD_SPECS, NN_METHODS, compute_metrics, compute_peakedness,
)


def discover_test_points():
    entries = []
    for shot_name in sorted(os.listdir(RESULT_BASE)):
        shot_path = os.path.join(RESULT_BASE, shot_name)
        if not os.path.isdir(shot_path): continue
        try: shot = int(shot_name)
        except ValueError: continue
        for td_name in sorted(os.listdir(shot_path)):
            td_path = os.path.join(shot_path, td_name)
            if not os.path.isdir(td_path): continue
            mt_path = os.path.join(td_path, 'profiles', 'mtanh_Te.npz')
            raw_path = os.path.join(td_path, 'raw', 'scatter_Te.npz')
            meta_path = os.path.join(td_path, 'metadata.json')
            if not os.path.exists(mt_path) or not os.path.exists(raw_path): continue
            meta = {}
            if os.path.exists(meta_path):
                try:
                    with open(meta_path) as f: meta = json.load(f)
                except Exception: pass
            entries.append({
                'shot': shot, 'time_dir': td_name,
                'mtanh_npz': mt_path, 'raw_npz': raw_path,
                'plasma_mode': meta.get('plasma_mode', '?'),
            })
    return entries


def run_inference_phase(model_dir, phase_label, entries):
    """加载 model_dir 下的模型，对 entries 做 Te 推理，保存到 TMP_DIR/{label}/。
    因为 set_model_dir 是全局状态，一次只能用一个 dir。"""
    import importlib

    # 强制重载所有 infer 模块以清除缓存
    modules_to_reload = []
    for mod_name in ['profile_nn.infer', 'profile_nn.infer_lstm',
                     'profile_nn.infer_cnn', 'profile_nn.infer_transformer']:
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        modules_to_reload.append(mod_name)

    # 加载模型
    fitters = {}

    try:
        from profile_nn.infer import nn_fit_te, set_model_dir
        set_model_dir(model_dir)
        fitters['nn'] = nn_fit_te
        print(f'  ProfileNet OK')
    except Exception as e:
        print(f'  ProfileNet: {e}')

    try:
        from profile_nn.infer_lstm import nn_fit_te_lstm, set_model_dir_lstm
        set_model_dir_lstm(model_dir)
        fitters['lstm'] = nn_fit_te_lstm
        print(f'  LSTM OK')
    except Exception as e:
        print(f'  LSTM: {e}')

    try:
        from profile_nn.infer_cnn import nn_fit_te_cnn, set_model_dir_cnn
        set_model_dir_cnn(model_dir)
        fitters['cnn'] = nn_fit_te_cnn
        print(f'  CNN-1D OK')
    except Exception as e:
        print(f'  CNN: {e}')

    try:
        from profile_nn.infer_transformer import nn_fit_te_transformer, set_model_dir_transformer
        set_model_dir_transformer(model_dir)
        fitters['transformer'] = nn_fit_te_transformer
        print(f'  Transformer OK')
    except Exception as e:
        print(f'  Transformer: {e}')

    # 推理并保存
    save_dir = os.path.join(TMP_DIR, phase_label)
    os.makedirs(save_dir, exist_ok=True)
    ok = 0

    for i, entry in enumerate(entries):
        key = f'{entry["shot"]}_{entry["time_dir"]}'
        raw = np.load(entry['raw_npz'])
        x_raw = np.asarray(raw['rho'], dtype=np.float64)
        y_raw = np.asarray(raw['y'], dtype=np.float64)

        for method in NN_METHODS:
            if method not in fitters:
                continue
            try:
                _, y_pred = fitters[method](x_raw, y_raw)
                npz_path = os.path.join(save_dir, f'{key}_{method}_Te.npz')
                np.savez(npz_path, y=np.asarray(y_pred, dtype=np.float64))
                ok += 1
            except Exception as e:
                print(f'  {key} {method}: {e}')

        if (i+1) % 5 == 0:
            print(f'  {i+1}/{len(entries)} done')

    print(f'  Phase "{phase_label}": {ok} profiles saved')
    return save_dir


def main():
    entries = discover_test_points()
    print(f'Found {len(entries)} Te test points\n')

    # ---- Phase 1: 消融模型 ----
    print('Phase 1: Ablation models (MSE-only) ---')
    abl_dir = run_inference_phase(MODEL_DIR_ABL, 'ablation', entries)

    # ---- Phase 2: 原始模型 ----
    print('\nPhase 2: Original models (physics-constrained) ---')
    orig_dir = run_inference_phase(MODEL_DIR_ORIG, 'original', entries)

    # ---- Phase 3: 对比 ----
    print('\nPhase 3: Comparison ---')
    all_rows = []

    for entry in entries:
        key = f'{entry["shot"]}_{entry["time_dir"]}'
        mt_data = np.load(entry['mtanh_npz'])
        y_ref = np.asarray(mt_data['y'], dtype=np.float64)
        rho_201 = np.linspace(0, 1, 201)

        for method in NN_METHODS:
            abl_path = os.path.join(abl_dir, f'{key}_{method}_Te.npz')
            orig_path = os.path.join(orig_dir, f'{key}_{method}_Te.npz')
            if not os.path.exists(abl_path) or not os.path.exists(orig_path):
                continue

            abl_data = np.load(abl_path)
            orig_data = np.load(orig_path)
            y_abl = np.asarray(abl_data['y'], dtype=np.float64)
            y_orig = np.asarray(orig_data['y'], dtype=np.float64)

            if np.isnan(y_abl).any() or np.isnan(y_orig).any():
                continue

            abl_m = compute_metrics(y_ref, y_abl, rho_201, rho_201)
            orig_m = compute_metrics(y_ref, y_orig, rho_201, rho_201)

            all_rows.append({
                'shot': entry['shot'], 'time_dir': entry['time_dir'],
                'plasma_mode': entry['plasma_mode'], 'method': method,
                'MAE_physics_on': orig_m['MAE'], 'RMSE_physics_on': orig_m['RMSE'],
                'MAE_physics_off': abl_m['MAE'], 'RMSE_physics_off': abl_m['RMSE'],
                'MAE_diff': abl_m['MAE'] - orig_m['MAE'],
                'RMSE_diff': abl_m['RMSE'] - orig_m['RMSE'],
                'MAE_degrade%': (abl_m['MAE'] - orig_m['MAE']) / max(orig_m['MAE'], 1e-10) * 100,
            })

    if not all_rows:
        print('ERROR: No comparison data')
        return

    # ---- 保存 CSV ----
    csv_path = os.path.join(RESULT_BASE, 'all_ablation_metrics.csv')
    fieldnames = ['shot', 'time_dir', 'plasma_mode', 'method',
                  'MAE_physics_on', 'RMSE_physics_on',
                  'MAE_physics_off', 'RMSE_physics_off',
                  'MAE_diff', 'RMSE_diff', 'MAE_degrade%']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)
    print(f'\nCSV: {csv_path} ({len(all_rows)} rows)')

    # ---- 打印汇总 ----
    print(f'\n{"Method":15s} {"MAE(on)":>10s} {"MAE(off)":>10s} {"Δ%":>8s}')
    print('-' * 45)
    for method in NN_METHODS:
        rows_m = [r for r in all_rows if r['method'] == method]
        if not rows_m:
            continue
        mae_on = np.mean([r['MAE_physics_on'] for r in rows_m])
        mae_off = np.mean([r['MAE_physics_off'] for r in rows_m])
        degrade = np.mean([r['MAE_degrade%'] for r in rows_m])
        print(f'{METHOD_SPECS[method]["label"]:15s} {mae_on:10.4f} {mae_off:10.4f} {degrade:+8.1f}%')

    # ---- 生成 fig7 ----
    generate_fig7(all_rows)

    # 清理临时文件
    import shutil
    shutil.rmtree(TMP_DIR, ignore_errors=True)


def generate_fig7(all_rows):
    methods = NN_METHODS
    labels = [METHOD_SPECS[m]['label'] for m in methods]
    colors = [METHOD_SPECS[m]['color'] for m in methods]
    x = np.arange(len(methods))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Panel 1: MAE bar chart
    mae_on, mae_off = [], []
    mae_on_std, mae_off_std = [], []
    for method in methods:
        rows_m = [r for r in all_rows if r['method'] == method]
        mae_on.append(np.mean([r['MAE_physics_on'] for r in rows_m]))
        mae_off.append(np.mean([r['MAE_physics_off'] for r in rows_m]))
        mae_on_std.append(np.std([r['MAE_physics_on'] for r in rows_m]))
        mae_off_std.append(np.std([r['MAE_physics_off'] for r in rows_m]))

    ax1.bar(x - width/2, mae_on, width, label='Physics-constrained', color='#2E86AB', alpha=0.85)
    ax1.bar(x + width/2, mae_off, width, label='Pure MSE (ablation)', color='#A23B72', alpha=0.85)
    ax1.errorbar(x - width/2, mae_on, yerr=mae_on_std, fmt='none', ecolor='black', capsize=3)
    ax1.errorbar(x + width/2, mae_off, yerr=mae_off_std, fmt='none', ecolor='black', capsize=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('MAE (keV)')
    ax1.set_title('Te MAE: Physics-constrained vs Pure MSE')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3, axis='y')

    # Panel 2: degradation scatter
    ax2.axhline(y=0, color='gray', ls='-', lw=0.5, zorder=1)
    for i, method in enumerate(methods):
        rows_m = [r for r in all_rows if r['method'] == method]
        degrade_vals = [r['MAE_degrade%'] for r in rows_m]
        jitter = np.random.RandomState(i).normal(0, 0.08, len(degrade_vals))
        ax2.scatter(np.full(len(degrade_vals), i) + jitter, degrade_vals,
                    c=colors[i], alpha=0.6, s=30, zorder=3, label=labels[i])
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('MAE Degradation (%)')
    ax2.set_title('Per-Profile MAE Degradation (no physics constraints)')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis='y')

    fig.suptitle('Physics Constraint Ablation — Te Profile Fitting',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(FIG_DIR, 'fig7_ablation.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'fig7_ablation.png saved')


if __name__ == '__main__':
    main()
