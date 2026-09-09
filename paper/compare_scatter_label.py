# -*- coding: utf-8 -*-
"""
compare_scatter_label.py — 对比 scatter-MSE 模型 vs mtanh-label 模型

评估两种训练范式的 ProfileNet Te 在测试集上谁更接近原始 TS 测量值。

用法:
  pytorch_run paper/compare_scatter_label.py
"""

import numpy as np
import os, sys, json, csv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
MODEL_SCATTER = os.path.join(SCRIPT_DIR, '..', 'profile_nn_models_scatter', 'ProfileNet_Te.pt')
MODEL_ORIG = os.path.join(SCRIPT_DIR, 'profile_nn_models', 'ProfileNet_Te.pt')
RESULT_BASE = os.path.join(SCRIPT_DIR, 'paper_results')

import torch
from profile_nn.model import ProfileNet_Te

def load_model(path, device='cpu'):
    model = ProfileNet_Te()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model

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
            raw_path = os.path.join(td_path, 'raw', 'scatter_Te.npz')
            mt_path = os.path.join(td_path, 'profiles', 'mtanh_Te.npz')
            if os.path.exists(raw_path) and os.path.exists(mt_path):
                entries.append({
                    'shot': shot, 'time_dir': td_name,
                    'raw_npz': raw_path, 'mtanh_npz': mt_path,
                })
    return entries

def evaluate_model(model, entries, device='cpu'):
    results = []
    for e in entries:
        raw = np.load(e['raw_npz'])
        rho_s = np.asarray(raw['rho'], dtype=np.float32)
        val_s = np.asarray(raw['y'], dtype=np.float32) / 1000.0  # eV → keV

        mt = np.load(e['mtanh_npz'])
        y_mtanh = np.asarray(mt['y'], dtype=np.float64)

        # NN 推理
        X_in = np.stack([rho_s, val_s], axis=-1)
        X_t = torch.from_numpy(X_in).unsqueeze(0).to(device)
        mask_t = torch.ones(1, X_in.shape[0], dtype=torch.bool).to(device)

        with torch.no_grad():
            y_pred = model(X_t, mask_t).cpu().numpy()[0]

        if np.isnan(y_pred).any():
            continue

        # 201-pt 网格
        rho_201 = np.linspace(0, 1, 201)

        # 指标 1: 插值到散点位置 → MSE vs 原始 TS 测量
        y_interp = np.interp(rho_s, rho_201, y_pred)
        mae_scatter = np.mean(np.abs(y_interp - val_s))
        rmse_scatter = np.sqrt(np.mean((y_interp - val_s)**2))

        mtanh_interp = np.interp(rho_s, rho_201, y_mtanh)
        mae_mtanh_scatter = np.mean(np.abs(mtanh_interp - val_s))
        rmse_mtanh_scatter = np.sqrt(np.mean((mtanh_interp - val_s)**2))

        # 指标 2: 201-pt MSE vs mtanh label
        mae_vs_mtanh = np.mean(np.abs(y_pred - y_mtanh))
        rmse_vs_mtanh = np.sqrt(np.mean((y_pred - y_mtanh)**2))

        results.append({
            'shot': e['shot'], 'time_dir': e['time_dir'],
            'mae_scatter': mae_scatter, 'rmse_scatter': rmse_scatter,
            'mae_mtanh_scatter': mae_mtanh_scatter, 'rmse_mtanh_scatter': rmse_mtanh_scatter,
            'mae_vs_mtanh': mae_vs_mtanh, 'rmse_vs_mtanh': rmse_vs_mtanh,
        })
    return results

def main():
    print('Loading models...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_scatter = load_model(MODEL_SCATTER, device)
    model_orig = load_model(MODEL_ORIG, device)
    print(f'  Scatter-MSE model loaded (device={device})')
    print(f'  Original model loaded')

    entries = discover_test_points()
    print(f'  {len(entries)} Te test points\n')

    print('Evaluating scatter-MSE model...')
    r_scatter = evaluate_model(model_scatter, entries, device)
    print('Evaluating original (mtanh-label) model...')
    r_orig = evaluate_model(model_orig, entries, device)

    # 汇总
    for name, results in [('Scatter-MSE', r_scatter), ('Mtanh-label', r_orig)]:
        if not results:
            continue
        mae_s = np.mean([r['mae_scatter'] for r in results])
        rmse_s = np.mean([r['rmse_scatter'] for r in results])
        mae_m = np.mean([r['mae_vs_mtanh'] for r in results])

        # mtanh baseline scatter error
        mae_mtanh_s = np.mean([r['mae_mtanh_scatter'] for r in results])
        rmse_mtanh_s = np.mean([r['rmse_mtanh_scatter'] for r in results])

        print(f'\n{"="*60}')
        print(f'{name} Model (n={len(results)})')
        print(f'{"="*60}')
        print(f'  vs TS scatter:  MAE={mae_s:.4f} keV  RMSE={rmse_s:.4f} keV')
        print(f'  vs mtanh label: MAE={mae_m:.4f} keV')
        print(f'  mtanh vs TS scatter: MAE={mae_mtanh_s:.4f} keV  RMSE={rmse_mtanh_s:.4f} keV')
        print(f'  Δ vs mtanh baseline: {mae_s - mae_mtanh_s:+.4f} keV ({(mae_s - mae_mtanh_s)/mae_mtanh_s*100:+.1f}%)')

    # 保存 CSV
    csv_path = os.path.join(RESULT_BASE, 'scatter_vs_label_comparison.csv')
    all_rows = []
    for r_s, r_o in zip(r_scatter, r_orig):
        all_rows.append({
            'shot': r_s['shot'], 'time_dir': r_s['time_dir'],
            'scatter_mae': r_s['mae_scatter'], 'scatter_rmse': r_s['rmse_scatter'],
            'orig_mae': r_o['mae_scatter'], 'orig_rmse': r_o['rmse_scatter'],
            'mtanh_mae': r_s['mae_mtanh_scatter'], 'mtanh_rmse': r_s['rmse_mtanh_scatter'],
            'scatter_delta%': (r_s['mae_scatter'] - r_o['mae_scatter']) / max(r_o['mae_scatter'], 1e-10) * 100,
        })
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        w.writeheader()
        w.writerows(all_rows)
    print(f'\nCSV: {csv_path}')


if __name__ == '__main__':
    main()
