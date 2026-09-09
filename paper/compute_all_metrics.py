# -*- coding: utf-8 -*-
"""
compute_all_metrics.py — 从 NPZ 读取所有剖面，计算 NN vs mtanh 指标

读取 run_all_fits.py 生成的 paper_results/{shot}/{time_dir}/profiles/*.npz，
以 mtanh 为参考，对 4 个 NN 方法计算 MAE/RMSE/MaxAE/peakedness 等指标。

用法：
  python3 paper/compute_all_metrics.py
  python3 paper/compute_all_metrics.py --output ./paper_results

输出：
  paper_results/all_metrics.csv   # 所有时间点×诊断×方法的指标
  paper_results/all_metrics.json  # 同上，JSON 格式
"""

import numpy as np
import os
import sys
import json
import csv
import argparse
import glob as _glob
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from metrics_utils import (
    compute_metrics, compute_peakedness, compute_core_bias,
    NN_METHODS, DIAGNOSTICS, TEST_SHOTS,
)

parser = argparse.ArgumentParser(description='Compute NN vs mtanh metrics from NPZ')
parser.add_argument('--output', default=os.path.join(SCRIPT_DIR, 'paper_results'),
                    help='paper_results directory (default: paper/paper_results)')
args = parser.parse_args()
RESULT_BASE = args.output


# ================================================================
# 扫描
# ================================================================

def discover_profiles():
    """扫描 paper_results/ 下所有 time_dir，返回 (shot, time_dir, time_val) 列表。"""
    entries = []
    for shot_dir_name in os.listdir(RESULT_BASE):
        shot_path = os.path.join(RESULT_BASE, shot_dir_name)
        if not os.path.isdir(shot_path):
            continue
        try:
            shot = int(shot_dir_name)
        except ValueError:
            continue

        for td_name in os.listdir(shot_path):
            td_path = os.path.join(shot_path, td_name)
            if not os.path.isdir(td_path):
                continue
            profiles_dir = os.path.join(td_path, 'profiles')
            if not os.path.isdir(profiles_dir):
                continue

            # 解析时间
            try:
                time_str = td_name.replace('s', '')
                time_val = float(time_str)
            except ValueError:
                continue

            # 读 metadata
            meta = {}
            meta_path = os.path.join(td_path, 'metadata.json')
            if os.path.exists(meta_path):
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                except Exception:
                    pass

            entries.append({
                'shot': shot,
                'time_dir': td_name,
                'time_val': time_val,
                'profiles_dir': profiles_dir,
                'plasma_mode': meta.get('plasma_mode', '?'),
                'h98': meta.get('h98'),
                'real_time': meta.get('real_time'),
            })

    entries.sort(key=lambda e: (e['shot'], e['time_val']))
    return entries


# ================================================================
# 主流程
# ================================================================

def main():
    entries = discover_profiles()
    print(f'Found {len(entries)} time points across paper_results/')
    if not entries:
        print('ERROR: No profile data found. Run run_all_fits.py first.')
        sys.exit(1)

    all_rows = []
    errors = 0

    for entry in entries:
        diag_profiles = {}
        for diag in DIAGNOSTICS:
            diag_profiles[diag] = {}

        # 加载所有可用方法的剖面
        for method in ['mtanh'] + NN_METHODS:
            for diag in DIAGNOSTICS:
                npz_path = os.path.join(entry['profiles_dir'], f'{method}_{diag}.npz')
                if os.path.exists(npz_path):
                    try:
                        data = np.load(npz_path)
                        diag_profiles[diag][method] = {
                            'rho': data['rho'],
                            'y': data['y'],
                        }
                    except Exception:
                        pass

        # 对每个诊断，以 mtanh 为参考计算 NN 指标
        for diag in DIAGNOSTICS:
            mtanh_data = diag_profiles[diag].get('mtanh')
            if mtanh_data is None:
                continue

            y_ref = np.asarray(mtanh_data['y'], dtype=np.float64)
            rho_ref = np.asarray(mtanh_data['rho'], dtype=np.float64)
            ref_peakedness = compute_peakedness(y_ref)
            ref_core = float(y_ref[0])

            for method in NN_METHODS:
                nn_data = diag_profiles[diag].get(method)
                if nn_data is None:
                    continue

                y_pred = np.asarray(nn_data['y'], dtype=np.float64)
                rho_pred = np.asarray(nn_data['rho'], dtype=np.float64)

                try:
                    m = compute_metrics(y_ref, y_pred, rho_ref, rho_pred)
                    nn_peakedness = compute_peakedness(y_pred)
                    nn_core = float(y_pred[0])
                    core_bias = (nn_core - ref_core) / max(abs(ref_core), 1e-10) * 100
                    peakedness_bias = (
                        (nn_peakedness - ref_peakedness) / max(abs(ref_peakedness), 1e-10) * 100
                    )
                except Exception as e:
                    errors += 1
                    continue

                row = {
                    'shot': entry['shot'],
                    'time_dir': entry['time_dir'],
                    'time_val': entry['time_val'],
                    'real_time': entry.get('real_time'),
                    'plasma_mode': entry['plasma_mode'],
                    'h98': entry['h98'],
                    'diagnostic': diag,
                    'method': method,
                    'MAE': m['MAE'],
                    'RMSE': m['RMSE'],
                    'MaxAE': m['MaxAE'],
                    'MeanRel%': m['MeanRel%'],
                    'MaxRel%': m['MaxRel%'],
                    'mtanh_core': ref_core,
                    'nn_core': nn_core,
                    'core_bias%': core_bias,
                    'mtanh_peakedness': ref_peakedness,
                    'nn_peakedness': nn_peakedness,
                    'peakedness_bias%': peakedness_bias,
                }
                all_rows.append(row)

        if all_rows:
            last = all_rows[-1]
            print(f'  {entry["shot"]}/{entry["time_dir"]}: '
                  f'{diag}×{sum(1 for r in all_rows if r["shot"]==entry["shot"] and r["time_dir"]==entry["time_dir"])} rows')

    print(f'\nTotal: {len(all_rows)} metric rows ({errors} errors)')

    if not all_rows:
        print('ERROR: No metric rows generated. Check NPZ files.')
        sys.exit(1)

    # ---- 汇总统计 ----
    print(f'\n{"="*70}')
    print(f'Summary by diagnostic+method (mean MAE):')
    for diag in DIAGNOSTICS:
        for method in NN_METHODS:
            vals = [r['MAE'] for r in all_rows if r['diagnostic'] == diag and r['method'] == method]
            if vals:
                print(f'  {diag:4s} {method:12s}: '
                      f'MAE={np.mean(vals):.4f} ± {np.std(vals):.4f}  (n={len(vals)})')

    # ---- 保存 CSV ----
    csv_path = os.path.join(RESULT_BASE, 'all_metrics.csv')
    fieldnames = [
        'shot', 'time_dir', 'time_val', 'real_time', 'plasma_mode', 'h98',
        'diagnostic', 'method',
        'MAE', 'RMSE', 'MaxAE', 'MeanRel%', 'MaxRel%',
        'mtanh_core', 'nn_core', 'core_bias%',
        'mtanh_peakedness', 'nn_peakedness', 'peakedness_bias%',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)
    print(f'\nCSV saved: {csv_path}')

    # ---- 保存 JSON ----
    json_path = os.path.join(RESULT_BASE, 'all_metrics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=False)
    print(f'JSON saved: {json_path}')

    print(f'\n{"="*70}')
    print(f'Done.')

    # ---- H/L 模式分报告 ----
    print(f'\n{"="*70}')
    print(f'H-mode vs L-mode breakdown:')
    h_rows = [r for r in all_rows if r.get('plasma_mode') == 'H']
    l_rows = [r for r in all_rows if r.get('plasma_mode') == 'L']

    for label, rows in [('H-mode (H98>=0.7)', h_rows), ('L-mode (H98<0.7)', l_rows)]:
        if not rows:
            print(f'  {label}: 0 samples')
            continue
        print(f'\n  {label}: {len(rows)} rows')

        # 按 diagnostic+method 聚合
        diags_in_mode = sorted(set(r['diagnostic'] for r in rows))
        methods_in_mode = sorted(set(r['method'] for r in rows))
        for diag in diags_in_mode:
            for method in methods_in_mode:
                vals = [r['MAE'] for r in rows
                        if r['diagnostic'] == diag and r['method'] == method]
                if vals:
                    print(f'    {diag:4s} {method:12s}: '
                          f'MAE={np.mean(vals):.4f} ± {np.std(vals):.4f}  (n={len(vals)})')

    # 保存 H/L 分报告 CSV
    for label, rows, suffix in [('H', h_rows, 'H_mode'), ('L', l_rows, 'L_mode')]:
        if rows:
            csv_path_mode = os.path.join(RESULT_BASE, f'all_metrics_{suffix}.csv')
            with open(csv_path_mode, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(rows)
            print(f'\n{suffix} CSV saved: {csv_path_mode}')


if __name__ == '__main__':
    main()
