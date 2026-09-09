# -*- coding: utf-8 -*-
"""
eval_cnn_deeponet.py — 离线评估 CNN-DeepONet 在测试集上的 MAE

无需 MDSplus 连接：直接复用 paper_results/ 中缓存的原始散点 (raw/scatter_*.npz)
与 mtanh 剖面 (profiles/mtanh_*.npz)，运行 CNN-DeepONet 推理，计算 vs mtanh 的 MAE。

用法：
  python3 paper/eval_cnn_deeponet.py            # 全部 4 测试炮号
  python3 paper/eval_cnn_deeponet.py --shot 156400

输出：
  - paper_results/{shot}/{time_dir}/profiles/cnn_deeponet_{diag}.npz  # 新增剖面
  - 终端打印 Te/ne/Ti 的 MAE 汇总（All/H/L）
"""

import numpy as np
import os
import sys
import json
import glob
import argparse
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
# 注意：paper/ 下存在旧的 profile_nn 快照（7月6日，无 cnn_deeponet.py），
# 会让 `import profile_nn` 命中旧副本。这里让 ROOT_DIR（fit/）优先，使用根目录
# 最新的 profile_nn（含 cnn_deeponet.py）与 profile_nn_models/。
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

from metrics_utils import compute_metrics, DIAGNOSTICS

MODEL_DIR = os.path.join(ROOT_DIR, 'profile_nn_models')
RESULT_BASE = os.path.join(SCRIPT_DIR, 'paper_results')
# === 原始版本（保留参考）===
# TEST_SHOTS = [156005, 156010, 156100, 156400]

# === 修改版：扩充测试集，加入 156900（156200 因 EFIT 覆盖不足整体排除）===
TEST_SHOTS = [156005, 156010, 156100, 156400, 156900]

parser = argparse.ArgumentParser()
parser.add_argument('--shot', type=int, default=None)
parser.add_argument('--output', default=RESULT_BASE)
args = parser.parse_args()
if args.shot is not None:
    TEST_SHOTS = [args.shot]
RESULT_BASE = args.output

from profile_nn.cnn_deeponet import (
    cnn_deeponet_fit_te, cnn_deeponet_fit_ne, cnn_deeponet_fit_ti,
)


def load_npz(path):
    d = np.load(path)
    return np.asarray(d['rho'], dtype=np.float64), np.asarray(d['y'], dtype=np.float64)


def discover_time_dirs(shot):
    """返回 shot 下所有含 profiles/ 的 time_dir 列表（排序）。"""
    shot_dir = os.path.join(RESULT_BASE, str(shot))
    if not os.path.isdir(shot_dir):
        return []
    tds = []
    for name in os.listdir(shot_dir):
        if os.path.isdir(os.path.join(shot_dir, name, 'profiles')):
            tds.append(name)
    tds.sort()
    return tds


def get_mode(shot, td):
    meta_path = os.path.join(RESULT_BASE, str(shot), td, 'metadata.json')
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                return json.load(f).get('plasma_mode', '?')
        except Exception:
            pass
    return '?'


def process_time_point(shot, td):
    """对单个时间点运行 CNN-DeepONet，保存剖面，返回 metric rows。"""
    pdir = os.path.join(RESULT_BASE, str(shot), td, 'profiles')
    rdir = os.path.join(RESULT_BASE, str(shot), td, 'raw')
    mode = get_mode(shot, td)
    rows = []

    # ---- Te ----
    sc_te = os.path.join(rdir, 'scatter_Te.npz')
    mt_te = os.path.join(pdir, 'mtanh_Te.npz')
    if os.path.exists(sc_te) and os.path.exists(mt_te):
        try:
            x_raw, y_raw = load_npz(sc_te)          # eV
            x_mt, y_mt = load_npz(mt_te)            # keV
            x_nn, y_nn = cnn_deeponet_fit_te(x_raw, y_raw, model_dir=MODEL_DIR)  # keV
            np.savez_compressed(os.path.join(pdir, 'cnn_deeponet_Te.npz'),
                                rho=x_nn.astype(np.float32), y=y_nn.astype(np.float32),
                                shot=shot, time=td, method='cnn_deeponet', unit='keV')
            m = compute_metrics(y_mt, y_nn, x_mt, x_nn)
            rows.append(('Te', mode, m['MAE'], m['RMSE'], m['MaxAE'], m['MeanRel%']))
        except Exception as e:
            print(f'  [Te FAIL: {e}]', end='')

    # ---- ne ----
    sc_ne = os.path.join(rdir, 'scatter_ne.npz')
    mt_ne = os.path.join(pdir, 'mtanh_ne.npz')
    if os.path.exists(sc_ne) and os.path.exists(mt_ne):
        try:
            x_raw, y_raw = load_npz(sc_ne)          # 1e19 m^-3
            x_mt, y_mt = load_npz(mt_ne)
            x_nn, y_nn = cnn_deeponet_fit_ne(x_raw, y_raw, model_dir=MODEL_DIR)
            np.savez_compressed(os.path.join(pdir, 'cnn_deeponet_ne.npz'),
                                rho=x_nn.astype(np.float32), y=y_nn.astype(np.float32),
                                shot=shot, time=td, method='cnn_deeponet', unit='1e19 m^-3')
            m = compute_metrics(y_mt, y_nn, x_mt, x_nn)
            rows.append(('ne', mode, m['MAE'], m['RMSE'], m['MaxAE'], m['MeanRel%']))
        except Exception as e:
            print(f'  [ne FAIL: {e}]', end='')

    # ---- Ti ----
    sc_ti = os.path.join(rdir, 'scatter_Ti.npz')
    mt_ti = os.path.join(pdir, 'mtanh_Ti.npz')
    mt_te = os.path.join(pdir, 'mtanh_Te.npz')
    if os.path.exists(sc_ti) and os.path.exists(mt_ti) and os.path.exists(mt_te):
        try:
            x_raw, y_raw = load_npz(sc_ti)          # keV
            x_mt, y_mt = load_npz(mt_ti)            # keV
            # Te pedestal from mtanh Te profile (rho >= 0.88)
            x_te, y_te = load_npz(mt_te)
            ped_mask = x_te >= 0.88
            te_ped_x, te_ped_y = x_te[ped_mask], y_te[ped_mask]
            x_nn, y_nn = cnn_deeponet_fit_ti(x_raw, y_raw, te_ped_x, te_ped_y,
                                             model_dir=MODEL_DIR)
            np.savez_compressed(os.path.join(pdir, 'cnn_deeponet_Ti.npz'),
                                rho=x_nn.astype(np.float32), y=y_nn.astype(np.float32),
                                shot=shot, time=td, method='cnn_deeponet', unit='keV')
            m = compute_metrics(y_mt, y_nn, x_mt, x_nn)
            rows.append(('Ti', mode, m['MAE'], m['RMSE'], m['MaxAE'], m['MeanRel%']))
        except Exception as e:
            print(f'  [Ti FAIL: {e}]', end='')

    return rows


def main():
    all_rows = []
    print(f'Model dir: {MODEL_DIR}')
    print(f'Result base: {RESULT_BASE}')
    print(f'Shots: {TEST_SHOTS}\n')

    for shot in TEST_SHOTS:
        tds = discover_time_dirs(shot)
        print(f'Shot {shot}: {len(tds)} time points')
        for td in tds:
            rows = process_time_point(shot, td)
            all_rows.extend(rows)

    if not all_rows:
        print('ERROR: no rows produced')
        sys.exit(1)

    print(f'\n{"="*78}')
    print(f'CNN-DeepONet test-set MAE vs mtanh (mean ± std)')
    print(f'{"="*78}')
    for diag in DIAGNOSTICS:
        for mode in ['All', 'H', 'L']:
            vals = [r[2] for r in all_rows if r[0] == diag
                    and (mode == 'All' or r[1] == mode)]
            if vals:
                print(f'  {diag:3s} {mode:3s}: MAE={np.mean(vals):.4f} ± {np.std(vals):.4f}  (n={len(vals)})')
    print()

    # 详细 per-mode 计数
    print('Mode counts per diag:')
    for diag in DIAGNOSTICS:
        modes = [r[1] for r in all_rows if r[0] == diag]
        from collections import Counter
        print(f'  {diag}: {dict(Counter(modes))}')


if __name__ == '__main__':
    main()
