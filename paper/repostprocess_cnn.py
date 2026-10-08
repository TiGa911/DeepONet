# -*- coding: utf-8 -*-
"""
repostprocess_cnn.py — CNN-1D 剖面离线重新后处理（无需 MDSplus）

对全部测试时间点，用修改后的 nn_fit_te/ne/ti_cnn（少节点单调 PCHIP 平滑）
重新推理并覆盖 profiles/cnn_{diag}.npz。

用法:
  python repostprocess_cnn.py
"""
import glob
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
RESULT_BASE = os.path.join(SCRIPT_DIR, 'paper_results')

from profile_nn.infer_cnn import nn_fit_te_cnn, nn_fit_ne_cnn, nn_fit_ti_cnn, set_model_dir_cnn

set_model_dir_cnn(os.path.join(SCRIPT_DIR, 'profile_nn_models'))

# 与 run_all_fits.py 相同的后处理工具（供对比参考）
from scipy.interpolate import PchipInterpolator


def enforce_monotone_pchip(x, y, num_points=201, decreasing=True):
    x, y = np.asarray(x), np.asarray(y)
    order = np.argsort(x)
    x, y = x[order], y[order]
    pchip = PchipInterpolator(x, -y if decreasing else y)
    y_new = pchip(np.linspace(0, 1, num_points))
    return -y_new if decreasing else y_new


def main():
    n_te = n_ne = n_ti = 0
    for raw_te in sorted(glob.glob(os.path.join(RESULT_BASE, '*', '*', 'raw', 'scatter_Te.npz'))):
        td_dir = os.path.dirname(os.path.dirname(raw_te))
        # ---- Te ----
        d = np.load(raw_te)
        rho_s, y_s = d['rho'], d['y']
        ok = np.isfinite(rho_s) & np.isfinite(y_s)
        try:
            x, y = nn_fit_te_cnn(rho_s[ok], y_s[ok])
            np.savez_compressed(os.path.join(td_dir, 'profiles', 'cnn_Te.npz'),
                                rho=x.astype(np.float32), y=y.astype(np.float32))
            n_te += 1
        except Exception as e:
            print(f'{td_dir} Te FAIL: {e}')

        # ---- ne ----
        raw_ne = os.path.join(td_dir, 'raw', 'scatter_ne.npz')
        if os.path.exists(raw_ne):
            d = np.load(raw_ne)
            rho_s, y_s = d['rho'], d['y']
            ok = np.isfinite(rho_s) & np.isfinite(y_s)
            try:
                x, y = nn_fit_ne_cnn(rho_s[ok], y_s[ok])
                np.savez_compressed(os.path.join(td_dir, 'profiles', 'cnn_ne.npz'),
                                    rho=x.astype(np.float32), y=y.astype(np.float32))
                n_ne += 1
            except Exception as e:
                print(f'{td_dir} ne FAIL: {e}')

        # ---- Ti（需要 te_ped：mtanh Te 的 ρ≥0.88 段）----
        raw_ti = os.path.join(td_dir, 'raw', 'scatter_Ti.npz')
        mt_te = os.path.join(td_dir, 'profiles', 'mtanh_Te.npz')
        if os.path.exists(raw_ti) and os.path.exists(mt_te):
            d = np.load(raw_ti)
            rho_s, y_s = d['rho'], d['y']
            ok = np.isfinite(rho_s) & np.isfinite(y_s)
            mt = np.load(mt_te)
            mask = mt['rho'] >= 0.88
            te_ped_x, te_ped_y = mt['rho'][mask], mt['y'][mask]
            try:
                x, y = nn_fit_ti_cnn(rho_s[ok], y_s[ok], te_ped_x, te_ped_y)
                np.savez_compressed(os.path.join(td_dir, 'profiles', 'cnn_Ti.npz'),
                                    rho=x.astype(np.float32), y=y.astype(np.float32))
                n_ti += 1
            except Exception as e:
                print(f'{td_dir} Ti FAIL: {e}')

    print(f'Done: Te={n_te} ne={n_ne} Ti={n_ti} CNN profiles reprocessed')


if __name__ == '__main__':
    main()
