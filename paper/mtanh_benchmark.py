# -*- coding: utf-8 -*-
"""
mtanh_benchmark.py — mtanh 七步流程端到端计时（审稿意见 m1）

对每个测试时间点计时 mtanh 拟合流程（与 run_all_fits.py 相同的函数链）：
  Te: fitting() + robust_interp()
  ne: fitting() + robust_interp()
  Ti: fitting(te_ped) + robust_interp() + enforce_monotone_pchip()
每个点重复 3 次取中位数，报告每诊断的均值/中位数和相对 NN 的加速比。

用法:
  python mtanh_benchmark.py
"""
import glob
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
RESULT_BASE = os.path.join(SCRIPT_DIR, 'paper_results')

from fitting_mtanh import fitting
from scipy.interpolate import interp1d, PchipInterpolator

NN_TIME_MS = {'ProfileNet': 15, 'LSTM': 14, 'CNN-1D': 16,
              'CNN-DeepONet': 16, 'Transformer': 19}


def robust_interp(x, y, datatype, num_points=201):
    thresholds = {"ne": 0.1, "Te": 0.05, "Ti": 0.05}
    offsets = {"ne": 0.05, "Te": 0.05, "Ti": 0.05}
    th, off = thresholds.get(datatype, 0.0), offsets.get(datatype, 0.0)
    x, y = np.array(x), np.array(y)
    mask = y >= th
    x_clean, y_clean = x[mask], y[mask]
    if len(y_clean) == 0:
        f = interp1d(x, y + off, kind='linear', fill_value="extrapolate")
        return f(np.linspace(0, 1.0, num_points))
    f = interp1d(x_clean, y_clean, kind='linear', fill_value="extrapolate")
    y_fit = f(np.linspace(0, 1.0, num_points))
    if np.min(y_fit) < th:
        f = interp1d(x, y + off, kind='linear', fill_value="extrapolate")
        y_fit = f(np.linspace(0, 1.0, num_points))
    return y_fit


def enforce_monotone_pchip(x, y, num_points=201, decreasing=True):
    x, y = np.asarray(x), np.asarray(y)
    order = np.argsort(x)
    x, y = x[order], y[order]
    pchip = PchipInterpolator(x, -y if decreasing else y)
    y_new = pchip(np.linspace(0, 1, num_points))
    return -y_new if decreasing else y_new


def time_mtanh_te(x, y):
    x_f, y_f, _, _, _ = fitting(x, y, 'Te')
    robust_interp(x_f, y_f, 'Te')


def time_mtanh_ne(x, y):
    x_f, y_f, _, _, _ = fitting(x, y, 'Refl')
    robust_interp(x_f, y_f, 'ne')


def time_mtanh_ti(x, y, te_ped_x, te_ped_y):
    x_ti, y_ti, _, _, _ = fitting(x, y, 'Ti', te_ped_x=te_ped_x, te_ped_y=te_ped_y)
    y_mt = robust_interp(x_ti, y_ti, 'Ti')
    enforce_monotone_pchip(np.linspace(0, 1, 201), y_mt)


def bench(fn, repeats=3):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)


def collect(diag, repeats=3):
    times = []
    for f in sorted(glob.glob(os.path.join(RESULT_BASE, '*', '*', 'raw', f'scatter_{diag}.npz'))):
        raw = np.load(f)
        rho_s = np.asarray(raw['rho'], dtype=np.float64)
        y_s = np.asarray(raw['y'], dtype=np.float64)
        ok = np.isfinite(rho_s) & np.isfinite(y_s)
        rho_s, y_s = rho_s[ok], y_s[ok]
        if diag == 'Te':
            t = bench(lambda: time_mtanh_te(rho_s, y_s), repeats)
        elif diag == 'ne':
            t = bench(lambda: time_mtanh_ne(rho_s, y_s), repeats)
        else:
            # Ti 需要 Te 台基（复用该时间点存储的 mtanh_Te 剖面）
            mt_te_path = f.replace('scatter_Ti.npz', 'mtanh_Te.npz').replace('raw', 'profiles')
            te_ped_x, te_ped_y = np.array([]), np.array([])
            if os.path.exists(mt_te_path):
                d = np.load(mt_te_path)
                mask = d['rho'] >= 0.88
                te_ped_x, te_ped_y = d['rho'][mask], d['y'][mask]
            if len(te_ped_x) == 0:
                continue
            t = bench(lambda: time_mtanh_ti(rho_s, y_s, te_ped_x, te_ped_y), repeats)
        times.append(t)
    return times


if __name__ == '__main__':
    for diag, unit in [('Te', 'eV'), ('ne', '1e19'), ('Ti', 'keV')]:
        times = collect(diag)
        if not times:
            print(f'{diag}: no points')
            continue
        t = np.array(times)
        print(f'mtanh {diag}: n={len(t)}  mean={t.mean():.1f} ms  median={np.median(t):.1f} ms  '
              f'min={t.min():.1f}  max={t.max():.1f}')
        for name, nn_ms in NN_TIME_MS.items():
            if diag == 'Te':  # 加速比以 Te 为例报告
                print(f'   vs {name} ({nn_ms} ms): {t.mean()/nn_ms:.1f}x slower')
        print()
