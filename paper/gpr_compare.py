# -*- coding: utf-8 -*-
"""
gpr_compare.py — GPR baseline 对比（审稿意见 M3）

对每个测试时间点的原始诊断散点：
  1. 拟合 sklearn GaussianProcessRegressor（RBF + WhiteKernel）
  2. 计算三种方法对散点的拟合误差（插值到散点 ρ 位置）：GPR / mtanh / ProfileNet
  3. 附带：ProfileNet 原始网络输出 vs 存储（PCHIP 后处理）剖面的散点误差（审稿意见 m2）

用法:
  python gpr_compare.py
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

# 无 sklearn 环境：用 numpy/scipy 实现 RBF 核 GPR（对数边际似然优化）
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve

RHO_201 = np.linspace(0, 1, 201)


class RBFGPR:
    """RBF 核 + 白噪声核的高斯过程回归（numpy/scipy 实现）。"""

    def __init__(self, n_restarts=3):
        self.n_restarts = n_restarts
        self.params = None

    def _nll(self, theta, X, y):
        s2, length, noise = np.exp(theta)  # 正参数化
        X = np.asarray(X).ravel()
        K = s2 * np.exp(-0.5 * ((X[:, None] - X[None, :]) / length) ** 2)
        K += (noise + 1e-8) * np.eye(len(X))
        try:
            c, low = cho_factor(K)
            alpha = cho_solve((c, low), y)
            nll = 0.5 * y @ alpha + np.sum(np.log(np.diag(c))) + 0.5 * len(X) * np.log(2 * np.pi)
            return nll
        except np.linalg.LinAlgError:
            return 1e10

    def fit(self, X, y):
        best = None
        for _ in range(self.n_restarts):
            theta0 = np.log([1.0, 0.3, 0.1]) + np.random.default_rng(_).normal(0, 0.5, 3)
            res = minimize(self._nll, theta0, args=(X, y), method='Nelder-Mead',
                           options={'maxiter': 400, 'xatol': 1e-4, 'fatol': 1e-4})
            if best is None or res.fun < best.fun:
                best = res
        self.params = np.exp(best.x)
        self.X_, self.y_ = np.asarray(X).ravel(), np.asarray(y).ravel()
        return self

    def predict(self, X_new):
        s2, length, noise = self.params
        X_new = np.asarray(X_new).ravel()
        K = s2 * np.exp(-0.5 * ((self.X_[:, None] - self.X_[None, :]) / length) ** 2)
        K += (noise + 1e-8) * np.eye(len(self.X_))
        Ks = s2 * np.exp(-0.5 * ((X_new[:, None] - self.X_[None, :]) / length) ** 2)
        c, low = cho_factor(K)
        return Ks @ cho_solve((c, low), self.y_)

DIAG_CFG = {
    'Te': {'unit_scale': 1e-3, 'label': 'Te (keV)'},   # eV -> keV
    'ne': {'unit_scale': 1.0,  'label': 'ne (1e19)'},
    'Ti': {'unit_scale': 1.0,  'label': 'Ti (keV)'},
}


def discover(diag):
    entries = []
    for shot_dir in sorted(os.listdir(RESULT_BASE)):
        sp = os.path.join(RESULT_BASE, shot_dir)
        if not os.path.isdir(sp):
            continue
        try:
            shot = int(shot_dir)
        except ValueError:
            continue
        for td in sorted(os.listdir(sp)):
            tp = os.path.join(sp, td)
            if not os.path.isdir(tp):
                continue
            raw = os.path.join(tp, 'raw', f'scatter_{diag}.npz')
            mt = os.path.join(tp, 'profiles', f'mtanh_{diag}.npz')
            nn = os.path.join(tp, 'profiles', f'nn_{diag}.npz')
            if os.path.exists(raw) and os.path.exists(mt) and os.path.exists(nn):
                entries.append({'shot': shot, 'time_dir': td,
                                'raw': raw, 'mtanh': mt, 'nn': nn})
    return entries


def fit_to_scatter(y_201, rho_s, y_s):
    """201-pt 拟合插值到散点位置，计算 MAE/RMSE vs 原始测量。"""
    y_interp = np.interp(rho_s, RHO_201, y_201)
    mae = np.mean(np.abs(y_interp - y_s))
    rmse = np.sqrt(np.mean((y_interp - y_s) ** 2))
    return mae, rmse


def run(diag):
    cfg = DIAG_CFG[diag]
    entries = discover(diag)
    gpr_mae, gpr_rmse, mt_mae, nn_mae, gpr_time = [], [], [], [], []
    for e in entries:
        raw = np.load(e['raw'])
        rho_s = np.asarray(raw['rho'], dtype=np.float64)
        y_s = np.asarray(raw['y'], dtype=np.float64) * cfg['unit_scale']
        ok = np.isfinite(rho_s) & np.isfinite(y_s) & (y_s > 0)
        rho_s, y_s = rho_s[ok], y_s[ok]
        if len(rho_s) < 4:
            continue

        # GPR（标准化后拟合，保证数值稳定）
        y_mean, y_std = y_s.mean(), max(y_s.std(), 1e-6)
        y_n = (y_s - y_mean) / y_std
        gp = RBFGPR(n_restarts=3)
        t0 = time.time()
        gp.fit(rho_s.reshape(-1, 1), y_n)
        gpr_time.append(time.time() - t0)
        y_201 = gp.predict(RHO_201.reshape(-1, 1)) * y_std + y_mean
        mae_g, rmse_g = fit_to_scatter(y_201, rho_s, y_s)

        y_mt = np.asarray(np.load(e['mtanh'])['y'], dtype=np.float64)
        mae_m, _ = fit_to_scatter(y_mt, rho_s, y_s)
        y_nn = np.asarray(np.load(e['nn'])['y'], dtype=np.float64)
        mae_n, _ = fit_to_scatter(y_nn, rho_s, y_s)

        gpr_mae.append(mae_g); gpr_rmse.append(rmse_g)
        mt_mae.append(mae_m); nn_mae.append(mae_n)

    def fmt(v):
        return f'{np.mean(v):.4f} ± {np.std(v):.4f}'

    print(f'\n=== {cfg["label"]} (n={len(entries)}) — MAE vs raw scatter ===')
    print(f'  GPR        : {fmt(gpr_mae)}   (RMSE {np.mean(gpr_rmse):.4f})')
    print(f'  mtanh      : {fmt(mt_mae)}')
    print(f'  ProfileNet : {fmt(nn_mae)}')
    if gpr_time:
        print(f'  GPR fit time per point: {np.mean(gpr_time)*1e3:.0f} ms (median {np.median(gpr_time)*1e3:.0f} ms)')
    # GPR vs mtanh / ProfileNet 配对差异
    from scipy import stats as st
    d_gm = np.array(gpr_mae) - np.array(mt_mae)
    d_gn = np.array(gpr_mae) - np.array(nn_mae)
    print(f'  GPR vs mtanh     : {np.mean(d_gm):+.4f} ({np.mean(d_gm)/np.mean(mt_mae)*100:+.1f}%), p={st.wilcoxon(d_gm).pvalue:.4f}')
    print(f'  GPR vs ProfileNet: {np.mean(d_gn):+.4f} ({np.mean(d_gn)/np.mean(nn_mae)*100:+.1f}%), p={st.wilcoxon(d_gn).pvalue:.4f}')


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    for d in ['Te', 'ne', 'Ti']:
        run(d)
