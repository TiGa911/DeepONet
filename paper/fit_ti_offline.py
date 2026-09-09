# -*- coding: utf-8 -*-
"""
fit_ti_offline.py — 离线 mtanh 基线拟合 Ti 剖面

读取离线 Ti 散点数据（NPZ 格式），用 mtanh 拟合并绘制结果。

NPZ 格式要求:
  rho: (N,) 归一化半径
  y:   (N,) Ti [keV]

用法:
  python paper/fit_ti_offline.py <npz_file> [--output OUTPUT_DIR]

示例:
  python paper/fit_ti_offline.py paper/ti_override/156005_004.01790s_Ti.npz
  python paper/fit_ti_offline.py data.npz --output ./my_results
"""

import numpy as np
import os
import sys
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from fitting_mtanh import fitting
from scipy.interpolate import interp1d, PchipInterpolator

# ================================================================
# 命令行
# ================================================================

parser = argparse.ArgumentParser(description='Offline mtanh Ti profile fitting')
parser.add_argument('npz_file', help='Path to .npz file with rho and y arrays')
parser.add_argument('--output', '-o', default='./ti_fit_output',
                    help='Output directory (default: ./ti_fit_output)')
parser.add_argument('--te-ped-npz', default=None,
                    help='Optional Te pedestal NPZ (rho_te, y_te_keV) for Ti boundary constraint')
args = parser.parse_args()

# ================================================================
# 加载数据
# ================================================================

data = np.load(args.npz_file)
x_raw = np.asarray(data['rho'], dtype=np.float64).flatten()
y_raw = np.asarray(data['y'], dtype=np.float64).flatten()

# 过滤 NaN
valid = ~np.isnan(x_raw) & ~np.isnan(y_raw)
if not valid.all():
    print(f'Dropped {sum(~valid)} NaN points')
    x_raw = x_raw[valid]
    y_raw = y_raw[valid]

# 按 ρ 排序
order = np.argsort(x_raw)
x_raw, y_raw = x_raw[order], y_raw[order]

print(f'Loaded: {len(x_raw)} Ti points')
print(f'  rho: [{x_raw.min():.4f}, {x_raw.max():.4f}]')
print(f'  Ti:  [{y_raw.min():.4f}, {y_raw.max():.4f}] keV')

# 可选的 Te 台基数据（用于约束 Ti 边界）
te_ped_x, te_ped_y = np.array([]), np.array([])
if args.te_ped_npz and os.path.exists(args.te_ped_npz):
    te_data = np.load(args.te_ped_npz)
    te_x = np.asarray(te_data['rho'], dtype=np.float64).flatten()
    te_y = np.asarray(te_data['y'], dtype=np.float64).flatten()
    mask = te_x >= 0.88
    te_ped_x, te_ped_y = te_x[mask], te_y[mask]
    print(f'Te pedestal: {len(te_ped_x)} pts (rho >= 0.88)')

# ================================================================
# mtanh 拟合
# ================================================================

print('\nFitting...')

def robust_interp(x, y, datatype, num_points=201):
    """插值到 201 均匀点。"""
    x, y = np.array(x), np.array(y)
    mask = y >= 0.05  # Ti: 剔除 < 0.05 keV
    x_clean, y_clean = x[mask], y[mask]
    if len(y_clean) == 0:
        f = interp1d(x, y + 0.05, kind='linear', fill_value='extrapolate')
        x_fit = np.linspace(0, 1.0, num_points)
        return x_fit, f(x_fit)
    f = interp1d(x_clean, y_clean, kind='linear', fill_value='extrapolate')
    x_fit = np.linspace(0, 1.0, num_points)
    y_fit = f(x_fit)
    if np.min(y_fit) < 0.05:
        f = interp1d(x, y + 0.05, kind='linear', fill_value='extrapolate')
        x_fit = np.linspace(0, 1.0, num_points)
        y_fit = f(x_fit)
    return x_fit, y_fit

def enforce_monotone_pchip(x, y, num_points=201):
    """PCHIP 强制单调递减。"""
    x, y = np.asarray(x), np.asarray(y)
    order = np.argsort(x)
    x, y = x[order], y[order]
    pchip = PchipInterpolator(x, -y)  # 取反实现递减
    x_new = np.linspace(0, 1, num_points)
    y_new = -pchip(x_new)
    return x_new, y_new

# 调用 mtanh 拟合
x_fit, y_fit, x_del, y_del, q = fitting(
    x_raw, y_raw, 'Ti',
    te_ped_x=te_ped_x, te_ped_y=te_ped_y
)

print(f'  After fitting: {len(x_fit)} pts (non-uniform grid)')
print(f'  Removed outliers: {len(x_del)} pts')

# 插值 + 单调化
x_201, y_201 = robust_interp(x_fit, y_fit, 'Ti')
x_201, y_201 = enforce_monotone_pchip(x_201, y_201)

print(f'  Final: {len(x_201)} uniform pts')
print(f'  Ti_core = {y_201[0]:.4f} keV')
print(f'  Ti_edge = {y_201[-1]:.4f} keV')

# ================================================================
# 绘图 + 保存
# ================================================================

os.makedirs(args.output, exist_ok=True)
basename = os.path.splitext(os.path.basename(args.npz_file))[0]

# 保存 201 点剖面
np.savez(os.path.join(args.output, f'{basename}_mtanh.npz'),
         rho=x_201.astype(np.float32), y=y_201.astype(np.float32),
         method='mtanh', unit='keV')

# 绘图
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(x_raw, y_raw, marker='.', c='#2E86AB', s=30, label='XCS Ti data')
if len(x_del) > 0:
    ax.scatter(x_del, y_del, marker='x', s=60, c='r', label='IQR removed')
ax.plot(x_201, y_201, 'r-', lw=2, label='mtanh fit')
ax.set_xlabel(r'$\rho$', fontsize=12)
ax.set_ylabel('Ti (keV)', fontsize=12)
ax.set_xlim(0, 1)
ax.set_title(f'mtanh Ti Profile Fit — {basename}', fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

fig.tight_layout()
save_path = os.path.join(args.output, f'{basename}_mtanh.png')
fig.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close(fig)

print(f'\nOutput:')
print(f'  NPZ:  {args.output}/{basename}_mtanh.npz')
print(f'  Plot: {save_path}')
