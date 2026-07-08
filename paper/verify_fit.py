# -*- coding: utf-8 -*-
"""
verify_fit.py — 单炮号拟合验证脚本

对单个测试炮号的所有时间点，运行 mtanh + 4 个 NN 模型，生成拟合对比图。

用法：
  python3 verify_fit.py 156010              # 验证单个炮号
  python3 verify_fit.py 156010 --methods nn  # 仅 NN 模型
  python3 verify_fit.py 156010 --time 5.0    # 仅单个时间点

输出：
  paper_results/{shot}/
    ├── summary.txt                          # 汇总统计
    ├── {time_dir}/
    │   ├── mtanh/   Te/ne/Ti 拟合图
    │   ├── nn/      ProfileNet 拟合图
    │   ├── lstm/    LSTM 拟合图
    │   ├── cnn/     CNN-1D 拟合图
    │   └── transformer/  Transformer 拟合图
"""

import numpy as np
import os
import sys
import time
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 配置
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)  # 确保 paper/ 内模块可导入
MODEL_DIR = os.path.join(SCRIPT_DIR, 'profile_nn_models')
OUTPUT_BASE = os.path.join(SCRIPT_DIR, 'paper_results')

# =============================================================================
# 命令行
# =============================================================================

parser = argparse.ArgumentParser(description='Verify profile fitting on a test shot')
parser.add_argument('shot', type=int, help='Shot number')
parser.add_argument('--methods', default='all',
                    choices=['all', 'mtanh', 'nn'],
                    help='Which methods to run (default: all)')
parser.add_argument('--time', type=float, default=None,
                    help='Single time point (default: all TS time points)')
parser.add_argument('--output', default=OUTPUT_BASE,
                    help='Output base directory')
args = parser.parse_args()

SHOT = args.shot
OUTPUT_BASE = args.output

# =============================================================================
# 导入诊断读取器
# =============================================================================

from readMDS_onetwo import readmds
from MDSplus.connection import Connection

# =============================================================================
# 导入 mtanh 拟合链
# =============================================================================

from fitting_mtanh import fitting
from scipy.interpolate import interp1d, PchipInterpolator

def robust_interp(x, y, datatype, thresholds=None, offsets=None, num_points=201):
    if thresholds is None:
        thresholds = {"ne": 0.1, "Te": 0.05, "Ti": 0.05}
    if offsets is None:
        offsets = {"ne": 0.05, "Te": 0.05, "Ti": 0.05}
    th = thresholds.get(datatype, 0.0)
    off = offsets.get(datatype, 0.0)
    x, y = np.array(x), np.array(y)
    mask = y >= th
    x_clean, y_clean = x[mask], y[mask]
    if len(y_clean) == 0:
        y_lifted = y + off
        f = interp1d(x, y_lifted, kind='linear', fill_value="extrapolate")
        x_fit = np.linspace(0, 1.0, num_points)
        return x_fit, f(x_fit)
    f = interp1d(x_clean, y_clean, kind='linear', fill_value="extrapolate")
    x_fit = np.linspace(0, 1.0, num_points)
    y_fit = f(x_fit)
    if np.min(y_fit) < th:
        y_lifted = y + off
        f = interp1d(x, y_lifted, kind='linear', fill_value="extrapolate")
        x_fit = np.linspace(0, 1.0, num_points)
        y_fit = f(x_fit)
    return x_fit, y_fit

def enforce_monotone_pchip(x, y, num_points=201, decreasing=True):
    x, y = np.asarray(x), np.asarray(y)
    order = np.argsort(x)
    x, y = x[order], y[order]
    if decreasing:
        y_proc = -y
    else:
        y_proc = y
    pchip = PchipInterpolator(x, y_proc)
    x_new = np.linspace(0, 1, num_points)
    y_new = pchip(x_new)
    if decreasing:
        y_new = -y_new
    return x_new, y_new


# =============================================================================
# 导入 NN 推理接口
# =============================================================================

NN_METHODS = {}

try:
    from profile_nn.infer import nn_fit_te, nn_fit_ne, nn_fit_ti, set_model_dir
    set_model_dir(MODEL_DIR)
    NN_METHODS['nn'] = {
        'te': nn_fit_te, 'ne': nn_fit_ne, 'ti': nn_fit_ti,
        'label': 'ProfileNet', 'color': 'b', 'suffix': 'nn'
    }
except Exception as e:
    print(f'⚠ ProfileNet not available: {e}')

try:
    from profile_nn.infer_lstm import nn_fit_te_lstm, nn_fit_ne_lstm, nn_fit_ti_lstm, set_model_dir_lstm
    set_model_dir_lstm(MODEL_DIR)
    NN_METHODS['lstm'] = {
        'te': nn_fit_te_lstm, 'ne': nn_fit_ne_lstm, 'ti': nn_fit_ti_lstm,
        'label': 'LSTM', 'color': 'r', 'suffix': 'lstm'
    }
except Exception as e:
    print(f'⚠ LSTM not available: {e}')

try:
    from profile_nn.infer_cnn import nn_fit_te_cnn, nn_fit_ne_cnn, nn_fit_ti_cnn, set_model_dir_cnn
    set_model_dir_cnn(MODEL_DIR)
    NN_METHODS['cnn'] = {
        'te': nn_fit_te_cnn, 'ne': nn_fit_ne_cnn, 'ti': nn_fit_ti_cnn,
        'label': 'CNN-1D', 'color': 'c', 'suffix': 'cnn'
    }
except Exception as e:
    print(f'⚠ CNN not available: {e}')

try:
    from profile_nn.infer_transformer import nn_fit_te_transformer, nn_fit_ne_transformer, nn_fit_ti_transformer, set_model_dir_transformer
    set_model_dir_transformer(MODEL_DIR)
    NN_METHODS['transformer'] = {
        'te': nn_fit_te_transformer, 'ne': nn_fit_ne_transformer, 'ti': nn_fit_ti_transformer,
        'label': 'Transformer', 'color': 'm', 'suffix': 'tf'
    }
except Exception as e:
    print(f'⚠ Transformer not available: {e}')


# =============================================================================
# 拟合函数
# =============================================================================

def fit_mtanh_te(x, y, datatype):
    """mtanh Te 拟合 → (x_201, y_201_keV)"""
    x_f, y_f, _, _, _ = fitting(x, y, datatype)
    x_fit, y_fit = robust_interp(x_f, y_f, datatype)
    return x_fit, y_fit  # y 单位: keV（fitting 输出 keV）

def fit_mtanh_ne(x, y, datatype):
    """mtanh ne 拟合 → (x_201, y_201_1e19)"""
    x_f, y_f, _, _, _ = fitting(x, y, datatype)
    x_fit, y_fit = robust_interp(x_f, y_f, datatype)
    return x_fit, y_fit  # y 单位: 1e19 m^-3

def fit_mtanh_ti(x, y, datatype, te_ped_x, te_ped_y):
    """mtanh Ti 拟合 → (x_201, y_201_keV)"""
    x_ti, y_ti, _, _, _ = fitting(x, y, datatype, te_ped_x=te_ped_x, te_ped_y=te_ped_y)
    x_ti, y_ti = robust_interp(x_ti, y_ti, datatype)
    x_ti, y_ti = enforce_monotone_pchip(x_ti, y_ti)
    return x_ti, y_ti  # y 单位: keV


# =============================================================================
# 绘图
# =============================================================================

def plot_comparison(x_raw, y_raw, fits_dict, datatype, title, save_path):
    """绘制多方法对比图。

    fits_dict: {'mtanh': (x, y, 'g--'), 'nn': (x, y, 'b-'), ...}
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # 原始散点
    ax.scatter(x_raw, y_raw, marker='.', c='gray', alpha=0.5, s=10, label='Raw data')

    for name, (x_fit, y_fit, style) in fits_dict.items():
        ax.plot(x_fit, y_fit, style, linewidth=1.5, label=name)

    ax.set_xlabel(r'$\rho$', fontsize=12)
    ax.set_xlim(0, 1)

    if datatype == 'Te':
        ax.set_ylabel('Te (keV)', fontsize=12)
    elif datatype == 'ne':
        ax.set_ylabel('ne (10$^{19}$ m$^{-3}$)', fontsize=12)
    else:
        ax.set_ylabel('Ti (keV)', fontsize=12)

    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# 主流程
# =============================================================================

def format_time_dir(t):
    ts = f'{t:.5f}'
    ip, dp = ts.split('.') if '.' in ts else (ts, '00000')
    return f'{int(ip):03d}.{dp.ljust(5, "0")[:5]}s'


def process_single_time(shot, time_val, data, status, real_time):
    """处理单个时间点：运行所有方法并生成对比图"""
    time_dir = format_time_dir(time_val)
    print(f'\n  [{time_dir}] real_time={real_time:.4f}s', end=' ', flush=True)

    output_dir = os.path.join(OUTPUT_BASE, str(shot), time_dir)
    os.makedirs(output_dir, exist_ok=True)

    stats = {'time': time_val, 'real_time': real_time}

    # === Te ===
    if status['TS_status'] == 1:
        x = data['Te']['TS']['Rho']
        y = data['Te']['TS']['data']  # eV
        dtype = data['Te']['TS']['type']

        y_kev = y / 1000.  # eV → keV（散点显示用）
        fits = {}

        # mtanh
        try:
            x_mt, y_mt = fit_mtanh_te(x, y, dtype)
            fits['mtanh'] = (x_mt, y_mt, 'g--')
            stats['te_mtanh_core'] = float(y_mt[0])
        except Exception as e:
            print(f'[mtanh_Te:{e}]', end='')

        # NN models
        for key, method in NN_METHODS.items():
            try:
                x_nn, y_nn = method['te'](x, y)
                fits[method['label']] = (x_nn, y_nn, f"{method['color']}-")
                stats[f'te_{key}_core'] = float(y_nn[0])
            except Exception as e:
                print(f'[{key}_Te:{e}]', end='')

        # 保存对比图
        if fits:
            os.makedirs(os.path.join(output_dir, 'comparison'), exist_ok=True)
            plot_comparison(x, y_kev, fits, 'Te',
                          f'Shot {shot} @ {time_val:.3f}s — Te',
                          os.path.join(output_dir, 'comparison', 'Te_comparison.png'))

    # === ne ===
    if status['Refl_status'] == 1:
        x = data['ne']['Refl']['Rho']
        y = data['ne']['Refl']['data']  # 1e19 m^-3
        dtype = data['ne']['Refl']['type']

        fits = {}
        try:
            x_mt, y_mt = fit_mtanh_ne(x, y, dtype)
            fits['mtanh'] = (x_mt, y_mt, 'g--')
            stats['ne_mtanh_core'] = float(y_mt[0])
        except Exception as e:
            print(f'[mtanh_ne:{e}]', end='')

        for key, method in NN_METHODS.items():
            try:
                x_nn, y_nn = method['ne'](x, y)
                fits[method['label']] = (x_nn, y_nn, f"{method['color']}-")
                stats[f'ne_{key}_core'] = float(y_nn[0])
            except Exception as e:
                print(f'[{key}_ne:{e}]', end='')

        if fits:
            os.makedirs(os.path.join(output_dir, 'comparison'), exist_ok=True)
            plot_comparison(x, y, fits, 'ne',
                          f'Shot {shot} @ {time_val:.3f}s — ne',
                          os.path.join(output_dir, 'comparison', 'ne_comparison.png'))

    # === Ti ===
    if status['TXCS_status'] == 1:
        x = data['Ti']['TXCS']['Rho']
        y = data['Ti']['TXCS']['data']  # keV
        dtype = data['Ti']['type']

        # 需要 Te pedestal
        te_ped_x, te_ped_y = None, None
        if status['TS_status'] == 1:
            te_x = data['Te']['TS']['Rho']
            te_y = data['Te']['TS']['data']
            try:
                x_te_mt, y_te_mt = fit_mtanh_te(te_x, te_y, data['Te']['TS']['type'])
                mask = x_te_mt >= 0.88
                te_ped_x, te_ped_y = x_te_mt[mask], y_te_mt[mask]
            except Exception:
                pass

        fits = {}
        if te_ped_x is not None:
            try:
                x_mt, y_mt = fit_mtanh_ti(x, y, dtype, te_ped_x, te_ped_y)
                fits['mtanh'] = (x_mt, y_mt, 'g--')
                stats['ti_mtanh_core'] = float(y_mt[0])
            except Exception as e:
                print(f'[mtanh_Ti:{e}]', end='')

            for key, method in NN_METHODS.items():
                try:
                    x_nn, y_nn = method['ti'](x, y, te_ped_x, te_ped_y)
                    fits[method['label']] = (x_nn, y_nn, f"{method['color']}-")
                    stats[f'ti_{key}_core'] = float(y_nn[0])
                except Exception as e:
                    print(f'[{key}_Ti:{e}]', end='')

        if fits:
            os.makedirs(os.path.join(output_dir, 'comparison'), exist_ok=True)
            plot_comparison(x, y, fits, 'Ti',
                          f'Shot {shot} @ {time_val:.3f}s — Ti',
                          os.path.join(output_dir, 'comparison', 'Ti_comparison.png'))

    n_ok = sum(1 for k in stats if 'core' in k)
    print(f'✓ {n_ok} profiles', end='', flush=True)
    return stats


def main():
    print(f'{"="*60}')
    print(f'Fit Verification — Shot {SHOT}')
    print(f'NN models: {list(NN_METHODS.keys())}')
    print(f'Output: {OUTPUT_BASE}/{SHOT}/')
    print(f'{"="*60}')

    # 获取 TS 时间点
    conn = Connection('202.127.204.42')
    conn.openTree('TS_EAST', SHOT)
    ts_times = conn.get(r'dim_of(\Te_coreTS)').data()
    conn.closeTree('TS_EAST', SHOT)

    if args.time is not None:
        ts_times = np.array([args.time])

    print(f'Time points: {len(ts_times)}')
    if args.time is None:
        print(f'Range: {ts_times[0]:.3f} — {ts_times[-1]:.3f}s')

    all_stats = []
    t0 = time.time()

    for t_val in ts_times:
        try:
            data, status, real_time = readmds(int(SHOT), float(t_val))
            stats = process_single_time(SHOT, float(t_val), data, status, real_time)
            all_stats.append(stats)
        except Exception as e:
            print(f'\n  [{t_val:.5f}s] ERROR: {e}')

    # 汇总
    elapsed = time.time() - t0
    n_success = len(all_stats)
    print(f'\n\n{"="*60}')
    print(f'SUMMARY: {n_success}/{len(ts_times)} time points completed ({elapsed:.1f}s)')
    print(f'Output: {OUTPUT_BASE}/{SHOT}/')
    print(f'{"="*60}')

    if all_stats:
        diag_types = []
        if any('te_' in k for s in all_stats for k in s):
            diag_types.append('Te')
        if any('ne_' in k for s in all_stats for k in s):
            diag_types.append('ne')
        if any('ti_' in k for s in all_stats for k in s):
            diag_types.append('Ti')
        print(f'Diagnostics available: {", ".join(diag_types)}')
        print(f'Methods tested: mtanh + {", ".join(NN_METHODS.keys())}')

        # 保存汇总
        summary_path = os.path.join(OUTPUT_BASE, str(SHOT), 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f'Shot {SHOT} — Fit Verification Summary\n')
            f.write(f'Time: {len(all_stats)} points processed\n')
            f.write(f'Diagnostics: {", ".join(diag_types)}\n')
            f.write(f'Methods: mtanh + {", ".join(NN_METHODS.keys())}\n')
            for s in all_stats[:5]:
                f.write(f'\n  t={s["time"]:.5f}s:\n')
                for k, v in s.items():
                    if k not in ('time', 'real_time'):
                        f.write(f'    {k}={v:.4f}\n')

        print(f'\nSummary saved to: {summary_path}')


if __name__ == '__main__':
    main()
