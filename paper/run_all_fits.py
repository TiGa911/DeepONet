# -*- coding: utf-8 -*-
"""
run_all_fits.py — 批量拟合 + NPZ 保存

对 4 个测试炮号的所有 TS 时间点，运行 mtanh + 4 个 NN 模型，
将 201 点剖面和原始散点保存为 NPZ，同时保存 metadata.json。

用法：
  pytorch_run paper/run_all_fits.py              # 全部 4 炮
  pytorch_run paper/run_all_fits.py --shot 156005 # 单炮
  pytorch_run paper/run_all_fits.py --methods nn  # 仅 NN（跳过 mtanh）

输出：
  paper_results/{shot}/{time_dir}/
    profiles/  {method}_{diag}.npz   # rho(201,) + y(201,)
    raw/       scatter_{diag}.npz    # 原始诊断散点
    metadata.json                    # shot, time, H98, plasma_mode
"""

import numpy as np
import os
import sys
import json
import argparse
import datetime
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 路径配置
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
MODEL_DIR = os.path.join(SCRIPT_DIR, 'profile_nn_models')
OUTPUT_BASE = os.path.join(SCRIPT_DIR, 'paper_results')

# 4 炮号测试集
TEST_SHOTS = [156005, 156010, 156100, 156400]

# =============================================================================
# 命令行
# =============================================================================

parser = argparse.ArgumentParser(description='Batch fit + NPZ save for paper Phase 2')
parser.add_argument('--shot', type=int, default=None,
                    help='Single shot (default: all 4 test shots)')
parser.add_argument('--methods', default='all',
                    choices=['all', 'mtanh', 'nn'],
                    help='Which methods (default: all)')
parser.add_argument('--output', default=OUTPUT_BASE,
                    help='Output base directory')
args = parser.parse_args()

if args.shot is not None:
    TEST_SHOTS = [args.shot]
OUTPUT_BASE = args.output

# =============================================================================
# 导入
# =============================================================================

from readMDS_onetwo import readmds
from MDSplus.connection import Connection
from fitting_mtanh import fitting
from scipy.interpolate import interp1d, PchipInterpolator

from metrics_utils import (
    METHOD_SPECS, NN_METHODS, DIAGNOSTICS,
    compute_metrics, compute_peakedness, compute_core_bias,
)

# =============================================================================
# 拟合工具（从 verify_fit.py 复用）
# =============================================================================

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


def fit_mtanh_te(x, y, datatype):
    """mtanh Te 拟合 → (x_201, y_201_keV)"""
    x_f, y_f, _, _, _ = fitting(x, y, datatype)
    x_fit, y_fit = robust_interp(x_f, y_f, datatype)
    return x_fit, y_fit


def fit_mtanh_ne(x, y, datatype):
    """mtanh ne 拟合 → (x_201, y_201_1e19)"""
    x_f, y_f, _, _, _ = fitting(x, y, datatype)
    x_fit, y_fit = robust_interp(x_f, y_f, datatype)
    return x_fit, y_fit


def fit_mtanh_ti(x, y, datatype, te_ped_x, te_ped_y):
    """mtanh Ti 拟合 → (x_201, y_201_keV)"""
    x_ti, y_ti, _, _, _ = fitting(x, y, datatype, te_ped_x=te_ped_x, te_ped_y=te_ped_y)
    x_ti, y_ti = robust_interp(x_ti, y_ti, datatype)
    x_ti, y_ti = enforce_monotone_pchip(x_ti, y_ti)
    return x_ti, y_ti


# =============================================================================
# NN 推理接口导入（同 verify_fit.py 模式）
# =============================================================================

NN_FITTERS = {}

try:
    from profile_nn.infer import nn_fit_te, nn_fit_ne, nn_fit_ti, set_model_dir
    set_model_dir(MODEL_DIR)
    NN_FITTERS['nn'] = {'te': nn_fit_te, 'ne': nn_fit_ne, 'ti': nn_fit_ti}
    print('✓ ProfileNet loaded')
except Exception as e:
    print(f'⚠ ProfileNet not available: {e}')

try:
    from profile_nn.infer_lstm import (
        nn_fit_te_lstm, nn_fit_ne_lstm, nn_fit_ti_lstm, set_model_dir_lstm
    )
    set_model_dir_lstm(MODEL_DIR)
    NN_FITTERS['lstm'] = {'te': nn_fit_te_lstm, 'ne': nn_fit_ne_lstm, 'ti': nn_fit_ti_lstm}
    print('✓ LSTM loaded')
except Exception as e:
    print(f'⚠ LSTM not available: {e}')

try:
    from profile_nn.infer_cnn import (
        nn_fit_te_cnn, nn_fit_ne_cnn, nn_fit_ti_cnn, set_model_dir_cnn
    )
    set_model_dir_cnn(MODEL_DIR)
    NN_FITTERS['cnn'] = {'te': nn_fit_te_cnn, 'ne': nn_fit_ne_cnn, 'ti': nn_fit_ti_cnn}
    print('✓ CNN-1D loaded')
except Exception as e:
    print(f'⚠ CNN not available: {e}')

try:
    from profile_nn.infer_transformer import (
        nn_fit_te_transformer, nn_fit_ne_transformer, nn_fit_ti_transformer,
        set_model_dir_transformer
    )
    set_model_dir_transformer(MODEL_DIR)
    NN_FITTERS['transformer'] = {
        'te': nn_fit_te_transformer, 'ne': nn_fit_ne_transformer, 'ti': nn_fit_ti_transformer
    }
    print('✓ Transformer loaded')
except Exception as e:
    print(f'⚠ Transformer not available: {e}')

RUN_MTANH = args.methods in ('all', 'mtanh')
RUN_NN = args.methods in ('all', 'nn')

# =============================================================================
# H98 读取消
# =============================================================================

def read_h98(shot: int, time_val: float) -> dict:
    """从 EAST MDSplus energy_east 读取 H98 并判定 H/L 模式。"""
    try:
        conn = Connection('202.127.204.42')
        conn.openTree('energy_east', shot)
        H98_times = conn.get(r'dim_of(\H98_MHD)').data()
        H98_data = conn.get(r'data(\H98_MHD)').data()
        conn.closeTree('energy_east', shot)

        H98_times = np.asarray(H98_times, dtype=np.float64).flatten()
        H98_data = np.asarray(H98_data, dtype=np.float64).flatten()

        idx = np.argmin(np.abs(H98_times - time_val))
        h98_val = float(H98_data[idx])

        mode = 'H' if (h98_val >= 0.7 and not np.isnan(h98_val)) else 'L'
        return {'h98': h98_val, 'plasma_mode': mode, 'h98_time': float(H98_times[idx])}
    except Exception as e:
        print(f'  [H98 read failed: {e}]', end='')
        return {'h98': None, 'plasma_mode': 'H', 'h98_time': None}


# =============================================================================
# 工具
# =============================================================================

def format_time_dir(t: float) -> str:
    ts = f'{t:.5f}'
    ip, dp = ts.split('.') if '.' in ts else (ts, '00000')
    return f'{int(ip):03d}.{dp.ljust(5, "0")[:5]}s'


def save_npz(path: str, rho: np.ndarray, y: np.ndarray, **meta):
    """保存剖面为 NPZ，附带到元数据。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_dict = {'rho': rho.astype(np.float32), 'y': y.astype(np.float32)}
    for k, v in meta.items():
        save_dict[k] = v
    np.savez_compressed(path, **save_dict)


# =============================================================================
# 单时间点处理
# =============================================================================

def process_time_point(shot: int, time_val: float):
    """运行所有方法，保存 NPZ + metadata。返回 stats dict。"""
    time_dir = format_time_dir(time_val)
    print(f'  [{time_dir}]', end='', flush=True)

    result_dir = os.path.join(OUTPUT_BASE, str(shot), time_dir)
    profiles_dir = os.path.join(result_dir, 'profiles')
    raw_dir = os.path.join(result_dir, 'raw')
    os.makedirs(profiles_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    # ---- 读诊断数据 ----
    try:
        data, status, real_time = readmds(shot, time_val)
    except Exception as e:
        print(f' readmds FAIL: {e}')
        return None

    # ---- 读 H98 ----
    h98_info = read_h98(shot, time_val)

    # ---- 保存元数据 ----
    metadata = {
        'shot': shot,
        'time': float(time_val),
        'real_time': float(real_time),
        'h98': h98_info['h98'],
        'plasma_mode': h98_info['plasma_mode'],
        'TS_status': bool(status.get('TS_status')),
        'Refl_status': bool(status.get('Refl_status')),
        'TXCS_status': bool(status.get('TXCS_status')),
    }
    meta_path = os.path.join(result_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f' mode={h98_info["plasma_mode"]} H98={h98_info["h98"]}', end='')

    stats = {'shot': shot, 'time': time_val, 'real_time': real_time,
             'plasma_mode': h98_info['plasma_mode'], 'h98': h98_info['h98']}

    # ---- Te ----
    if status['TS_status'] == 1:
        x_raw = np.asarray(data['Te']['TS']['Rho'], dtype=np.float64)
        y_raw = np.asarray(data['Te']['TS']['data'], dtype=np.float64)  # eV
        # 过滤 NaN（MDSplus 原始数据可能含 NaN，mtanh 的 clean() 会自动处理，但 NN 不会）
        valid_te = ~np.isnan(x_raw) & ~np.isnan(y_raw)
        if not valid_te.all():
            print(f' [Te:drop {sum(~valid_te)} NaN]', end='')
            x_raw = x_raw[valid_te]; y_raw = y_raw[valid_te]
        datatype = data['Te']['TS']['type']

        # 保存原始散点（eV，与 NN 推理输入一致）
        save_npz(os.path.join(raw_dir, 'scatter_Te.npz'), x_raw, y_raw,
                 shot=shot, time=time_val, unit='eV')

        # mtanh
        if RUN_MTANH:
            try:
                x_mt, y_mt = fit_mtanh_te(x_raw, y_raw, datatype)
                save_npz(os.path.join(profiles_dir, 'mtanh_Te.npz'), x_mt, y_mt,
                         shot=shot, time=time_val, method='mtanh', unit='keV')
                stats['te_mtanh_core'] = float(y_mt[0])
                stats['te_mtanh_peakedness'] = float(compute_peakedness(y_mt))
                print(' Te:mtanh', end='')
            except Exception as e:
                print(f' Te:mtanh!{e}', end='')

        # NN
        if RUN_NN:
            for key in NN_FITTERS:
                try:
                    x_nn, y_nn = NN_FITTERS[key]['te'](x_raw, y_raw)
                    save_npz(os.path.join(profiles_dir, f'{key}_Te.npz'), x_nn, y_nn,
                             shot=shot, time=time_val, method=key, unit='keV')
                    stats[f'te_{key}_core'] = float(y_nn[0])
                except Exception as e:
                    print(f' Te:{key}!{e}', end='')

        try:
            del_x_mt_ped = x_mt[x_mt >= 0.88]
            del_y_mt_ped = y_mt[x_mt >= 0.88]
        except Exception:
            del_x_mt_ped = np.array([])
            del_y_mt_ped = np.array([])

    # ---- ne ----
    if status['Refl_status'] == 1:
        x_raw = np.asarray(data['ne']['Refl']['Rho'], dtype=np.float64)
        y_raw = np.asarray(data['ne']['Refl']['data'], dtype=np.float64)  # 1e19 m^-3
        # 过滤 NaN
        valid_ne = ~np.isnan(x_raw) & ~np.isnan(y_raw)
        if not valid_ne.all():
            print(f' [ne:drop {sum(~valid_ne)} NaN]', end='')
            x_raw = x_raw[valid_ne]; y_raw = y_raw[valid_ne]
        datatype = data['ne']['Refl']['type']

        save_npz(os.path.join(raw_dir, 'scatter_ne.npz'), x_raw, y_raw,
                 shot=shot, time=time_val, unit='1e19 m^-3')

        if RUN_MTANH:
            try:
                x_mt, y_mt = fit_mtanh_ne(x_raw, y_raw, datatype)
                save_npz(os.path.join(profiles_dir, 'mtanh_ne.npz'), x_mt, y_mt,
                         shot=shot, time=time_val, method='mtanh', unit='1e19 m^-3')
                stats['ne_mtanh_core'] = float(y_mt[0])
                stats['ne_mtanh_peakedness'] = float(compute_peakedness(y_mt))
                print(' ne:mtanh', end='')
            except Exception as e:
                print(f' ne:mtanh!{e}', end='')

        if RUN_NN:
            for key in NN_FITTERS:
                try:
                    x_nn, y_nn = NN_FITTERS[key]['ne'](x_raw, y_raw)
                    save_npz(os.path.join(profiles_dir, f'{key}_ne.npz'), x_nn, y_nn,
                             shot=shot, time=time_val, method=key, unit='1e19 m^-3')
                    stats[f'ne_{key}_core'] = float(y_nn[0])
                except Exception as e:
                    print(f' ne:{key}!{e}', end='')

    # 若 Te 不可用，初始化空的台基数据（供 Ti 使用）
    if status['TS_status'] != 1:
        del_x_mt_ped = np.array([])
        del_y_mt_ped = np.array([])

    # ---- Ti ----
    if status['TXCS_status'] == 1:
        x_raw = np.asarray(data['Ti']['TXCS']['Rho'], dtype=np.float64)
        y_raw = np.asarray(data['Ti']['TXCS']['data'], dtype=np.float64)  # keV
        # 过滤 NaN
        valid_ti = ~np.isnan(x_raw) & ~np.isnan(y_raw)
        if not valid_ti.all():
            print(f' [Ti:drop {sum(~valid_ti)} NaN]', end='')
            x_raw = x_raw[valid_ti]; y_raw = y_raw[valid_ti]
        datatype = data['Ti']['type']

        save_npz(os.path.join(raw_dir, 'scatter_Ti.npz'), x_raw, y_raw,
                 shot=shot, time=time_val, unit='keV')

        # 需要 Te pedestal 作为 Ti 拟合约束
        te_ped_x, te_ped_y = del_x_mt_ped, del_y_mt_ped
        if len(te_ped_x) == 0 and status['TS_status'] == 1:
            # 回退：用 Te 数据拟合获取 pedestal
            try:
                te_x_r = np.asarray(data['Te']['TS']['Rho'], dtype=np.float64)
                te_y_r = np.asarray(data['Te']['TS']['data'], dtype=np.float64)
                valid_te2 = ~np.isnan(te_x_r) & ~np.isnan(te_y_r)
                te_x_r = te_x_r[valid_te2]; te_y_r = te_y_r[valid_te2]
                x_te_mt, y_te_mt = fit_mtanh_te(te_x_r, te_y_r, data['Te']['TS']['type'])
                mask = x_te_mt >= 0.88
                te_ped_x, te_ped_y = x_te_mt[mask], y_te_mt[mask]
            except Exception:
                te_ped_x, te_ped_y = np.array([]), np.array([])

        if RUN_MTANH and len(te_ped_x) > 0:
            try:
                x_mt, y_mt = fit_mtanh_ti(x_raw, y_raw, datatype, te_ped_x, te_ped_y)
                save_npz(os.path.join(profiles_dir, 'mtanh_Ti.npz'), x_mt, y_mt,
                         shot=shot, time=time_val, method='mtanh', unit='keV')
                stats['ti_mtanh_core'] = float(y_mt[0])
                stats['ti_mtanh_peakedness'] = float(compute_peakedness(y_mt))
                print(' Ti:mtanh', end='')
            except Exception as e:
                print(f' Ti:mtanh!{e}', end='')

        if RUN_NN and len(te_ped_x) > 0:
            for key in NN_FITTERS:
                try:
                    x_nn, y_nn = NN_FITTERS[key]['ti'](x_raw, y_raw, te_ped_x, te_ped_y)
                    save_npz(os.path.join(profiles_dir, f'{key}_Ti.npz'), x_nn, y_nn,
                             shot=shot, time=time_val, method=key, unit='keV')
                    stats[f'ti_{key}_core'] = float(y_nn[0])
                except Exception as e:
                    print(f' Ti:{key}!{e}', end='')

    n_ok = sum(1 for k in stats if 'core' in k)
    print(f' ({n_ok} profiles)', flush=True)
    return stats


# =============================================================================
# 主流程
# =============================================================================

def main():
    print(f'{"="*60}')
    print(f'Batch Fit + NPZ Save — Phase 2')
    print(f'Shots: {TEST_SHOTS}')
    print(f'Methods: mtanh={"ON" if RUN_MTANH else "OFF"}, NN={"ON" if RUN_NN else "OFF"}')
    print(f'NN models: {list(NN_FITTERS.keys())}')
    print(f'Output: {OUTPUT_BASE}/')
    print(f'{"="*60}')

    import time as _time
    t_start = _time.time()
    total_profiles = 0

    for shot in TEST_SHOTS:
        print(f'\n--- Shot {shot} ---')

        # 获取 TS 时间点列表
        try:
            conn = Connection('202.127.204.42')
            conn.openTree('TS_EAST', shot)
            ts_times = conn.get(r'dim_of(\Te_coreTS)').data()
            conn.closeTree('TS_EAST', shot)
            ts_times = np.asarray(ts_times, dtype=np.float64).flatten()
            print(f'  {len(ts_times)} TS time points')
        except Exception as e:
            print(f'  Cannot read TS times: {e}')
            continue

        shot_stats = []
        for t_val in ts_times:
            try:
                stats = process_time_point(shot, float(t_val))
                if stats:
                    shot_stats.append(stats)
            except Exception as e:
                print(f'  [{t_val:.5f}s] FAIL: {e}', flush=True)

        total_profiles += len(shot_stats)
        print(f'  Shot {shot}: {len(shot_stats)}/{len(ts_times)} time points OK')

    elapsed = _time.time() - t_start
    print(f'\n{"="*60}')
    print(f'DONE: {total_profiles} time points ({elapsed:.1f}s)')
    print(f'Output: {OUTPUT_BASE}/')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
