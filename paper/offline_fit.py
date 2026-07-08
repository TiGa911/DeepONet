# -*- coding: utf-8 -*-
"""
offline_fit.py — 离线剖面拟合脚本（不需要 MDSplus）

从本地数据文件读取 Te/ne/Ti 散点，运行 mtanh + 4 个 NN 模型拟合。

== 输入格式 ==

方式A — 单文件 JSON（推荐）：
  {
    "shot": 156010, "time": 5.0,
    "Te": {"rho": [0.1, 0.2, ...], "value": [1500, 1200, ...], "unit": "eV"},
    "ne": {"rho": [0.1, 0.2, ...], "value": [2.1, 1.9, ...], "unit": "1e19"},
    "Ti": {"rho": [0.5, 0.6, ...], "value": [0.8, 0.7, ...], "unit": "keV"}
  }

方式B — 分列文本文件：
  data/{shot}/{time}_Te.txt   → 两列: ρ Te(eV)
  data/{shot}/{time}_ne.txt   → 两列: ρ ne(1e19 m^-3)
  data/{shot}/{time}_Ti.txt   → 两列: ρ Ti(keV)  [可选]

== 用法 ==

  # JSON 输入
  python3 offline_fit.py input.json

  # 文本文件输入（指定目录）
  python3 offline_fit.py --txt-dir data/156010 --shot 156010 --time 5.0

  # 带 gfile（用于 LHW 计算）
  python3 offline_fit.py input.json --gfile g0_input

== 输出 ==

  offline_results/{shot}_{time}/
    ├── comparison/          # 5 方法叠加对比图
    │   ├── Te_comparison.png
    │   ├── ne_comparison.png
    │   └── Ti_comparison.png
    └── fit_result.json     # 所有方法的拟合值 (201 点数组)
"""

import numpy as np
import os
import sys
import json
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

MODEL_DIR = os.path.join(SCRIPT_DIR, 'profile_nn_models')
OUTPUT_BASE = os.path.join(SCRIPT_DIR, 'offline_results')

# =============================================================================
# 命令行
# =============================================================================

parser = argparse.ArgumentParser(description='Offline profile fitting (no MDSplus)')
parser.add_argument('input', nargs='?', help='JSON input file path')
parser.add_argument('--txt-dir', default=None, help='Directory with _Te.txt, _ne.txt, _Ti.txt files')
parser.add_argument('--shot', type=int, default=0, help='Shot number (for txt mode)')
parser.add_argument('--time', type=float, default=0.0, help='Time point (for txt mode)')
parser.add_argument('--gfile', default=None, help='gfile 路径（启用 LHW + inone 生成）')
parser.add_argument('--z-eff', type=float, default=2.0, help='有效电荷数（默认 2.0）')
parser.add_argument('--real-time', type=float, default=None, help='gfile 实际时间（默认用 --time）')
parser.add_argument('--lhw-method', default='nn', help='用于 LHW 的 Te/ne 拟合方法 (nn/lstm/cnn/transformer/mtanh)')
parser.add_argument('--output', default=OUTPUT_BASE, help='Output base directory')
parser.add_argument('--methods', default='all', choices=['all', 'mtanh', 'nn'])
args = parser.parse_args()

OUTPUT_BASE = args.output

# =============================================================================
# 数据加载
# =============================================================================

def load_txt_data(txt_dir, shot, time_val):
    """从文本文件加载 Te/ne/Ti 散点数据。

    文件命名：{time}_Te.txt, {time}_ne.txt, {time}_Ti.txt
    格式：两列空格分隔，第一列 ρ，第二列 value
    """
    def _load(label, suffix, default_unit):
        fpath = os.path.join(txt_dir, f'{time_val}_{suffix}.txt')
        if not os.path.exists(fpath):
            return None
        data = np.loadtxt(fpath)
        if data.ndim == 1:
            data = data.reshape(-1, 2)
        return {'rho': data[:, 0].tolist(), 'value': data[:, 1].tolist(), 'unit': default_unit}

    result = {'shot': shot, 'time': time_val}
    te = _load('Te', 'Te', 'eV')
    if te: result['Te'] = te
    ne = _load('ne', 'ne', '1e19')
    if ne: result['ne'] = ne
    ti = _load('Ti', 'Ti', 'keV')
    if ti: result['Ti'] = ti
    return result


def load_json_data(path):
    """从 JSON 文件加载数据。"""
    with open(path, 'r') as f:
        return json.load(f)


def load_data():
    """统一入口：根据命令行参数加载数据。"""
    if args.txt_dir:
        return load_txt_data(args.txt_dir, args.shot, args.time)
    elif args.input:
        return load_json_data(args.input)
    else:
        print('ERROR: 需要 JSON 文件 或 --txt-dir + --shot + --time')
        sys.exit(1)


def normalize_te(data_dict):
    """将 Te 统一转为 keV（NN 模型输入要求 eV，输出 keV）"""
    d = data_dict.copy()
    if 'Te' in d:
        unit = d['Te'].get('unit', 'eV')
        if unit == 'eV':
            d['Te_eV'] = np.array(d['Te']['value'], dtype=np.float64)
            d['Te_keV'] = d['Te_eV'] / 1000.0
        elif unit == 'keV':
            d['Te_keV'] = np.array(d['Te']['value'], dtype=np.float64)
            d['Te_eV'] = d['Te_keV'] * 1000.0
        else:
            d['Te_eV'] = np.array(d['Te']['value'], dtype=np.float64)
            d['Te_keV'] = d['Te_eV'] / 1000.0
        d['Te_rho'] = np.array(d['Te']['rho'], dtype=np.float64)
    return d


# =============================================================================
# 导入拟合方法
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
    y_proc = -y if decreasing else y
    pchip = PchipInterpolator(x, y_proc)
    x_new = np.linspace(0, 1, num_points)
    y_new = pchip(x_new)
    return x_new, -y_new if decreasing else y_new


# NN 模型加载
NN_METHODS = {}

def _try_load_nn(name, infer_module, set_func, te_fn, ne_fn, ti_fn, label, color, suffix):
    try:
        mod = __import__(infer_module, fromlist=[set_func, te_fn, ne_fn, ti_fn])
        getattr(mod, set_func)(MODEL_DIR)
        NN_METHODS[name] = {
            'te': getattr(mod, te_fn), 'ne': getattr(mod, ne_fn), 'ti': getattr(mod, ti_fn),
            'label': label, 'color': color, 'suffix': suffix,
        }
    except Exception as e:
        print(f'  ⚠ {label} not available: {e}')

_try_load_nn('nn', 'profile_nn.infer', 'set_model_dir',
             'nn_fit_te', 'nn_fit_ne', 'nn_fit_ti', 'ProfileNet', 'b', 'nn')
_try_load_nn('lstm', 'profile_nn.infer_lstm', 'set_model_dir_lstm',
             'nn_fit_te_lstm', 'nn_fit_ne_lstm', 'nn_fit_ti_lstm', 'LSTM', 'r', 'lstm')
_try_load_nn('cnn', 'profile_nn.infer_cnn', 'set_model_dir_cnn',
             'nn_fit_te_cnn', 'nn_fit_ne_cnn', 'nn_fit_ti_cnn', 'CNN-1D', 'c', 'cnn')
_try_load_nn('transformer', 'profile_nn.infer_transformer', 'set_model_dir_transformer',
             'nn_fit_te_transformer', 'nn_fit_ne_transformer', 'nn_fit_ti_transformer',
             'Transformer', 'm', 'tf')

# =============================================================================
# 拟合流程
# =============================================================================

def fit_all(data):
    """对所有诊断运行所有拟合方法。"""
    data = normalize_te(data)
    shot = data.get('shot', 0)
    time_val = data.get('time', 0.0)
    results = {'shot': shot, 'time': time_val, 'Te': {}, 'ne': {}, 'Ti': {}}
    rho_grid = np.linspace(0, 1, 201)

    # === Te ===
    if 'Te_eV' in data:
        x_raw, y_raw = data['Te_rho'], data['Te_eV']
        # mtanh
        try:
            dtype = 'Te'
            x_f, y_f, _, _, _ = fitting(x_raw, y_raw, dtype)
            x_mt, y_mt = robust_interp(x_f, y_f, dtype)
            results['Te']['mtanh'] = {'x': x_mt.tolist(), 'y': y_mt.tolist(), 'unit': 'keV'}
        except Exception as e:
            print(f'  ✗ mtanh_Te: {e}')

        # NN
        for key, method in NN_METHODS.items():
            try:
                x_nn, y_nn = method['te'](x_raw, y_raw)
                results['Te'][key] = {'x': x_nn.tolist(), 'y': y_nn.tolist(), 'unit': 'keV'}
            except Exception as e:
                print(f'  ✗ {key}_Te: {e}')

    # === ne ===
    if 'ne' in data:
        x_raw = np.array(data['ne']['rho'], dtype=np.float64)
        y_raw = np.array(data['ne']['value'], dtype=np.float64)
        dtype = 'Refl'
        try:
            x_f, y_f, _, _, _ = fitting(x_raw, y_raw, dtype)
            x_mt, y_mt = robust_interp(x_f, y_f, dtype)
            results['ne']['mtanh'] = {'x': x_mt.tolist(), 'y': y_mt.tolist(), 'unit': '1e19 m^-3'}
        except Exception as e:
            print(f'  ✗ mtanh_ne: {e}')

        for key, method in NN_METHODS.items():
            try:
                x_nn, y_nn = method['ne'](x_raw, y_raw)
                results['ne'][key] = {'x': x_nn.tolist(), 'y': y_nn.tolist(), 'unit': '1e19 m^-3'}
            except Exception as e:
                print(f'  ✗ {key}_ne: {e}')

    # === Ti ===
    if 'Ti' in data:
        x_raw = np.array(data['Ti']['rho'], dtype=np.float64)
        y_raw = np.array(data['Ti']['value'], dtype=np.float64)
        dtype = 'Ti'

        # Te pedestal (ρ ≥ 0.88)
        te_ped_x, te_ped_y = None, None
        if 'mtanh' in results.get('Te', {}):
            te_mt = np.array(results['Te']['mtanh']['x']), np.array(results['Te']['mtanh']['y'])
            mask = te_mt[0] >= 0.88
            te_ped_x, te_ped_y = te_mt[0][mask], te_mt[1][mask]

        try:
            x_ti, y_ti, _, _, _ = fitting(x_raw, y_raw, dtype,
                                           te_ped_x=te_ped_x, te_ped_y=te_ped_y)
            x_mt, y_mt = robust_interp(x_ti, y_ti, dtype)
            x_mt, y_mt = enforce_monotone_pchip(x_mt, y_mt)
            results['Ti']['mtanh'] = {'x': x_mt.tolist(), 'y': y_mt.tolist(), 'unit': 'keV'}
        except Exception as e:
            print(f'  ✗ mtanh_Ti: {e}')

        for key, method in NN_METHODS.items():
            try:
                if te_ped_x is not None and te_ped_y is not None:
                    x_nn, y_nn = method['ti'](x_raw, y_raw, te_ped_x, te_ped_y)
                else:
                    x_nn, y_nn = method['ti'](x_raw, y_raw,
                                              np.array([0.9, 0.95, 1.0]),
                                              np.array([0.1, 0.05, 0.01]))
                results['Ti'][key] = {'x': x_nn.tolist(), 'y': y_nn.tolist(), 'unit': 'keV'}
            except Exception as e:
                print(f'  ✗ {key}_Ti: {e}')

    return results


# =============================================================================
# 绘图
# =============================================================================

def plot_comparison(x_raw, y_raw, fits_dict, datatype, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x_raw, y_raw, marker='.', c='gray', alpha=0.5, s=10, label='Raw data')

    styles = {'mtanh': 'g--', 'nn': 'b-', 'lstm': 'r-', 'cnn': 'c-', 'transformer': 'm-'}
    for name, (x_fit, y_fit) in fits_dict.items():
        ax.plot(x_fit, y_fit, styles.get(name, '-'), linewidth=1.5, label=name)

    ax.set_xlabel(r'$\rho$', fontsize=12)
    ax.set_xlim(0, 1)
    if datatype == 'Te':    ax.set_ylabel('Te (keV)', fontsize=12)
    elif datatype == 'ne':  ax.set_ylabel('ne (10$^{19}$ m$^{-3}$)', fontsize=12)
    else:                   ax.set_ylabel('Ti (keV)', fontsize=12)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# 主流程
# =============================================================================

def main():
    print(f'Loading data...')
    input_data = load_data()
    shot = input_data.get('shot', 0)
    time_val = input_data.get('time', 0.0)

    print(f'Shot {shot} @ {time_val}s')
    has_te = 'Te' in input_data
    has_ne = 'ne' in input_data
    has_ti = 'Ti' in input_data
    print(f'Diagnostics: Te={"✓" if has_te else "✗"} ne={"✓" if has_ne else "✗"} Ti={"✓" if has_ti else "✗"}')
    print(f'Methods: mtanh + {", ".join(NN_METHODS.keys())}' if NN_METHODS else 'Methods: mtanh only')
    print()

    results = fit_all(input_data)

    # 输出目录
    ts = f'{time_val:.5f}'
    ip, dp = ts.split('.') if '.' in ts else (ts, '00000')
    time_dir = f'{int(ip):03d}.{dp.ljust(5, "0")[:5]}s'
    out_dir = os.path.join(OUTPUT_BASE, f'{shot}_{time_dir}')
    cmp_dir = os.path.join(out_dir, 'comparison')
    os.makedirs(cmp_dir, exist_ok=True)

    # 生成对比图
    for diag, unit, raw_key, raw_unit in [
        ('Te', 'keV', 'Te_keV', 'keV'),
        ('ne', '1e19', 'ne', '1e19 m^-3'),
        ('Ti', 'keV', 'Ti', 'keV'),
    ]:
        if diag not in results or not results[diag]:
            continue

        # 原始数据
        if raw_key in input_data:
            x_raw = np.array(input_data[raw_key]['rho'], dtype=np.float64)
            y_raw = np.array(input_data[raw_key]['value'], dtype=np.float64)
        elif raw_key in input_data.get('Te', {}):
            # Te 特殊处理
            x_raw = np.array(input_data['Te']['rho'], dtype=np.float64)
            y_raw = np.array(input_data['Te']['value'], dtype=np.float64)
            if input_data['Te'].get('unit') == 'eV':
                y_raw = y_raw / 1000.0
        else:
            continue

        fits_dict = {}
        for method_name, prof in results[diag].items():
            fits_dict[method_name] = (np.array(prof['x']), np.array(prof['y']))

        if fits_dict:
            plot_comparison(x_raw, y_raw, fits_dict, diag,
                          f'Shot {shot} @ {time_val:.3f}s — {diag} (offline)',
                          os.path.join(cmp_dir, f'{diag}_comparison.png'))

    # 保存数值结果
    result_path = os.path.join(out_dir, 'fit_result.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 汇总
    n_te = len(results.get('Te', {}))
    n_ne = len(results.get('ne', {}))
    n_ti = len(results.get('Ti', {}))
    print(f'\nDone. {n_te} Te + {n_ne} ne + {n_ti} Ti methods fitted.')
    print(f'Results: {out_dir}/')
    print(f'  comparison/  → 对比图 (PNG)')
    print(f'  fit_result.json → 数值结果 (201 点数组)')

    # =========================================================================
    # LHW 计算 + Namelist 装配（需要 --gfile）
    # =========================================================================
    if args.gfile and 'Te' in results and 'ne' in results:
        method = args.lhw_method
        if method not in results['Te']:
            # 回退到第一个可用的方法
            available = list(results['Te'].keys())
            if not available:
                print('\n⚠ No Te profile available for LHW, skipping.')
                return
            method = available[0]
            print(f'\n⚠ Method "{args.lhw_method}" not available, using "{method}"')

        te_prof = np.array(results['Te'][method]['y'])  # keV
        ne_prof = np.array(results['ne'][method]['y'])  # 1e19 m^-3
        rho_te = np.array(results['Te'][method]['x'])
        rho_ne = np.array(results['ne'][method]['x'])

        print(f'\n--- LHW Calculation ({method}) ---')
        print(f'  Te core: {te_prof[0]:.2f} keV, ne core: {ne_prof[0]:.2f} ×10¹⁹')

        # 检查 gfile 存在
        if not os.path.exists(args.gfile):
            print(f'  ✗ gfile not found: {args.gfile}')
        else:
            try:
                from lower_view_final_nn import lower_onetwo_nn
                gfile_time = args.real_time if args.real_time is not None else time_val
                lhw_dir = os.path.join(out_dir, 'lhw')
                os.makedirs(lhw_dir, exist_ok=True)

                rho_lhw, power, current = lower_onetwo_nn(
                    shot, time_val, lhw_dir,
                    te_profile_nn=te_prof, ne_profile_nn=ne_prof,
                    te_rho_nn=rho_te, ne_rho_nn=rho_ne,
                    gfile_path=os.path.abspath(args.gfile),
                    real_time=gfile_time, z_eff=args.z_eff,
                )
                print(f'  ✓ LHW done: {len(rho_lhw)} points, '
                      f'P_tot={np.trapz(power, rho_lhw):.4f} MW/m³, '
                      f'I_tot={np.trapz(current, rho_lhw):.4f} MA/m²')

                # 装配 ONETWO Namelist
                from Namelist3 import Namelist
                obj = Namelist()
                inone_template = os.path.join(SCRIPT_DIR, 'inone_template')
                if os.path.exists(inone_template):
                    obj.read(inone_template)
                else:
                    print('  ⚠ inone_template not found, creating minimal namelist')
                    obj['NAMELIS1'] = {}
                    obj['NAMELIS2'] = {}

                # Te (keV → eV)
                obj['NAMELIS1']['RTEIN'] = rho_te.tolist()
                obj['NAMELIS1']['TEIN'] = (te_prof * 1e3).tolist()
                # ne (1e19 m⁻³ → cm⁻³)
                obj['NAMELIS1']['RENEIN'] = rho_ne.tolist()
                obj['NAMELIS1']['ENEIN'] = (ne_prof * 1e13).tolist()
                # Ti (if available)
                if 'Ti' in results and method in results['Ti']:
                    ti_prof = np.array(results['Ti'][method]['y'])
                    rho_ti = np.array(results['Ti'][method]['x'])
                    obj['NAMELIS1']['RTIIN'] = rho_ti.tolist()
                    obj['NAMELIS1']['TIIN'] = ti_prof.tolist()
                # LHW
                obj['NAMELIS2']['extcurrf'] = [1.0]
                obj['NAMELIS2']['extcurrf_rho'] = rho_lhw.tolist()
                obj['NAMELIS2']['extcurrf_curr'] = (current * 1e6 / 1e4).tolist()
                obj['NAMELIS2']['extqerf'] = [1.0]
                obj['NAMELIS2']['extqerf_rho'] = rho_lhw.tolist()
                obj['NAMELIS2']['extqerf_qe'] = power.tolist()

                inone_path = os.path.join(lhw_dir, 'inone')
                obj.write(inone_path)
                print(f'  ✓ inone saved to: {inone_path}')

            except Exception as e:
                print(f'  ✗ LHW failed: {e}')
                import traceback
                traceback.print_exc()


if __name__ == '__main__':
    main()
