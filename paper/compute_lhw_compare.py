# -*- coding: utf-8 -*-
"""
compute_lhw_compare.py — LHW 下游影响对比

读取 NPZ 剖面 + gfile + P_in，计算 mtanh 与 4 个 NN 方法的 LHW 功率/电流剖面，
生成对比图和指标。

前置条件：
  1. 服务器已运行 fetch_lhw_data.py（获取 gfile + lhw_power.json）
  2. 已下载 paper_results/ 到本地（含 gfiles/ 和 lhw_power.json）

用法：
  python3 paper/compute_lhw_compare.py

输出：
  paper_results/figures/fig9_lhw_comparison.png    # LHW 典型对比图
  paper_results/all_lhw_metrics.csv                 # LHW 指标汇总
"""

import numpy as np
import os
import sys
import json
import csv
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
RESULT_BASE = os.path.join(SCRIPT_DIR, 'paper_results')
FIG_DIR = os.path.join(RESULT_BASE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

from metrics_utils import METHOD_SPECS, NN_METHODS

# LHW 物理参数（从 lower_view.py 的默认值）
N_PARALLEL0 = 2.0
DELTA_A = 0.4
FREQUENCY = 4.6e9
MAIN_ION_MASS = 2.0  # 氘

# ================================================================
# 数据加载
# ================================================================

def load_profile(shot, time_dir, method, diag):
    npz_path = os.path.join(RESULT_BASE, str(shot), time_dir,
                            'profiles', f'{method}_{diag}.npz')
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        return np.asarray(data['rho'], dtype=np.float64), np.asarray(data['y'], dtype=np.float64)
    return None, None


def discover_lhw_entries():
    """扫描 paper_results/ 下同时有 NPZ + gfile + power 的时间点。"""
    entries = []
    for shot_name in os.listdir(RESULT_BASE):
        shot_path = os.path.join(RESULT_BASE, shot_name)
        if not os.path.isdir(shot_path):
            continue
        try:
            shot = int(shot_name)
        except ValueError:
            continue

        # 检查 lhw_power.json
        power_path = os.path.join(shot_path, 'lhw_power.json')
        if not os.path.exists(power_path):
            continue
        with open(power_path) as f:
            power_data = json.load(f)

        gfile_dir = os.path.join(shot_path, 'gfiles')

        for td_name in os.listdir(shot_path):
            td_path = os.path.join(shot_path, td_name)
            if not os.path.isdir(td_path):
                continue
            profiles_dir = os.path.join(td_path, 'profiles')
            if not os.path.isdir(profiles_dir):
                continue

            # 必须有 mtanh_Te.npz 作为基准
            if not os.path.exists(os.path.join(profiles_dir, 'mtanh_Te.npz')):
                continue

            pw = power_data.get(td_name)
            if pw is None:
                continue

            gfile_name = pw.get('gfile', '')
            gfile_path = os.path.join(gfile_dir, gfile_name)
            if not os.path.exists(gfile_path):
                continue

            entries.append({
                'shot': shot,
                'time_dir': td_name,
                'gfile_path': gfile_path,
                'p_in_kw': pw['p_in_kw'],
                'zeff': pw.get('zeff', 2.0),
                'real_time': pw.get('real_time'),
                'time_val': pw.get('time_val'),
            })

    entries.sort(key=lambda e: (e['shot'], e.get('time_val', 0)))
    return entries


# ================================================================
# LHW 计算
# ================================================================

def compute_lhw_for_method(shot, entry, method):
    """对单个方法计算 LHW 电流和功率剖面。"""
    import geqdsk
    from mesh2 import g_to_23
    from lower_view import lower_hybrid_power_deposition

    # 加载 Te/ne
    rho_te, te_kev = load_profile(shot, entry['time_dir'], method, 'Te')
    rho_ne, ne_1e19 = load_profile(shot, entry['time_dir'], method, 'ne')
    if rho_te is None or rho_ne is None:
        return None

    # 加载 gfile + 计算几何
    gdate = geqdsk.load(entry['gfile_path'])
    rho = np.linspace(0, 1, gdate['nw'])

    # 插值 Te/ne 到 gfile rho 网格
    from scipy.interpolate import interp1d
    te_on_rho = interp1d(rho_te, te_kev, bounds_error=False, fill_value='extrapolate')(rho)
    ne_on_rho = interp1d(rho_ne, ne_1e19, bounds_error=False, fill_value='extrapolate')(rho)
    ne_m3 = np.clip(ne_on_rho * 1e19, 0, None)

    # Vrho 计算
    psinorm = np.linspace(0, 1, len(rho))
    theta = np.linspace(0, 2 * np.pi, len(rho))
    grid = g_to_23(entry['gfile_path'], psinorm, theta)
    dvdpsi = np.asarray(grid['dvdpsi'], dtype=float)
    phi = np.asarray(grid['phi'], dtype=float)
    phi0, phi_sep = phi[0], phi[-1]
    rho_psi = np.sqrt((phi - phi0) / (phi_sep - phi0 + 1e-30))
    psia, psib = gdate['simag'], gdate['sibry']
    psi_array = psia + np.linspace(0, 1, len(dvdpsi)) * (psib - psia)
    dpsi_drho = np.gradient(psi_array, rho_psi)
    Vrho_on_psi = dvdpsi * dpsi_drho
    Vrho_on_rho = interp1d(rho_psi, Vrho_on_psi,
                           bounds_error=False, fill_value='extrapolate')(rho)

    # 几何参数
    filtered = gdate['bbbsrz'][gdate['bbbsrz'][:, 0] != 0]
    bbbsrz = filtered[:, 0]
    R_axis = (np.max(bbbsrz) + np.min(bbbsrz)) / 2.0
    a = (np.max(bbbsrz) - np.min(bbbsrz)) / 2.0

    # q profile
    qpsi = gdate.get('qpsi', np.ones_like(rho))
    q_on_rho = interp1d(np.linspace(0, 1, len(qpsi)), qpsi,
                         bounds_error=False, fill_value='extrapolate')(rho)

    # 调用核心 LHW 模型
    try:
        p_lh, j_lhcd = lower_hybrid_power_deposition(
            rho=rho,
            Te_keV=np.clip(te_on_rho, 0.001, None),
            ne_m3=ne_m3,
            n_parallel0=N_PARALLEL0,
            delta_a=DELTA_A,
            R_axis=R_axis,
            a=a,
            q=q_on_rho,
            B0=abs(gdate.get('bcentr', 0.0)),
            frequency=FREQUENCY,
            Z_eff=entry['zeff'],
            main_ion_mass=MAIN_ION_MASS,
            P_in=entry['p_in_kw'] * 1e3,  # kW → W
            Vrho=Vrho_on_rho,
        )
    except Exception as e:
        print(f'    LHW computation failed for {method}: {e}')
        return None

    # 积分得总电流/功率
    from scipy.integrate import cumulative_trapezoid
    I_total_A = np.trapz(j_lhcd * Vrho_on_rho, rho) / (2 * np.pi * R_axis)
    P_total_W = np.trapz(p_lh * Vrho_on_rho, rho)

    return {
        'rho': rho,
        'p_lh_wm3': p_lh,        # W/m^3
        'j_lhcd_am2': j_lhcd,    # A/m^2
        'I_total_kA': I_total_A / 1e3,
        'P_total_MW': P_total_W / 1e6,
    }


# ================================================================
# 主流程
# ================================================================

def main():
    entries = discover_lhw_entries()
    print(f'Found {len(entries)} LHW-ready time points')

    if not entries:
        print('ERROR: No entries with gfile + power data.')
        print('Run fetch_lhw_data.py on the server first.')
        return

    all_rows = []

    for entry in entries:
        shot = entry['shot']
        td = entry['time_dir']
        p_in = entry['p_in_kw']
        print(f'\n  {shot}/{td}: P_in={p_in:.1f} kW')

        # 基准: mtanh
        mtanh_result = compute_lhw_for_method(shot, entry, 'mtanh')
        if mtanh_result is None:
            print(f'    mtanh LHW failed, skipping')
            continue

        print(f'    mtanh: I_LH={mtanh_result["I_total_kA"]:.1f} kA, '
              f'P_LH={mtanh_result["P_total_MW"]:.2f} MW')

        for method in NN_METHODS:
            nn_result = compute_lhw_for_method(shot, entry, method)
            if nn_result is None:
                print(f'    {method}: LHW failed')
                continue

            # 偏差
            dI = nn_result['I_total_kA'] - mtanh_result['I_total_kA']
            dI_pct = dI / max(abs(mtanh_result['I_total_kA']), 1e-10) * 100
            dP = nn_result['P_total_MW'] - mtanh_result['P_total_MW']
            dP_pct = dP / max(abs(mtanh_result['P_total_MW']), 1e-10) * 100

            # 电流密度剖面 RMS 偏差
            from metrics_utils import compute_metrics
            j_metrics = compute_metrics(
                mtanh_result['j_lhcd_am2'], nn_result['j_lhcd_am2'],
                mtanh_result['rho'], nn_result['rho']
            )

            print(f'    {method:12s}: I={nn_result["I_total_kA"]:.1f} kA ({dI_pct:+.1f}%), '
                  f'P={nn_result["P_total_MW"]:.2f} MW ({dP_pct:+.1f}%), '
                  f'j_RMS={j_metrics["RMSE"]:.4f} A/cm^2')

            all_rows.append({
                'shot': shot, 'time_dir': td,
                'p_in_kw': p_in,
                'method': method,
                'I_LH_kA': nn_result['I_total_kA'],
                'I_LH_mtanh_kA': mtanh_result['I_total_kA'],
                'dI_pct': dI_pct,
                'P_LH_MW': nn_result['P_total_MW'],
                'P_LH_mtanh_MW': mtanh_result['P_total_MW'],
                'dP_pct': dP_pct,
                'j_RMSE': j_metrics['RMSE'],
                'j_MAE': j_metrics['MAE'],
            })

    # ---- 保存 CSV ----
    if all_rows:
        csv_path = os.path.join(RESULT_BASE, 'all_lhw_metrics.csv')
        fieldnames = ['shot', 'time_dir', 'p_in_kw', 'method',
                      'I_LH_kA', 'I_LH_mtanh_kA', 'dI_pct',
                      'P_LH_MW', 'P_LH_mtanh_MW', 'dP_pct',
                      'j_RMSE', 'j_MAE']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_rows)
        print(f'\nCSV saved: {csv_path}')

        # 汇总
        for method in NN_METHODS:
            dI_vals = [abs(r['dI_pct']) for r in all_rows if r['method'] == method]
            dP_vals = [abs(r['dP_pct']) for r in all_rows if r['method'] == method]
            if dI_vals:
                print(f'  {method:12s}: |ΔI|={np.mean(dI_vals):.2f}%  |ΔP|={np.mean(dP_vals):.2f}%')

    # ---- 生成 fig9 ----
    generate_fig9(entries)


def generate_fig9(entries):
    """生成 LHW 典型对比图（选 LHW 功率最高的可用时间点，即 shot 156400）。"""
    # 找一个有真实 LHW 功率（P_in > 0）且 mtanh/NN 结果完整的时间点，取功率最高者
    best = None
    best_mtanh = None
    for entry in entries:
        if entry['p_in_kw'] <= 0:
            continue
        mtanh_r = compute_lhw_for_method(entry['shot'], entry, 'mtanh')
        if mtanh_r is None:
            continue
        nn_r = compute_lhw_for_method(entry['shot'], entry, 'nn')
        if nn_r is None:
            continue
        if best is None or entry['p_in_kw'] > best['p_in_kw']:
            best = entry
            best_mtanh = mtanh_r
    if best is None:
        print('No complete LHW entry for fig9')
        return
    entry = best
    mtanh_r = best_mtanh

    shot = entry['shot']
    td = entry['time_dir']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    fig.suptitle(f'LHW Current Drive & Power Deposition — Shot {shot} {td}s\n'
                 f'NN vs mtanh Comparison',
                 fontsize=12, fontweight='bold')

    rho = mtanh_r['rho']
    rho_lower = np.linspace(0, 1, 51)
    from scipy.interpolate import CubicSpline
    from scipy.signal import savgol_filter

    # Panel 1: 电流密度
    ax = axes[0, 0]
    for method in ['mtanh'] + NN_METHODS:
        ms = METHOD_SPECS[method]
        result = compute_lhw_for_method(shot, entry, method)
        if result is None:
            continue
        j_smooth = savgol_filter(CubicSpline(result['rho'], result['j_lhcd_am2'])(rho_lower), 11, 3)
        ax.plot(rho_lower, j_smooth * 1e-4, color=ms['color'], ls=ms['ls'], lw=ms['lw'],
                label=ms['label'])
    ax.set_xlabel(r'$\rho$')
    ax.set_ylabel('Current Density (A/cm²)')
    ax.set_title('LHCD Current Density')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 2: 功率密度
    ax = axes[1, 0]
    for method in ['mtanh'] + NN_METHODS:
        ms = METHOD_SPECS[method]
        result = compute_lhw_for_method(shot, entry, method)
        if result is None:
            continue
        p_smooth = savgol_filter(CubicSpline(result['rho'], result['p_lh_wm3'])(rho_lower), 11, 3)
        ax.plot(rho_lower, p_smooth / 1e6, color=ms['color'], ls=ms['ls'], lw=ms['lw'],
                label=ms['label'])
    ax.set_xlabel(r'$\rho$')
    ax.set_ylabel('Power Density (MW/m³)')
    ax.set_title('LHCD Power Deposition')
    ax.grid(alpha=0.3)

    # Panel 3: 总电流偏差
    ax = axes[0, 1]
    I_mtanh = mtanh_r['I_total_kA']
    methods_labels = []
    I_vals = []
    I_colors = []
    for method in NN_METHODS:
        result = compute_lhw_for_method(shot, entry, method)
        if result is None:
            continue
        methods_labels.append(METHOD_SPECS[method]['label'])
        I_vals.append(result['I_total_kA'])
        I_colors.append(METHOD_SPECS[method]['color'])
    x_pos = np.arange(len(methods_labels))
    bars = ax.bar(x_pos, [abs(v - I_mtanh) for v in I_vals], color=I_colors, alpha=0.7)
    ax.axhline(y=I_mtanh, color=METHOD_SPECS['mtanh']['color'], ls='--', lw=2,
               label=f'mtanh: {I_mtanh:.1f} kA')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods_labels, rotation=15, fontsize=9)
    ax.set_ylabel('Total Driven Current (kA)')
    ax.set_title('Total LHCD Current')
    ax.legend(fontsize=8)

    # Panel 4: 电流剖面差异（NN - mtanh）
    ax = axes[1, 1]
    for method in NN_METHODS:
        ms = METHOD_SPECS[method]
        result = compute_lhw_for_method(shot, entry, method)
        if result is None:
            continue
        from scipy.interpolate import interp1d as i1d
        j_nn = i1d(result['rho'], result['j_lhcd_am2'],
                    bounds_error=False, fill_value='extrapolate')(rho)
        delta_j = (j_nn - mtanh_r['j_lhcd_am2']) * 1e-4  # A/cm²
        ax.plot(rho, delta_j, color=ms['color'], ls=ms['ls'], lw=ms['lw'], label=ms['label'])
    ax.axhline(y=0, color='gray', ls='-', lw=0.5)
    ax.set_xlabel(r'$\rho$')
    ax.set_ylabel('Δ Current Density (A/cm²)')
    ax.set_title('NN − mtanh Current Density Deviation')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(FIG_DIR, 'fig9_lhw_comparison.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  fig9_lhw_comparison.png saved')


if __name__ == '__main__':
    main()
