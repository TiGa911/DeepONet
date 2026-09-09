# -*- coding: utf-8 -*-
"""
lhw_decompose.py — LHW 24% 偏差来源分解（审稿意见 M5）

对每个 LHW-ready 时间点，分别以 (mtanh_Te, mtanh_ne)、(X_Te, X_ne)、
(mtanh_Te, X_ne)、(X_Te, mtanh_ne) 四种组合计算总驱动电流 I_LH，
将 ΔI 分解为 Te 贡献 + ne 贡献 + 交叉项。

用法:
  python lhw_decompose.py [--shot 156400]
"""
import argparse
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from compute_lhw_compare import (
    N_PARALLEL0, DELTA_A, FREQUENCY, MAIN_ION_MASS,
    load_profile, discover_lhw_entries,
)

parser = argparse.ArgumentParser()
parser.add_argument('--shot', type=int, default=None)
args = parser.parse_args()

NN_METHODS = ['nn', 'lstm', 'cnn', 'cnn_deeponet', 'transformer']


def lhw_pair(entry, te_method, ne_method):
    """用 te_method 的 Te 剖面 + ne_method 的 ne 剖面计算 I_LH。"""
    import geqdsk
    from mesh2 import g_to_23
    from lower_view import lower_hybrid_power_deposition
    from scipy.interpolate import interp1d

    rho_te, te_kev = load_profile(entry['shot'], entry['time_dir'], te_method, 'Te')
    rho_ne, ne_1e19 = load_profile(entry['shot'], entry['time_dir'], ne_method, 'ne')
    if rho_te is None or rho_ne is None:
        return None

    gdate = geqdsk.load(entry['gfile_path'])
    rho = np.linspace(0, 1, gdate['nw'])
    te_on_rho = interp1d(rho_te, te_kev, bounds_error=False, fill_value='extrapolate')(rho)
    ne_on_rho = interp1d(rho_ne, ne_1e19, bounds_error=False, fill_value='extrapolate')(rho)
    ne_m3 = np.clip(ne_on_rho * 1e19, 0, None)

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

    filtered = gdate['bbbsrz'][gdate['bbbsrz'][:, 0] != 0]
    bbbsrz = filtered[:, 0]
    R_axis = (np.max(bbbsrz) + np.min(bbbsrz)) / 2.0
    a = (np.max(bbbsrz) - np.min(bbbsrz)) / 2.0
    qpsi = gdate.get('qpsi', np.ones_like(rho))
    q_on_rho = interp1d(np.linspace(0, 1, len(qpsi)), qpsi,
                        bounds_error=False, fill_value='extrapolate')(rho)

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
        P_in=entry['p_in_kw'] * 1e3,
        Vrho=Vrho_on_rho,
    )
    I_total_A = np.trapz(j_lhcd * Vrho_on_rho, rho) / (2 * np.pi * R_axis)
    return I_total_A / 1e3  # kA


def main():
    entries = discover_lhw_entries()
    if args.shot is not None:
        entries = [e for e in entries if e['shot'] == args.shot]
    print(f'{len(entries)} LHW-ready time points')
    if not entries:
        return

    # 逐点分解
    rows = []
    for e in entries:
        I_mm = lhw_pair(e, 'mtanh', 'mtanh')
        if I_mm is None:
            continue
        for m in NN_METHODS:
            I_xx = lhw_pair(e, m, m)
            I_mx = lhw_pair(e, 'mtanh', m)   # 仅 ne 换
            I_xm = lhw_pair(e, m, 'mtanh')   # 仅 Te 换
            if I_xx is None or I_mx is None or I_xm is None:
                continue
            d_total = (I_xx - I_mm) / I_mm * 100
            d_te = (I_xm - I_mm) / I_mm * 100    # Te 单独贡献
            d_ne = (I_mx - I_mm) / I_mm * 100    # ne 单独贡献
            d_cross = d_total - d_te - d_ne
            rows.append({
                'shot': e['shot'], 'time_dir': e['time_dir'], 'method': m,
                'I_mtanh': I_mm, 'I_nn': I_xx,
                'd_total%': d_total, 'd_te%': d_te, 'd_ne%': d_ne, 'd_cross%': d_cross,
            })

    shots = sorted(set(r['shot'] for r in rows))
    for shot in shots:
        rs = [r for r in rows if r['shot'] == shot]
        print(f'\n=== shot {shot} ({len(rs)} method-point rows) ===')
        for r in rs:
            print(f"  {r['time_dir']} {r['method']:>13s}: I_mtanh={r['I_mtanh']:7.2f} kA  "
                  f"I_nn={r['I_nn']:7.2f} kA  Δ={r['d_total%']:+7.1f}%  "
                  f"(Te {r['d_te%']:+6.1f}% + ne {r['d_ne%']:+6.1f}% + cross {r['d_cross%']:+6.1f}%)")
    print('\n=== 汇总（|Δ| 平均，按炮） ===')
    for shot in shots:
        rs = [r for r in rows if r['shot'] == shot]
        print(f"shot {shot}: |Δ_total|={np.mean([abs(r['d_total%']) for r in rs]):.1f}%  "
              f"|Te|={np.mean([abs(r['d_te%']) for r in rs]):.1f}%  "
              f"|ne|={np.mean([abs(r['d_ne%']) for r in rs]):.1f}%  "
              f"|cross|={np.mean([abs(r['d_cross%']) for r in rs]):.1f}%")


if __name__ == '__main__':
    main()
