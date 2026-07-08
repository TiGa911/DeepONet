# -*- coding: utf-8 -*-
"""
lower_view_final_nn.py — LHW 功率沉积与电流驱动计算（NN 剖面版本）

与 lower_view_final.py 的区别：
  - 接受外部传入的 NN 拟合剖面（te_profile_nn, ne_profile_nn）
  - 当提供 NN 剖面时，跳过内部 mtanh 拟合，直接使用 NN 结果
  - 添加 Te/ne 安全裁剪，防止负值导致 sqrt/exp NaN
  - 其余逻辑（gfile 加载、几何计算、ONETWO 调用）与原始完全相同

原版 lower_view_final.py 不受影响，仍用于 onetwo_output.py 的 mtanh 流程。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, m_e, epsilon_0, pi, c, m_p
from scipy.interpolate import CubicSpline, interp1d
from scipy.signal import savgol_filter
from scipy.integrate import cumulative_trapezoid
from MDSplus.connection import Connection
# === 原始版本（保留参考）===
# from readMDS import readmds

# === 修改版：统一使用 ONETWO 专用读取器（含 gfile 自动保存 + Refl/Zeff 诊断）===
from readMDS_onetwo import readmds
from fitting_mtanh import fitting
from mesh2 import g_to_23
import geqdsk
import os


# ====================================================
# 核心函数：计算并保存 LHCD 功率沉积与电流驱动结果
# ====================================================
def lower_onetwo_nn(shot, time, onetwo_save_path,
                    te_profile_nn=None, ne_profile_nn=None,
                    te_rho_nn=None, ne_rho_nn=None,
                    # === 离线模式参数 ===
                    gfile_path=None, real_time=None, z_eff=None,
                    p_in_kw=None):
    """
    计算并保存 EAST 托卡马克的 LHCD 功率沉积与电流驱动剖面。

    支持 3 种模式：
      1. NN 在线模式：传入 te/ne_profile_nn，从 MDSplus 读取 Zeff 和 gfile
      2. NN 离线模式：传入 te/ne_profile_nn + gfile_path + real_time + z_eff
      3. 传统模式（fallback）：不传 NN 剖面，使用 mtanh 拟合

    参数:
        shot : int           放电号
        time : float         时间点（s）
        onetwo_save_path : str  输出保存路径
        te_profile_nn : ndarray or None  NN Te 剖面 [keV]
        ne_profile_nn : ndarray or None  NN ne 剖面 [1e19 m^-3]
        te_rho_nn : ndarray or None     Te 的 rho 网格
        ne_rho_nn : ndarray or None     ne 的 rho 网格
        gfile_path : str or None        离线模式：gfile 文件路径
        real_time : float or None       离线模式：gfile 实际时间
        z_eff : float or None           离线模式：有效电荷数
        p_in_kw : float or None         离线模式：net LH power [kW]（None=从MDSplus读取）

    返回:
        rho_lower, p_LH(MW/m^3), j_LHCD(MA/m^2)
    """

    # -------------------------------
    # === Step 1. 数据加载与拟合 ===
    # -------------------------------
    use_nn = (te_profile_nn is not None and ne_profile_nn is not None)
    offline_mode = (gfile_path is not None)

    if use_nn:
        # ---- NN 路径：跳过 mtanh 拟合，直接使用传入的 NN 剖面 ----
        if offline_mode:
            z_eff_value = z_eff if z_eff is not None else 2.0
        else:
            data, status, real_time = readmds(shot, time)
            z_eff_value = data.get('zeff', 2.0)

        # 将 NN 剖面插值到统一的 rho 网格（nh = gfile['nw'] 点）
        x_Te = te_rho_nn if te_rho_nn is not None else np.linspace(0, 1, len(te_profile_nn))
        y_Te = np.asarray(te_profile_nn, dtype=float)
        x_ne = ne_rho_nn if ne_rho_nn is not None else np.linspace(0, 1, len(ne_profile_nn))
        y_ne = np.asarray(ne_profile_nn, dtype=float)
    else:
        # ---- 传统路径：mtanh 拟合（与原版 lower_view_final.py 完全一致） ----
        data, status, real_time = readmds(shot, time)
        z_eff_value = data.get('zeff', 2.0)

        # Te 剖面拟合
        x_Te, y_Te, _ = data['Te']['TS']['Rho'], data['Te']['TS']['data'], data['Te']['TS']['type']
        x_Te, y_Te, _, _, _ = fitting(x_Te, y_Te, 'TS')

        # ne 剖面拟合
        x_ne, y_ne, _ = data['ne']['Refl']['Rho'], data['ne']['Refl']['data'], data['ne']['Refl']['type']
        x_ne, y_ne, _, _, _ = fitting(x_ne, y_ne, 'Refl')

    # 平衡文件加载（按时间最近匹配，解决 TS/EFIT 时间精度不一致问题）
    def _find_gfile(shot, time):
        """按时间最近匹配查找 gfile 文件

        查找顺序：
        1. 精确匹配: {shot}_{time}_gfile
        2. 同炮号模糊匹配: 该炮号下时间最近的 gfile
        3. 跨炮号搜索: 当前目录下所有 gfile，按时间最近匹配（回退策略）

        Raises:
            FileNotFoundError: 三种策略均未找到 gfile
        """
        gfile_exact = f'{shot}_{time}_gfile'
        if os.path.exists(gfile_exact):
            return gfile_exact

        # 策略 2: 同炮号模糊匹配
        import glob as _glob
        candidates = _glob.glob(f'{shot}_*_gfile')
        if candidates:
            best = None
            best_dt = float('inf')
            for gf in candidates:
                try:
                    t_str = os.path.basename(gf).replace(f'{shot}_', '', 1).rsplit('_gfile', 1)[0]
                    if not t_str or t_str == os.path.basename(gf):
                        continue
                    t_val = float(t_str)
                    dt = abs(t_val - time)
                    if dt < best_dt:
                        best_dt = dt
                        best = gf
                except (ValueError, IndexError):
                    continue
            if best is not None:
                print(f'  gfile: 同炮号匹配 shot={shot} t={best_dt*1000:.0f}ms 偏差')
                return best

        # 策略 3: 跨炮号搜索（所有可用 gfile）
        all_gfiles = _glob.glob('*_*_gfile')
        if all_gfiles:
            best = None
            best_dt = float('inf')
            for gf in all_gfiles:
                try:
                    basename = os.path.basename(gf)
                    # 格式: {shot}_{time}_gfile
                    parts = basename.rsplit('_gfile', 1)[0].split('_', 1)
                    if len(parts) < 2:
                        continue
                    t_val = float(parts[1])
                    dt = abs(t_val - time)
                    if dt < best_dt:
                        best_dt = dt
                        best = gf
                        best_shot = parts[0]
                except (ValueError, IndexError):
                    continue
            if best is not None:
                print(
                    f'  ⚠ gfile: 跨炮号回退！使用 shot={best_shot} 的 gfile '
                    f'(时间偏差 {best_dt*1000:.0f}ms) — 平衡位形可能不同'
                )
                return best

        raise FileNotFoundError(
            f'No gfile found for shot {shot} near time {time}. '
            f'Tried: exact match, same-shot fuzzy, cross-shot fallback — all failed. '
            f'This shot likely has no EFIT equilibrium data. '
            f'Run: ls *_gfile  to see available gfile cache.'
        )

    # ---- 离线模式：使用传入的 gfile_path；在线模式：自动搜索 ----
    if offline_mode:
        _gfile_path = gfile_path
        if real_time is not None:
            gfile_time = real_time
        else:
            gfile_time = time
    else:
        _gfile_path = _find_gfile(shot, time)
        gfile_time = time
    gdate = geqdsk.load(_gfile_path)
    rho = np.linspace(0, 1, gdate['nw'])
    ne_profile = np.interp(rho, x_ne, y_ne)
    te_profile = np.interp(rho, x_Te, y_Te)

    # ---- 安全裁剪：防止负值导致 lower_view 中 sqrt(负数)→NaN ----
    te_profile = np.clip(te_profile, 0.001, None)   # ≥ 1 eV
    ne_profile = np.clip(ne_profile, 0.0, None)

    # q(ρ)
    qpsi = gdate.get('qpsi', np.ones_like(rho))
    psinorm = np.linspace(0, 1, len(rho))
    theta = np.linspace(0, 2 * np.pi, len(rho))
    grid = g_to_23(_gfile_path, psinorm, theta)

    dvdpsi = np.asarray(grid['dvdpsi'], dtype=float)
    phi = np.asarray(grid['phi'], dtype=float)
    npsi = len(dvdpsi)
    phi0, phi_sep = phi[0], phi[-1]
    rho_psi = np.sqrt((phi - phi0) / (phi_sep - phi0 + 1e-30))
    psia, psib = gdate['simag'], gdate['sibry']
    psi_array = psia + np.linspace(0, 1, npsi) * (psib - psia)
    dpsi_drho = np.gradient(psi_array, rho_psi)
    Vrho_on_psi = dvdpsi * dpsi_drho
    f_Vrho = interp1d(rho_psi, Vrho_on_psi, bounds_error=False, fill_value="extrapolate")
    Vrho_on_rho = f_Vrho(rho)

    # === Step 2. 获取输入功率 ===
    if p_in_kw is not None:
        # 离线模式：使用外部提供的功率值
        PLH2 = float(p_in_kw)  # kW
    else:
        # 在线模式：从 MDSplus 读取
        conn = Connection('202.127.204.12')
        try:
            conn.openTree('EAST', shot)
            PLHI2 = conn.get(r'data(\PLHI2)').data()
            PLHI2_times = conn.get(r'dim_of(\PLHI2)').data()
            PLHR2 = conn.get(r'data(\PLHR2)').data()
            PLHR2_times = conn.get(r'dim_of(\PLHR2)').data()
            conn.closeTree('EAST', shot)
            mask1 = (PLHI2_times >= time - 0.1) & (PLHI2_times <= time)
            mask2 = (PLHR2_times >= time - 0.1) & (PLHR2_times <= time)
            PLH2 = np.mean(PLHI2[mask1]) - np.mean(PLHR2[mask2])
        except Exception:
            PLH2 = 0.0

    # === Step 3. 几何参数 ===
    filtered_data = gdate['bbbsrz'][gdate['bbbsrz'][:, 0] != 0]
    bbbsrz = filtered_data[:, 0]
    a_R = (np.max(bbbsrz) + np.min(bbbsrz)) / 2.0
    a_r = (np.max(bbbsrz) - np.min(bbbsrz)) / 2.0
    invR_on_rho = np.full_like(rho, 1.0 / a_R)

    # === Step 4. 调用模型 ===
    from lower_view import lower_hybrid_power_deposition

    params = dict(
        rho=rho, Te_keV=te_profile, ne_m3=ne_profile * 1e19,
        n_parallel0=2.0, delta_a=0.4, q=np.interp(rho, rho, qpsi),
        R_axis=a_R, a=a_r, B0=abs(gdate.get("bcentr", 0.0)),
        frequency=4.6e9, Z_eff=z_eff_value, main_ion_mass=2.0,
        P_in=PLH2 * 1e3, Vrho=Vrho_on_rho
    )

    p_LH, j_LHCD = lower_hybrid_power_deposition(**params)

    # === Step 5. 处理结果 ===
    rho_lower = np.linspace(0, 1, 51)
    p_LH_smooth = savgol_filter(CubicSpline(rho, p_LH)(rho_lower), 11, 3)
    j_LHCD_smooth = savgol_filter(CubicSpline(rho, j_LHCD)(rho_lower), 11, 3)

    # 单位换算
    p_plot_MWpm3 = p_LH_smooth / 1e6
    j_plot_MApm2 = j_LHCD_smooth / 1e6

    # === Step 6. 写入txt ===
    time_str = str(time).replace('.', 'd')
    output_filename = f"LHCD_data_{shot}_{time_str}.txt"
    save_path = os.path.join(onetwo_save_path, output_filename)
    with open(save_path, 'w') as f:
        f.write("rho\tPower_Density(MW/m3)\tCurrent_Density(MA/m2)\n")
        for i in range(len(rho_lower)):
            f.write(f"{rho_lower[i]:.6f}\t{p_plot_MWpm3[i]:.6e}\t{j_plot_MApm2[i]:.6e}\n")
    print(f"数据已保存至: {save_path}")

    # === Step 7. 保存图片 ===
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(rho_lower, j_plot_MApm2, 'b-', lw=2)
    axs[0].set_xlabel(r'$\rho$')
    axs[0].set_ylabel('Current Density (MA/m²)')
    axs[0].set_title('LHCD Current Drive')
    axs[1].plot(rho_lower, p_plot_MWpm3, 'r-', lw=2)
    axs[1].set_xlabel(r'$\rho$')
    axs[1].set_ylabel('Power Density (MW/m³)')
    axs[1].set_title('LHCD Power Deposition')
    plt.tight_layout()
    fig.savefig(os.path.join(onetwo_save_path, f"LHCD_plot_{shot}_{time_str}.png"))
    plt.close(fig)

    print(f"图像已保存至: LHCD_plot_{shot}_{time_str}.png")

    return rho_lower, p_plot_MWpm3, j_plot_MApm2
