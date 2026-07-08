# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, m_e, epsilon_0, pi, c, m_p
from scipy.interpolate import CubicSpline, interp1d
from scipy.signal import savgol_filter
from scipy.integrate import cumulative_trapezoid
from MDSplus.connection import Connection
# === 原始版本（保留参考）===
# from readMDS import readmds

# === 修改版：统一使用 ONETWO 专用读取器（含 gfile 自动保存）===
from readMDS_onetwo import readmds
from fitting_mtanh import fitting
from mesh2 import g_to_23
import geqdsk
import os


# ====================================================
# 核心函数：计算并保存 LHCD 功率沉积与电流驱动结果
# ====================================================
def lower_onetwo(shot, time, onetwo_save_path):
    """
    计算并保存 EAST 托卡马克的 LHCD 功率沉积与电流驱动剖面。
    参数:
        shot : int
            放电号
        time : float
            时间点（s）
        onetwo_save_path : str
            输出保存路径

    返回:
        rho_lower, p_LH(MW/m^3), j_LHCD(MA/m^2)
    """

    # -------------------------------
    # === Step 1. 数据加载与拟合 ===
    # -------------------------------
    data, status, real_time = readmds(shot, time)
    z_eff_value = data.get('zeff', 2.0)

    # Te 剖面拟合
    x_Te, y_Te, _ = data['Te']['TS']['Rho'], data['Te']['TS']['data'], data['Te']['TS']['type']
    x_Te, y_Te, _, _, _ = fitting(x_Te, y_Te, 'TS')

    # ne 剖面拟合
    x_ne, y_ne, _ = data['ne']['Refl']['Rho'], data['ne']['Refl']['data'], data['ne']['Refl']['type']
    x_ne, y_ne, _, _, _ = fitting(x_ne, y_ne, 'Refl')

    # 平衡文件加载
    # === 原始版本（保留参考）===
    # gdate = geqdsk.load(f'{shot}_{time}_gfile')

    # === 修改版：使用 gfile 实际时间 real_time（readMDS_onetwo 以此命名保存）===
    gdate = geqdsk.load(f'{shot}_{real_time}_gfile')
    rho = np.linspace(0, 1, gdate['nw'])
    ne_profile = np.interp(rho, x_ne, y_ne)
    te_profile = np.interp(rho, x_Te, y_Te)

    # === 安全裁剪：防止负值/NaN 导致 lower_view 中 sqrt(负数)→NaN ===
    # mtanh 拟合在边界区域可能产生负值或 NaN，此处做防御性处理
    # （NN 版本 lower_view_final_nn.py 已有相同保护）
    te_profile = np.nan_to_num(te_profile, nan=0.5, posinf=10.0, neginf=0.001)
    ne_profile = np.nan_to_num(ne_profile, nan=1.0, posinf=50.0, neginf=0.0)
    te_profile = np.clip(te_profile, 0.001, None)   # Te ≥ 1 eV（防止 sqrt(Te)→NaN）
    ne_profile = np.clip(ne_profile, 0.0, None)     # ne ≥ 0

    # q(ρ)
    qpsi = gdate.get('qpsi', np.ones_like(rho))
    psinorm = np.linspace(0, 1, len(rho))
    theta = np.linspace(0, 2 * np.pi, len(rho))
    # === 原始版本（保留参考）===
    # grid = g_to_23(f"{shot}_{time}_gfile", psinorm, theta)

    # === 修改版：使用 gfile 实际时间 ===
    grid = g_to_23(f"{shot}_{real_time}_gfile", psinorm, theta)

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
    from lower_view import lower_hybrid_power_deposition  # 直接引用你的模型函数

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



# shot = 63948
# time = 6
# onetwo_save_path = r"F:\code\6_11基本完成\results\test"

# rho, p_LH, j_LHCD = lower_onetwo(shot, time, onetwo_save_path)

