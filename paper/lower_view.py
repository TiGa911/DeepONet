# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, m_e, epsilon_0, pi, c, m_p
from scipy import interpolate
# === 原始版本（保留参考）===
# from readMDS import readmds

# === 修改版：paper/ 目录独立部署 ===
from readMDS_onetwo import readmds
from fitting_mtanh import fitting
import geqdsk
from MDSplus.connection import Connection
from mesh2 import g_to_23
from scipy.interpolate import CubicSpline, interp1d
from scipy.signal import savgol_filter
from scipy.integrate import cumulative_trapezoid
import os

# -------------------------
# 你原来的辅助函数（保留）
# -------------------------
def calc_total_current(j_prof, rho, Vprime_rho, R0):
    # 输入 j_prof 单位为 A/cm^2 -> convert to A/m^2
    j_prof_SI = j_prof * 1e4
    # 注意：这里用的是近似表达 I = ∫ j * V_rho * dρ / (2π R0)
    I_total = np.trapz(j_prof_SI * Vprime_rho, rho) / (2 * np.pi * R0)
    return I_total

def calc_total_power(p_prof, rho, Vprime_rho):
    # p_prof 单位为 MW/m^3 -> convert to W/m^3
    p_prof_SI = p_prof * 1e6
    P_total = np.trapz(p_prof_SI * Vprime_rho, rho)   # W
    return P_total

# -------------------------
# lower_hybrid_power_deposition（你原函数，保持不改主要式子）
# 返回：
#   P_abs_normalized: Array, 单位 W/m^3 (在内部用 P_in (W) 归一化）
#   j_LHCD:           Array, 单位 A/m^2
# -------------------------
def lower_hybrid_power_deposition(
    rho: np.ndarray,
    Te_keV: np.ndarray,
    ne_m3: np.ndarray,
    n_parallel0: float,
    delta_a: float,
    R_axis: float,
    a: float,
    q: np.ndarray,
    B0: float,
    frequency: float,
    Z_eff: np.ndarray,
    main_ion_mass: float,
    P_in: float,
    Vrho: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # 尺度与物理量处理（保持你原实现，仅注意 Vrho 被视作数组，与 rho 对应）
    Te_eV = Te_keV * 1e3
    r = a * rho

    omega = 2 * pi * frequency
    delta_n_parallel0 = (c / (frequency * delta_a))

    omega_pe = np.sqrt(ne_m3 * e**2 / (m_e * epsilon_0))
    omega_ce = e * B0 / m_e
    omega_pi = np.sqrt(ne_m3 * e**2 / (main_ion_mass * m_p * epsilon_0))

    p = 1 - ((omega_pe**2 / omega**2) + (omega_pi**2 / omega**2))
    s = 1 + (omega_pe**2 / omega_ce**2) - (omega_pi**2 / omega**2)

    sqrt_arg = 1 + (omega_pe**2 / omega_ce**2) - (omega_pi**2 / omega**2)
    n_parallel_acc = (omega_pe / omega_ce) + np.sqrt(np.clip(sqrt_arg, 0.0, None))

    n_Landau = 6.5 / np.sqrt(1e-3 * Te_eV + 1e-30)

    delta_n_parallel = (
        (delta_n_parallel0 + n_parallel0 * delta_a / R_axis) *
        (1 + n_Landau / n_parallel0)
    )

    sqrt_arg2 = -p / (s + 1e-30)
    sqrt_term = np.sqrt(np.clip(sqrt_arg2, 0.0, None))

    # 防止 q 中的零/负值传播——clip q
    q_safe = np.maximum(q, 1e-6)

    denominator_lc = 1 + (rho / (q_safe * R_axis)) * sqrt_term
    n_parallel_lc = n_parallel0 / np.clip(denominator_lc, 1e-12, None)

    denominator_hc = 1 - (rho / (q_safe * R_axis)) * sqrt_term
    n_parallel_hc = n_parallel0 / np.clip(denominator_hc, 1e-12, None)

    exponent = (n_Landau - n_parallel0 - 0.5 * delta_n_parallel) / np.clip(delta_n_parallel, 1e-12, None)
    P_Landau = np.exp(-exponent**2)

    L_acc = np.minimum(1.0, np.exp((n_Landau - n_parallel_acc - delta_n_parallel0) / (delta_n_parallel0 + 1e-30)))
    L_lc = np.minimum(1.0, np.exp((n_Landau - (n_parallel_lc + 0.5*delta_n_parallel)) / (delta_n_parallel + 1e-30)))
    L_hc = np.minimum(1.0, np.exp(-(n_Landau - (n_parallel_hc - 0.5*delta_n_parallel)) / (delta_n_parallel + 1e-30)))

    P_abs = P_Landau * L_acc * L_lc * L_hc

    # 功率归一化：确保 Vrho 与 rho 长度一致
    Vrho = np.asarray(Vrho, dtype=float)
    if Vrho.size != rho.size:
        # 插值 Vrho 到 rho 网格（若传入 Vrho 长度不同）
        ftmp = interp1d(np.linspace(0,1,Vrho.size), Vrho, bounds_error=False, fill_value="extrapolate")
        Vrho_on_rho = ftmp(rho)
    else:
        Vrho_on_rho = Vrho

    integral = np.trapz(P_abs * Vrho_on_rho, rho)
    if integral <= 0 or np.isnan(integral):
        raise RuntimeError(f"吸收概率积分异常 integral={integral:.3e}")

    P_abs_normalized = P_in * P_abs / integral  # W/m^3

    # 电流驱动效率部分（保持原式）
    ln_Lambda = 14.9 + 0.5 * np.log(np.maximum(ne_m3/1e20, 1e-30)) + np.log(np.maximum(Te_eV/1e3, 1e-30))
    eta_0 = 3.1e21 * np.sqrt(np.maximum(Te_eV/1e3, 1e-30)) / np.maximum(ln_Lambda, 1e-30)

    D_parallel = 0.32 * (P_in/1e6) * (n_Landau**2) * np.sqrt(np.maximum(Te_eV/1e3, 1e-30))
    D_parallel /= (np.sqrt(np.maximum(s, 1e-12)) * np.maximum(delta_n_parallel, 1e-12) * np.maximum(delta_a, rho) * R_axis * (np.maximum(ne_m3, 1e-30)/1e19)**1.5)

    Q_1 = 1.5 + 0.5 * np.tanh(np.log(np.maximum(10.0 * D_parallel, 1e-30)))
    eta_acc = np.minimum(1.0, np.exp((n_parallel0 - np.max(n_parallel_acc)) / (delta_n_parallel0 + 1e-30)))

    omega1 = 1.0 / np.maximum(n_Landau, 1e-30)
    omega2 = 1.0 / np.maximum(n_parallel0, 1e-30)
    # 防止 log(omega2/omega1) 为 0
    denom_log = np.log(np.maximum(omega2/omega1, 1e-30))
    eta_LHCD = eta_0 * eta_acc * Q_1 * (omega2**2 - omega1**2) / np.maximum(denom_log, 1e-30)

    epsilon = r / np.maximum(R_axis, 1e-30)
    z_ratio = (5.0 + Z_eff) / (2.0 * (1.0 + Z_eff))
    trapping_factor = (1.0 - np.power(epsilon, z_ratio)) / (5.0 + Z_eff + 1e-30)

    j_LHCD = eta_LHCD * (P_abs_normalized / np.maximum(ne_m3, 1e-30)) * trapping_factor  # A/m^2

    return P_abs_normalized, j_LHCD


# ===========================
# 主程序入口（保持你原流程）
# ===========================
if __name__ == '__main__':
    shot = int(input('shot: '))
    time = float(input('time: '))
    # 读诊断
    data, status, real_time = readmds(shot, time)
    z_eff_value = data.get('zeff', 2.0)
    print('zeff:::', z_eff_value)

    # Te 剖面拟合
    x_Te, y_Te, _ = data['Te']['TS']['Rho'], data['Te']['TS']['data'], data['Te']['TS']['type']
    x_Te, y_Te, _, _, q_e = fitting(x_Te, y_Te, 'TS')

    # ne 剖面拟合
    x_ne, y_ne, _ = data['ne']['Refl']['Rho'], data['ne']['Refl']['data'], data['ne']['Refl']['type']
    x_ne, y_ne, _, _, q_e = fitting(x_ne, y_ne, 'Refl')

    # 平衡文件加载
    gdate = geqdsk.load(f'{shot}_{time}_gfile')
    # 使用 gdate['nw'] 作为 rho 网格长度
    rho = np.linspace(0, 1, gdate['nw'])
    ne_profile = np.interp(rho, x_ne, y_ne)  # 注意：这里 y_ne 单位需与你前面的处理一致；若原来是相对数请乘以 1e19
    te_profile = np.interp(rho, x_Te, y_Te)  # keV

    # q(ψ) -> q(ρ)
    qpsi = gdate.get('qpsi', np.ones_like(rho))
    psinorm = np.linspace(0, 1, len(rho))
    theta = np.linspace(0, 2 * np.pi, len(rho))
    grid = g_to_23(f"{shot}_{time}_gfile", psinorm, theta)

    # Vrho 计算（在 psi 网格上）
    dvdpsi = np.asarray(grid['dvdpsi'], dtype=float)
    phi = np.asarray(grid['phi'], dtype=float)
    npsi = len(dvdpsi)

    # 构造 rho_psi (确保和 dvdpsi 对应)
    phi0 = phi[0]
    phi_sep = phi[-1]
    rho_psi = np.sqrt((phi - phi0) / (phi_sep - phi0 + 1e-30))

    # psi_array 对应 dvdpsi
    psia = gdate['simag']
    psib = gdate['sibry']
    psi_array = psia + np.linspace(0, 1, npsi) * (psib - psia)

    # dψ/dρ
    dpsi_drho = np.gradient(psi_array, rho_psi)

    # Vrho 在 psi 网格
    Vrho_on_psi = dvdpsi * dpsi_drho

    # 将 Vrho 插值到 rho 网格（如果 rho 与 rho_psi 长度不一致）
    f_Vrho = interp1d(rho_psi, Vrho_on_psi, bounds_error=False, fill_value="extrapolate")
    Vrho_on_rho = f_Vrho(rho)  # 与 'rho' 对齐

    # 读取并计算低杂净功率（你原来那段）
    conn = Connection('202.127.204.12')
    try:
        conn.openTree('EAST', shot)
        PLHI2 = conn.get(r'data(\PLHI2)').data()
        PLHI2_times = conn.get(r'dim_of(\PLHI2)').data()
        window_start = time - 0.1
        window_end = time
        mask = (PLHI2_times >= window_start) & (PLHI2_times <= window_end)
        PLHI2_in = np.mean(PLHI2[mask]) if mask.any() else 0.0

        PLHR2 = conn.get(r'data(\PLHR2)').data()
        PLHR2_times = conn.get(r'dim_of(\PLHR2)').data()
        mask2 = (PLHR2_times >= window_start) & (PLHR2_times <= window_end)
        PLHR2_in = np.mean(PLHR2[mask2]) if mask2.any() else 0.0

        conn.closeTree('EAST', shot)
        PLH2 = PLHI2_in - PLHR2_in  # kW
        print(f"Net Power at {time}s: {PLH2} kW")
    except Exception as e:
        print(f"Error reading power from MDSplus: {e}")
        PLH2 = 0.0

    # 计算几何参数
    filtered_data = gdate['bbbsrz'][gdate['bbbsrz'][:, 0] != 0]
    bbbsrz = filtered_data[:, 0]
    a_R = (np.max(bbbsrz) + np.min(bbbsrz)) / 2.0   # 近似磁轴大半径 R_axis
    a_r = (np.max(bbbsrz) - np.min(bbbsrz)) / 2.0   # 近似小半径 a
    invR_on_psi = np.full_like(rho_psi, 1.0 / a_R)  # fallback invR on psi grid
    # 插值到 rho 网格
    f_invR = interp1d(rho_psi, invR_on_psi, bounds_error=False, fill_value="extrapolate")
    invR_on_rho = f_invR(rho)

    # 准备模型参数并调用 LHW 模型
    params = {
        "rho": rho,
        "Te_keV": te_profile,
        "ne_m3": ne_profile * 1e19,   # 如果 y_ne 原来是 in units of 1 (i.e., 1e19 scale)，请调整，这里假设需要 *1e19
        "n_parallel0": 2.2,
        "delta_a": 0.15,
        "q": np.interp(rho, rho_psi, qpsi),
        "R_axis": a_R,
        "a": a_r,
        "B0": abs(gdate.get("bcentr", 0.0)),
        "frequency": 4.6e9,
        "Z_eff": z_eff_value,
        "main_ion_mass": 2.0,
        "P_in": PLH2 * 1e3,   # kW -> W
        "Vrho": Vrho_on_rho
    }

    p_LH, j_LHCD = lower_hybrid_power_deposition(**params)  # p_LH: W/m^3, j_LHCD: A/m^2

    # 重采样到更稀疏的 rho_lower 网格用于平滑与绘图
    rho_lower = np.linspace(0, 1, 51)
    func_p_LH = CubicSpline(rho, p_LH)
    func_j_LHCD = CubicSpline(rho, j_LHCD)

    p_LH_lower = func_p_LH(rho_lower)   # W/m^3
    j_LHCD_lower = func_j_LHCD(rho_lower)  # A/m^2

    # 滤波平滑
    p_LH_smooth = savgol_filter(p_LH_lower, window_length=11, polyorder=3, mode='interp')
    j_LHCD_smooth = savgol_filter(j_LHCD_lower, window_length=11, polyorder=3, mode='interp')

    # ------------------------
    # 单位用于绘图
    # ------------------------
    p_plot_MWpm3 = p_LH_smooth / 1e6        # MW/m^3 for plotting
    j_plot_Apcm2 = j_LHCD_smooth * 1e-4    # A/cm^2 for plotting

    # ------------------------
    # Vrho 和 invR 插值到 rho_lower
    # ------------------------
    f_Vrho_to_lower = interp1d(rho, Vrho_on_rho, bounds_error=False, fill_value="extrapolate")
    Vrho_lower = f_Vrho_to_lower(rho_lower)

    f_invR_to_lower = interp1d(rho, invR_on_rho, bounds_error=False, fill_value="extrapolate")
    invR_lower = f_invR_to_lower(rho_lower)

    # ------------------------
    # 累积功率 & 累积电流（向量化）
    # P_enc(ρ) = ∫ p(ρ') * V_rho(ρ') dρ'   -> W  -> /1e6 -> MW
    # I_enc(ρ) = ∫ j(ρ') * V_rho(ρ') * invR(ρ')/(2π) dρ' -> A -> /1e3 -> kA
    # ------------------------
    cumulative_power_W = cumulative_trapezoid(p_LH_smooth * Vrho_lower, rho_lower, initial=0)
    cumulative_power_MW = cumulative_power_W / 1e6

    cumulative_current_A = cumulative_trapezoid(j_LHCD_smooth * Vrho_lower * invR_lower / (2.0 * np.pi),
                                                rho_lower, initial=0)
    cumulative_current_kA = cumulative_current_A / 1e3

    print(f"Total injected power (from integration) = {cumulative_power_MW[-1]:.6f} MW (input P_in = {params['P_in']/1e6:.6f} MW)")
    print(f"Total driven current (from integration) = {cumulative_current_kA[-1]:.3f} kA")

    # ------------------------
    # 绘图（四图）
    # ------------------------
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Shot:{shot}    Time:{time}s')

    # 电流密度 (A/cm^2)
    ax1 = axs[0,0]
    ax1.plot(rho_lower, j_plot_Apcm2, 'b-', lw=2)
    ax1.set_xlabel(r'$\rho$', fontsize=12)
    ax1.set_ylabel('Current Density (A/cm²)', fontsize=12)
    ax1.set_title('Lower Hybrid Current Drive', fontsize=14)
    ax1.grid(alpha=0.3)

    # 功率沉积 (MW/m^3)
    ax2 = axs[1,0]
    ax2.plot(rho_lower, p_plot_MWpm3, 'r-', lw=2)
    ax2.set_xlabel(r'$\rho$', fontsize=12)
    ax2.set_ylabel('Power Density (MW/m³)', fontsize=12)
    ax2.set_title('Lower Hybrid Power Deposition', fontsize=14)
    ax2.grid(alpha=0.3)

    # 累积总电流 (kA)
    ax3 = axs[0,1]
    ax3.plot(rho_lower, cumulative_current_kA, 'g-', lw=2)
    ax3.axhline(y=cumulative_current_kA[-1], color='r', linestyle='--',
                label=f'Total Current: {cumulative_current_kA[-1]:.2f} kA')
    ax3.set_xlabel(r'$\rho$', fontsize=12)
    ax3.set_ylabel('Cumulative Current (kA)', fontsize=12)
    ax3.set_title('Cumulative LHCD Current', fontsize=14)
    ax3.grid(alpha=0.3)
    ax3.legend()

    # 累积总功率 (MW)
    ax4 = axs[1,1]
    ax4.plot(rho_lower, cumulative_power_MW, 'm-', lw=2)
    ax4.axhline(y=cumulative_power_MW[-1], color='r', linestyle='--',
                label=f'Total Power: {cumulative_power_MW[-1]:.3f} MW')
    ax4.set_xlabel(r'$\rho$', fontsize=12)
    ax4.set_ylabel('Cumulative Power (MW)', fontsize=12)
    ax4.set_title('Cumulative LHCD Power', fontsize=14)
    ax4.grid(alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.show()
