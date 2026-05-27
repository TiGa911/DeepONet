# -*- coding: utf-8 -*-
"""
LHW_main.py

自动从 EAST 读数据，构造 METIS-LH 所需的 cons / profil / option，
调用 external_call_metis_lh_model 计算 LHCD 电流驱动和功率沉积，
并画出剖面（电流密度 / 功率密度 / 累积电流 / 累积功率）。

依赖：
- LHW_metis.py          提供 external_call_metis_lh_model
- readMDS.py            提供 readmds(shot, time)
- geqdsk.py             提供 geqdsk.load() 读取 gfile
- mesh2.py              提供 g_to_23()
- fitting_mtanh.py      提供 fitting()
- MDSplus               提供 Connection
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

from MDSplus.connection import Connection
import geqdsk
from mesh2 import g_to_23
from fitting_mtanh import fitting

from readMDS import readmds
from LHW_metis import external_call_metis_lh_model


# ============================================================
# 用 EAST 数据构造 METIS 的 cons / profil / option
# ============================================================

def build_east_metis_inputs(
    shot,
    time,
    npar0=2.1,
    wlh=0.35,
    freqlh_GHz=4.6,
    gaz=2,
    etalh=0.75,
    lhmode=0,
    upshiftmode="newmodel",
    fupshift=1.0,
    xlh0=0.2,
    dlh0=0.3,
):
    """
    使用你在 lower_view.py 里的处理方式，从 EAST 读取一个时间点数据，
    生成 METIS-LH 所需的 cons / profil / option。

    参数:
        shot       : EAST shot
        time       : 目标时间 (s)
        其它参数   : 天线 / 模型经验参数（npar0、wlh、xlh0、dlh0、upshift 等）

    返回:
        cons, profil, option, rho, real_time, PLH_W
    """
    # --------------------- 1. 读诊断数据 ---------------------
    data, status, real_time = readmds(shot, time)
    z_eff_value = data.get("zeff", 2.0)
    print(f"real_time from EFIT: {real_time}")
    print("Z_eff =", z_eff_value)

    # --- Te 剖面拟合（TS） ---
    x_Te, y_Te, _type_Te = (
        data["Te"]["TS"]["Rho"],
        data["Te"]["TS"]["data"],
        data["Te"]["TS"]["type"],
    )
    x_Te, y_Te, _, _, _ = fitting(x_Te, y_Te, "TS")  # 拟合平滑

    # --- ne 剖面拟合（Refl） ---
    x_ne, y_ne, _type_ne = (
        data["ne"]["Refl"]["Rho"],
        data["ne"]["Refl"]["data"],
        data["ne"]["Refl"]["type"],
    )
    x_ne, y_ne, _, _, _ = fitting(x_ne, y_ne, "Refl")

    # --------------------- 2. 读 gfile 平衡 ---------------------
    gdate = geqdsk.load(f"{shot}_{time}_gfile")

    # rho 网格使用 gfile 的 nw
    rho = np.linspace(0.0, 1.0, gdate["nw"])

    # 插值 ne / Te 到 rho 网格
    ne_profile = np.interp(rho, x_ne, y_ne)    # 这里 y_ne 单位和原脚本一致
    te_profile_keV = np.interp(rho, x_Te, y_Te)  # keV
    te_profile_eV = te_profile_keV * 1e3        # METIS 用 eV

    # --------------------- 3. Vrho（体积导数）计算 ---------------------
    # q(ψ) -> q(ρ) 中用到的 qpsi（后面单独再处理 q）
    qpsi = gdate.get("qpsi", np.ones_like(rho))

    psinorm = np.linspace(0, 1, len(rho))                 # 归一化 ψ 网格
    theta = np.linspace(0, 2 * np.pi, len(rho))           # θ 网格（这里只用到 psi 方向上的 dvdpsi 和 phi）
    grid = g_to_23(f"{shot}_{time}_gfile", psinorm, theta)

    # Vrho 计算（在 psi 网格上）
    dvdpsi = np.asarray(grid["dvdpsi"], dtype=float)      # dV/dψ
    phi = np.asarray(grid["phi"], dtype=float)            # 对应的 φ(ψ)
    npsi = len(dvdpsi)

    # 构造 rho_psi (确保和 dvdpsi 对应)
    phi0 = phi[0]
    phi_sep = phi[-1]
    rho_psi = np.sqrt((phi - phi0) / (phi_sep - phi0 + 1e-30))

    # psi_array 对应 dvdpsi
    psia = gdate["simag"]
    psib = gdate["sibry"]
    psi_array = psia + np.linspace(0, 1, npsi) * (psib - psia)

    # dψ/dρ
    dpsi_drho = np.gradient(psi_array, rho_psi)

    # Vrho 在 psi 网格：dV/dρ = dV/dψ * dψ/dρ
    Vrho_on_psi = dvdpsi * dpsi_drho

    # 将 Vrho 插值到 rho 网格
    f_Vrho = interp1d(
        rho_psi,
        Vrho_on_psi,
        bounds_error=False,
        fill_value="extrapolate",
    )
    Vrho_on_rho = f_Vrho(rho)  # 与 'rho' 对齐 - 这就是 METIS 的 vpr

    # --------------------- 4. 低杂净功率 PLH2 读取 ---------------------
    conn = Connection("202.127.204.12")
    try:
        conn.openTree("EAST", shot)
        PLHI2 = conn.get(r"data(\PLHI2)").data()
        PLHI2_times = conn.get(r"dim_of(\PLHI2)").data()

        window_start = time - 0.1
        window_end = time
        mask = (PLHI2_times >= window_start) & (PLHI2_times <= window_end)
        PLHI2_in = np.mean(PLHI2[mask]) if mask.any() else 0.0

        PLHR2 = conn.get(r"data(\PLHR2)").data()
        PLHR2_times = conn.get(r"dim_of(\PLHR2)").data()
        mask2 = (PLHR2_times >= window_start) & (PLHR2_times <= window_end)
        PLHR2_in = np.mean(PLHR2[mask2]) if mask2.any() else 0.0

        conn.closeTree("EAST", shot)
        PLH2_kW = PLHI2_in - PLHR2_in  # kW
        print(f"Net Power at {time}s: {PLH2_kW:.2f} kW")
    except Exception as e:
        print(f"Error reading power from MDSplus: {e}")
        PLH2_kW = 0.0

    PLH_W = PLH2_kW * 1e3

    # --------------------- 5. 几何参数（R_axis, a, 1/R) ---------------------
    filtered_data = gdate["bbbsrz"][gdate["bbbsrz"][:, 0] != 0]
    bbbsrz_R = filtered_data[:, 0]

    a_R = (np.max(bbbsrz_R) + np.min(bbbsrz_R)) / 2.0   # 近似磁轴大半径 R_axis
    a_r = (np.max(bbbsrz_R) - np.min(bbbsrz_R)) / 2.0   # 近似小半径 a

    # invR_on_psi：这里先用 1/R_axis 近似（如果需要可以换成更精确的 1/R(ψ)）
    invR_on_psi = np.full_like(rho_psi, 1.0 / a_R)

    # 插值到 rho 网格
    f_invR = interp1d(
        rho_psi,
        invR_on_psi,
        bounds_error=False,
        fill_value="extrapolate",
    )
    invR_on_rho = f_invR(rho)  # 目前暂时没直接喂给 METIS，用于你后续扩展也行

    # --------------------- 6. 构建 METIS 的 cons / profil / option ---------

    # 总电流 / B0 / Raxis 等从 gfile 里拿
    Ip = float(gdate.get("cpasma", 0.0))     # A
    B0 = float(gdate.get("bcentr", 2.0))     # T
    R_axis_from_g = float(gdate.get("rmaxis", a_R))

    # --- cons ---
    cons = dict(
        temps=np.array([real_time]),  # 时间 [s]
        ip=np.array([Ip]),            # 等离子体电流 [A]
        plh=np.array([PLH_W]),        # LH 净功率 [W]
    )

    nt = 1
    nx = rho.size
    ve = np.ones(nx)

    # --- profil ---
    # Raxe 用 gfile 的 rmaxis（也可以用 a_R，差别不大）
    Raxe_2d = np.full((nt, nx), R_axis_from_g)
    # epsi = a/R
    epsi_2d = np.full((nt, nx), a_r / R_axis_from_g)
    # fdia = R * B_T
    fdia_2d = Raxe_2d * B0

    # q(rho)：用 qpsi(ψ) & ψ->rho 映射 sqrt(psin)
    psin_q = np.linspace(0.0, 1.0, len(qpsi))
    rho_q = np.sqrt(psin_q)
    q_profile = np.interp(rho, rho_q, qpsi)
    q_profile = np.nan_to_num(q_profile, nan=1.0, posinf=10.0, neginf=1e-3)

    qjli_2d = np.tile(q_profile.reshape(1, -1), (nt, 1))

    # 电子密度 [m^-3]：ne_profile * 1e19（沿用你原脚本习惯）
    nep_2d = np.tile((ne_profile * 1e19).reshape(1, -1), (nt, 1))  # [m^-3]
    # 电子温度 [eV]
    tep_2d = np.tile(te_profile_eV.reshape(1, -1), (nt, 1))        # [eV]

    # 小半径 r(ρ) = ρ * a_r
    r = rho * a_r
    rmx_2d = np.tile(r.reshape(1, -1), (nt, 1))

    # vpr = dV/dρ，从 gfile + g_to_23 计算而来
    vpr_1d = Vrho_on_rho              # [m^3]
    vpr_2d = np.tile(vpr_1d.reshape(1, -1), (nt, 1))

    # R(ρ) ≈ R_axis_from_g + r(ρ)
    R_profile_1d = R_axis_from_g + r  # [m]

    # spr：按照 ONETWO 逻辑，用 gfile 几何重新定义
    # spr(ρ) = [1 / (2π R(ρ))] * dV/dρ
    # 这样 METIS 里的 ∫ spr * j dρ = ∫ j/(2πR) * dV/dρ dρ
    spr_1d = vpr_1d / (2.0 * np.pi * np.maximum(R_profile_1d, 1e-6))
    spr_2d = np.tile(spr_1d.reshape(1, -1), (nt, 1))


    # Zeff 剖面：先用常数值
    zeff_2d = np.full((nt, nx), z_eff_value)

    profil = dict(
        xli=rho,
        Raxe=Raxe_2d,
        epsi=epsi_2d,
        fdia=fdia_2d,
        qjli=qjli_2d,
        nep=nep_2d,
        tep=tep_2d,
        rmx=rmx_2d,
        spr=spr_2d,
        vpr=vpr_2d,
        zeff=zeff_2d,
        # epar / plh 剖面可选（算 vloop_lh 时再加）
    )

    # --- option ---
    option = dict(
        gaz=gaz,
        freqlh=freqlh_GHz,
        etalh=etalh,
        npar0=npar0,
        wlh=wlh,
        npar_neg=None,       # None => 在 LHW_metis 里按 upshiftmode 自动处理
        fupshift=fupshift,
        xlh=xlh0,
        dlh=dlh0,
        lhmode=lhmode,
        upshiftmode=upshiftmode,
    )

    return cons, profil, option, rho, real_time, PLH_W

def build_east_metis_inputs_from_txt(
    shot,
    time,
    txt_filename,
    npar0=2.0,
    wlh=0.4,
    freqlh_GHz=4.6,
    gaz=2,
    etalh=0.75,
    eta_scale=1.0,          # ★★★ 新增：电流整体放大因子
    lhmode=0,
    upshiftmode="newmodel",
    fupshift=1.0,
    xlh0=0.2,
    dlh0=0.3,
    z_eff_value=2.0,
):
    """
    从 txt 读取 Te, ne, rho 构造 METIS 输入
    """

    import numpy as np

    data = np.loadtxt(txt_filename)
    rho = data[:, 0]
    Te_eV = data[:, 1]
    ne_19 = data[:, 2]

    Te_keV = Te_eV / 1e3
    ne_m3 = ne_19 * 1e19

    profil = {
        "rho": rho,
        "ne": ne_m3,
        "Te": Te_keV,
        "zeff": np.full_like(rho, z_eff_value),
    }

    cons = {
        "rlim": 2.0,
        "alim": 0.45,
        "freq": freqlh_GHz,
        "gaz": gaz,
    }

    # ===== 关键：把 eta_scale 写进 option =====
    option = {
        "npar0": npar0,
        "wlh": wlh,
        "etalh": etalh,
        "eta_scale": eta_scale,   # ★★★ 新增：整体电流缩放因子
        "lhmode": lhmode,
        "upshiftmode": upshiftmode,
        "fupshift": fupshift,
        "xlh": xlh0,
        "dlh": dlh0,
    }

    return cons, profil, option, rho, time, None

# ============================================================
# 主程序：调用 METIS-LH 并画剖面
# ============================================================

if __name__ == "__main__":
    shot = int(input("Input EAST shot: "))
    time = float(input("Input time (s): "))

    # 构造 METIS 输入
    cons, profil, option, rho, real_time, PLH_W = build_east_metis_inputs(
        shot=shot,
        time=time,
        # 以下是可调 LH 经验参数，可以根据 EAST 调整
        npar0=2.0,
        wlh=0.4,
        freqlh_GHz=4.6,
        gaz=2,
        etalh=0.75,
        lhmode=0,
        upshiftmode="newmodel",
        fupshift=1.0,
        xlh0=0.2,
        dlh0=0.3,
    )

    # 调用 METIS 低杂波模块
    time_arr, plh_tot, ilh, x, plh_prof, jlh_prof, eff = external_call_metis_lh_model(
        cons, profil, option
    )

    this_time = float(time_arr[0])
    print("\n===== METIS-LH 结果 =====")
    print(f"time = {this_time:.3f} s")
    print(f"Input LH power (net): {PLH_W/1e6:.3f} MW")
    print(f"Absorbed LH power:    {plh_tot[0]/1e6:.3f} MW")
    print(f"Driven LH current:    {ilh[0]/1e3:.2f} kA")
    print(f"Global efficiency:    {eff[0]:.3e} A/W/m^2")
    print("=========================\n")

    # 单一时间剖面
    j_A_m2 = jlh_prof[0, :]      # A/m^2
    p_W_m3 = plh_prof[0, :]      # W/m^3

    # 单位转换（仅用于画图/输出）
    j_Apcm2 = j_A_m2 / 1e4       # A/m^2 -> A/cm^2
    p_MWpm3 = p_W_m3 / 1e6       # W/m^3 -> MW/m^3

    # 几何量
    vpr = np.asarray(profil["vpr"])[0, :]     # dV/dρ [m^3]
    Raxe = np.asarray(profil["Raxe"])[0, :]   # R_axis(ρ) [m]
    R0 = float(Raxe[-1])

    print(f"R0 used in Ip integration = {R0:.3f} m")

    # 累积总电流：I(ρ) = ∫ [ J(ρ')/(2π R0) ] * dV/dρ'(ρ') dρ'
    J_over_2piR = j_A_m2 / (2.0 * np.pi * R0)   # A/m^3
    integrand_I = J_over_2piR * vpr             # A
    cumulative_current_A = cumulative_trapezoid(integrand_I, x, initial=0.0)
    cumulative_current_kA = cumulative_current_A / 1e3

    # 累积总功率：∫ p(ρ) * dV/dρ dρ
    cumulative_power_W = cumulative_trapezoid(p_W_m3 * vpr, x, initial=0.0)
    cumulative_power_MW = cumulative_power_W / 1e6


    # 画图
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Shot: {shot}    Time: {this_time:.3f} s", fontsize=16)

    # (1) 电流密度 (A/cm^2)
    ax1 = axs[0, 0]
    ax1.plot(x, j_Apcm2, lw=2)
    ax1.set_xlabel(r"$\rho$", fontsize=12)
    ax1.set_ylabel("Current Density (A/cm²)", fontsize=12)
    ax1.set_title("Lower Hybrid Current Drive", fontsize=14)
    ax1.grid(alpha=0.3)

    # (2) 功率沉积 (MW/m^3)
    ax2 = axs[1, 0]
    ax2.plot(x, p_MWpm3, lw=2)
    ax2.set_xlabel(r"$\rho$", fontsize=12)
    ax2.set_ylabel("Power Density (MW/m³)", fontsize=12)
    ax2.set_title("Lower Hybrid Power Deposition", fontsize=14)
    ax2.grid(alpha=0.3)

    # (3) 累积总电流 (kA)
    ax3 = axs[0, 1]
    ax3.plot(x, cumulative_current_kA, "g-", lw=2)
    ax3.axhline(
        y=cumulative_current_kA[-1],
        color="r",
        linestyle="--",
        label=f"Total Current: {cumulative_current_kA[-1]:.2f} kA",
    )
    ax3.set_xlabel(r"$\rho$", fontsize=12)
    ax3.set_ylabel("Cumulative Current (kA)", fontsize=12)
    ax3.set_title("Cumulative LHCD Current", fontsize=14)
    ax3.grid(alpha=0.3)
    ax3.legend()

    # (4) 累积总功率 (MW)
    ax4 = axs[1, 1]
    ax4.plot(x, cumulative_power_MW, "m-", lw=2)
    ax4.axhline(
        y=cumulative_power_MW[-1],
        color="r",
        linestyle="--",
        label=f"Total Power: {cumulative_power_MW[-1]:.3f} MW",
    )
    ax4.set_xlabel(r"$\rho$", fontsize=12)
    ax4.set_ylabel("Cumulative Power (MW)", fontsize=12)
    ax4.set_title("Cumulative LHCD Power", fontsize=14)
    ax4.grid(alpha=0.3)
    ax4.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
