# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

from MDSplus.connection import Connection
import geqdsk
from mesh2 import g_to_23

from LHW_main import build_east_metis_inputs
from LHW_metis import external_call_metis_lh_model

import tkinter as tk
from tkinter import ttk, messagebox, filedialog


# ===========================
# 从 TXT 读 rho, ne, Te
# ===========================

def read_txt_profiles(filename):
    """
    从文本文件读取 rho, ne, Te 剖面.

    按 数据wu1.txt 结构：
        col1: rho
        col2: ne (1e19 m^-3)
        col3: Te(exp) (keV)
        col4: Ti(exp)
        col5: Te(simu)
        col6: Ti(simu)

    默认用 Te(exp)，如果想用 Te(simu)，把下面改成 data[:,4] 即可。
    """
    data = np.loadtxt(filename, skiprows=1)
    rho_txt = data[:, 0]
    ne_txt_1e19 = data[:, 1]
    te_txt_keV = data[:, 2]  # 用 Te(exp)
    return rho_txt, ne_txt_1e19, te_txt_keV


# ===========================
# 纯 txt 模式：不调用 readMDS，只用 gfile + PLH + txt
# ===========================

def build_east_metis_inputs_from_txt(
    shot,
    time,
    txt_filename,
    npar0=2.0,
    wlh=0.4,
    freqlh_GHz=4.6,
    gaz=2,
    etalh=0.75,
    eta_scale=1.0,
    lhmode=0,
    upshiftmode="newmodel",
    fupshift=1.0,
    xlh0=0.2,
    dlh0=0.3,
    z_eff_value=2.0,
):
    """
    txt 模式专用：
        - 不调用 readMDS / TS / Refl
        - Te / ne 完全来自 txt 文件
        - 几何和功率来自 gfile + PLHI2/PLHR2

    返回:
        cons, profil, option, rho, time(原样返回), PLH_W
    """

    # # ---------- 1. 读 gfile ----------
    gfile_name = f"{shot}_{time}_gfile"
    if not os.path.exists(gfile_name):
        raise FileNotFoundError(
            f"未找到 gfile: {gfile_name}\n"
            f"请先确保当前目录有该 gfile 文件。"
        )

    gdate = geqdsk.load(gfile_name)

    # rho 网格
    rho = np.linspace(0.0, 1.0, gdate["nw"])

    # ---------- 2. Vrho 计算 ----------
    qpsi = gdate.get("qpsi", np.ones_like(rho))

    psinorm = np.linspace(0, 1, len(rho))
    theta = np.linspace(0, 2 * np.pi, len(rho))
    grid = g_to_23(gfile_name, psinorm, theta)

    dvdpsi = np.asarray(grid["dvdpsi"], dtype=float)
    phi = np.asarray(grid["phi"], dtype=float)
    npsi = len(dvdpsi)

    phi0 = phi[0]
    phi_sep = phi[-1]
    rho_psi = np.sqrt((phi - phi0) / (phi_sep - phi0 + 1e-30))

    psia = gdate["simag"]
    psib = gdate["sibry"]
    psi_array = psia + np.linspace(0, 1, npsi) * (psib - psia)

    dpsi_drho = np.gradient(psi_array, rho_psi)
    Vrho_on_psi = dvdpsi * dpsi_drho

    f_Vrho = interp1d(
        rho_psi,
        Vrho_on_psi,
        bounds_error=False,
        fill_value="extrapolate",
    )
    Vrho_on_rho = f_Vrho(rho)  # METIS 用的 vpr = dV/dρ

    # ---------- 3. 几何参数 ----------
    filtered_data = gdate["bbbsrz"][gdate["bbbsrz"][:, 0] != 0]
    bbbsrz_R = filtered_data[:, 0]

    a_R = (np.max(bbbsrz_R) + np.min(bbbsrz_R)) / 2.0   # 近似磁轴大半径 R_axis
    a_r = (np.max(bbbsrz_R) - np.min(bbbsrz_R)) / 2.0   # 近似小半径 a

    invR_on_psi = np.full_like(rho_psi, 1.0 / a_R)
    f_invR = interp1d(
        rho_psi,
        invR_on_psi,
        bounds_error=False,
        fill_value="extrapolate",
    )
    invR_on_rho = f_invR(rho)  # 目前没直接用到

    # ---------- 4. 读取并计算低杂净功率 ----------
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
        PLH2 = PLHI2_in - PLHR2_in  # kW
        print(f"Net Power at {time}s: {PLH2:.2f} kW")
    except Exception as e:
        print(f"Error reading power from MDSplus (PLH2): {e}")
        PLH2 = 0.0

    PLH_W = PLH2 * 1e3

    # ---------- 5. Te / ne 从 TXT 读取 ----------
    if not os.path.exists(txt_filename):
        raise FileNotFoundError(f"找不到 TXT 文件：{txt_filename}")

    rho_txt, ne_txt_1e19, te_txt_keV = read_txt_profiles(txt_filename)

    ne_interp_1e19 = np.interp(rho, rho_txt, ne_txt_1e19)
    te_interp_keV = np.interp(rho, rho_txt, te_txt_keV)

    ne_interp_m3 = ne_interp_1e19 * 1e19
    te_interp_eV = te_interp_keV * 1e3

    # ---------- 6. 构建 METIS 输入 ----------
    Ip = float(gdate.get("cpasma", 0.0))
    B0 = float(gdate.get("bcentr", 2.0))
    R_axis_from_g = float(gdate.get("rmaxis", a_R))

    cons = dict(
        temps=np.array([time]),
        ip=np.array([Ip]),
        plh=np.array([PLH_W]),
    )

    nt = 1
    nx = rho.size

    Raxe_2d = np.full((nt, nx), R_axis_from_g)
    epsi_2d = np.full((nt, nx), a_r / R_axis_from_g)
    fdia_2d = Raxe_2d * B0

    psin_q = np.linspace(0.0, 1.0, len(qpsi))
    rho_q = np.sqrt(psin_q)
    q_profile = np.interp(rho, rho_q, qpsi)
    q_profile = np.nan_to_num(q_profile, nan=1.0, posinf=10.0, neginf=1e-3)
    qjli_2d = np.tile(q_profile.reshape(1, -1), (nt, 1))

    nep_2d = np.tile(ne_interp_m3.reshape(1, -1), (nt, 1))
    tep_2d = np.tile(te_interp_eV.reshape(1, -1), (nt, 1))

    r = rho * a_r
    rmx_2d = np.tile(r.reshape(1, -1), (nt, 1))
    vpr_2d = np.tile(Vrho_on_rho.reshape(1, -1), (nt, 1))

    spr_1d = 4.0 * np.pi**2 * a_R * np.maximum(r, 1e-4)
    spr_2d = np.tile(spr_1d.reshape(1, -1), (nt, 1))

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
    )

    # 在 option 里加上 eta_scale（机器标定因子）
    option = dict(
        gaz=gaz,
        freqlh=freqlh_GHz,
        etalh=etalh,
        npar0=npar0,
        wlh=wlh,
        npar_neg=None,
        fupshift=fupshift,
        xlh=xlh0,
        dlh=dlh0,
        lhmode=lhmode,
        upshiftmode=upshiftmode,
        eta_scale=eta_scale,
    )

    return cons, profil, option, rho, time, PLH_W


# ===========================
# 统一计算入口（GUI 调这个）
# ===========================

def run_metis_lh_calculation(
    shot,
    time,
    data_source,
    txt_filename,
    gaz,
    freqlh_GHz,
    etalh,
    eta_scale,
    npar0,
    wlh,
    lhmode,
    upshiftmode,
    fupshift,
    xlh0,
    dlh0,
):
    if data_source == "mds":
        print("[MDS] 使用 TS + Refl + gfile + PLHI2/PLHR2")
        cons, profil, option, rho, real_time, PLH_W = build_east_metis_inputs(
            shot=shot,
            time=time,
            npar0=npar0,
            wlh=wlh,
            freqlh_GHz=freqlh_GHz,
            gaz=gaz,
            etalh=etalh,
            lhmode=lhmode,
            upshiftmode=upshiftmode,
            fupshift=fupshift,
            xlh0=xlh0,
            dlh0=dlh0,
        )
        # 在 option 里挂上 eta_scale
        option["eta_scale"] = eta_scale

    else:
        print("[TXT] 使用 gfile + PLHI2/PLHR2 + txt 中的 Te/ne")
        cons, profil, option, rho, real_time, PLH_W = build_east_metis_inputs_from_txt(
            shot=shot,
            time=time,
            txt_filename=txt_filename,
            npar0=npar0,
            wlh=wlh,
            freqlh_GHz=freqlh_GHz,
            gaz=gaz,
            etalh=etalh,
            eta_scale=eta_scale,  # 传进去构造 option
            lhmode=lhmode,
            upshiftmode=upshiftmode,
            fupshift=fupshift,
            xlh0=xlh0,
            dlh0=dlh0,
            z_eff_value=3.0,
        )
        # 再保险一次（即使上面已经写进 option）
        option["eta_scale"] = eta_scale

    time_arr, plh_tot, ilh, x, plh_prof, jlh_prof, eff = external_call_metis_lh_model(
        cons, profil, option
    )

    this_time = float(time_arr[0])
    print("\n===== METIS-LH 结果 =====")
    print(f"time = {this_time:.3f} s")
    print(f"Input LH net power: {PLH_W / 1e6:.3f} MW")
    print(f"Absorbed LH power:  {plh_tot[0] / 1e6:.3f} MW")
    print(f"Driven LH current:  {ilh[0] / 1e3:.2f} kA")
    print(f"Global efficiency:  {eff[0]:.3e} A/W/m^2")
    print("=========================\n")

    # jlh_prof: A/m^2（METIS 通量面平均平行电流密度）
    j_A_m2 = jlh_prof[0, :]
    p_W_m3 = plh_prof[0, :]

    # 只做单位转换：A/m^2 → A/cm^2（与论文 / 数据 current.txt 一致）
    j_Apcm2 = j_A_m2 / 1e4
    p_MWpm3 = p_W_m3 / 1e6

    # ========== 从 profil 中取几何量 ==========
    vpr = np.asarray(profil["vpr"])[0, :]        # dV/dρ [m^3]
    Raxe = np.asarray(profil["Raxe"])[0, :]      # R_axis(ρ) [m]
    print('Raxe:',Raxe)
    rmx  = np.asarray(profil["rmx"])[0, :]       # 小半径 r(ρ) [m]
    print('rmx',rmx)

    # ---------- 构造 R(ρ) ----------
    # 真实 tokamak 的 R(ρ) ~ R_axis + r(ρ)
    # 与 g_to_23 平均 <R(ψ)> 十分一致
    R_profile = Raxe + rmx     # [m] 每个 ρ 的大半径

    # =====================================================
    # 用你的目标公式：
    #      I(ρ) = ∫ [ J(ρ') / (2π R(ρ')) ] * dV/dρ'(ρ') dρ'
    # =====================================================
    J_over_2piR = j_A_m2 / (2.0 * np.pi * R_profile)   # [A/m^3]
    integrand_I = J_over_2piR * vpr                    # [A]

    cumulative_current_A = cumulative_trapezoid(
        integrand_I, x, initial=0.0
    )
    cumulative_current_kA = cumulative_current_A / 1e3

    # 功率：仍然是 ∫ p(ρ) dV/dρ dρ
    cumulative_power_W = cumulative_trapezoid(p_W_m3 * vpr, x, initial=0.0)
    cumulative_power_MW = cumulative_power_W / 1e6



    # ---------- 把“4 张图”的数据保存成 txt ----------
    out_dir = "LHW_results"
    os.makedirs(out_dir, exist_ok=True)
    file_name = f"LHW_{data_source}_shot{shot}_t{this_time:.3f}.txt"
    file_path = os.path.join(out_dir, file_name)

    header = (
        "# LHW METIS profile output (for 4-figure plotting)\n"
        f"# shot = {shot}\n"
        f"# time = {this_time:.6f} s\n"
        f"# data_source = {data_source}\n"
        f"# eta_scale = {eta_scale:.6e}\n"
        f"# Input LH net power (W)  = {PLH_W:.6e}\n"
        f"# Absorbed LH power (W)   = {plh_tot[0]:.6e}\n"
        f"# Driven LH current (A)   = {ilh[0]:.6e}\n"
        "# Columns:\n"
        "# rho   j_surf(A/cm^2)   p(MW/m^3)   Icum(kA)   Pcum(MW)\n"
    )

    data_to_save = np.column_stack(
        [
            x,
            j_Apcm2,
            p_MWpm3,
            cumulative_current_kA,
            cumulative_power_MW,
        ]
    )

    np.savetxt(file_path, data_to_save, header=header)
    print(f"剖面已保存到: {file_path}")

    # ---------- 画图 ----------
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Shot: {shot}    Time: {this_time:.3f} s", fontsize=16)

    # 图1：j_surf(A/cm^2)
    ax1 = axs[0, 0]
    ax1.plot(x, j_Apcm2, lw=2)
    ax1.set_xlabel(r"$\rho$", fontsize=12)
    ax1.set_ylabel("Current Density (A/cm²)", fontsize=12)
    ax1.set_title("Lower Hybrid Current Drive", fontsize=14)
    ax1.grid(alpha=0.3)

    # 图2：p(MW/m^3)
    ax2 = axs[1, 0]
    ax2.plot(x, p_MWpm3, lw=2)
    ax2.set_xlabel(r"$\rho$", fontsize=12)
    ax2.set_ylabel("Power Density (MW/m³)", fontsize=12)
    ax2.set_title("Lower Hybrid Power Deposition", fontsize=14)
    ax2.grid(alpha=0.3)

    # 图3：累计总电流 (kA)
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

    # 图4：累计功率 (MW)
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


# ===========================
# Tkinter GUI
# ===========================

def launch_gui():
    root = tk.Tk()
    root.title("EAST METIS 低杂波 LHCD 模型（txt / mds）")

    frm_top = ttk.LabelFrame(root, text="基本设置")
    frm_top.pack(fill="x", padx=10, pady=5)

    tk.Label(frm_top, text="Shot:").grid(row=0, column=0, sticky="e", padx=5, pady=3)
    tk.Label(frm_top, text="Time (s):").grid(row=0, column=2, sticky="e", padx=5, pady=3)

    shot_var = tk.StringVar(value="81481")
    time_var = tk.StringVar(value="5.30")
    tk.Entry(frm_top, textvariable=shot_var, width=8).grid(row=0, column=1, padx=5, pady=3)
    tk.Entry(frm_top, textvariable=time_var, width=8).grid(row=0, column=3, padx=5, pady=3)

    tk.Label(frm_top, text="数据源:").grid(row=1, column=0, sticky="e", padx=5, pady=3)
    source_var = tk.StringVar(value="txt")   # 默认直接用 txt
    cmb_source = ttk.Combobox(
        frm_top,
        textvariable=source_var,
        values=["mds", "txt"],
        width=8,
        state="readonly",
    )
    cmb_source.grid(row=1, column=1, padx=5, pady=3)

    tk.Label(frm_top, text="TXT 文件:").grid(row=1, column=2, sticky="e", padx=5, pady=3)
    txt_file_var = tk.StringVar(value="数据wu1.txt")
    txt_entry = tk.Entry(frm_top, textvariable=txt_file_var, width=22)
    txt_entry.grid(row=1, column=3, padx=5, pady=3)

    def browse_txt():
        fname = filedialog.askopenfilename(
            title="选择 Te/ne TXT 文件",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if fname:
            txt_file_var.set(fname)

    ttk.Button(frm_top, text="浏览...", command=browse_txt).grid(
        row=1, column=4, padx=5, pady=3
    )

    def on_source_change(*args):
        if source_var.get() == "txt":
            txt_entry.config(state="normal")
        else:
            txt_entry.config(state="disabled")

    source_var.trace_add("write", on_source_change)
    on_source_change()

    # --- 参数区域 ---
    frm_param = ttk.LabelFrame(root, text="METIS-LH 参数")
    frm_param.pack(fill="x", padx=10, pady=5)

    tk.Label(frm_param, text="主离子 gaz:").grid(row=0, column=0, sticky="e", padx=5, pady=3)
    gaz_var = tk.StringVar(value="2")
    cmb_gaz = ttk.Combobox(
        frm_param,
        textvariable=gaz_var,
        values=["1", "2", "3", "4"],
        width=6,
        state="readonly",
    )
    cmb_gaz.grid(row=0, column=1, padx=5, pady=3)

    tk.Label(frm_param, text="freqlh (GHz):").grid(row=0, column=2, sticky="e", padx=5, pady=3)
    freqlh_var = tk.StringVar(value="4.6")
    tk.Entry(frm_param, textvariable=freqlh_var, width=8).grid(
        row=0, column=3, padx=5, pady=3
    )

    tk.Label(frm_param, text="npar0:").grid(row=1, column=0, sticky="e", padx=5, pady=3)
    npar0_var = tk.StringVar(value="2.0")
    tk.Entry(frm_param, textvariable=npar0_var, width=8).grid(
        row=1, column=1, padx=5, pady=3
    )

    tk.Label(frm_param, text="wlh (m):").grid(row=1, column=2, sticky="e", padx=5, pady=3)
    wlh_var = tk.StringVar(value="0.4")
    tk.Entry(frm_param, textvariable=wlh_var, width=8).grid(
        row=1, column=3, padx=5, pady=3
    )

    tk.Label(frm_param, text="etalh:").grid(row=2, column=0, sticky="e", padx=5, pady=3)
    etalh_var = tk.StringVar(value="0.75")
    tk.Entry(frm_param, textvariable=etalh_var, width=8).grid(
        row=2, column=1, padx=5, pady=3
    )

    # 新增：eta_scale 输入
    tk.Label(frm_param, text="eta_scale:").grid(row=2, column=2, sticky="e", padx=5, pady=3)
    eta_scale_var = tk.StringVar(value="1.0")
    tk.Entry(frm_param, textvariable=eta_scale_var, width=8).grid(
        row=2, column=3, padx=5, pady=3
    )

    tk.Label(frm_param, text="lhmode:").grid(row=3, column=0, sticky="e", padx=5, pady=3)
    lhmode_var = tk.StringVar(value="0")
    cmb_lhmode = ttk.Combobox(
        frm_param,
        textvariable=lhmode_var,
        values=["0", "1", "2", "3", "4"],
        width=6,
        state="readonly",
    )
    cmb_lhmode.grid(row=3, column=1, padx=5, pady=3)

    tk.Label(frm_param, text="upshiftmode:").grid(row=3, column=2, sticky="e", padx=5, pady=3)
    upshiftmode_var = tk.StringVar(value="newmodel")
    cmb_up = ttk.Combobox(
        frm_param,
        textvariable=upshiftmode_var,
        values=[
            "newmodel",
            "newmodel + tail",
            "1/q",
            "Bpol",
            "x^2",
            "sqrt(x)",
            "null",
            "step@edge",
        ],
        width=14,
        state="readonly",
    )
    cmb_up.grid(row=3, column=3, padx=5, pady=3)

    tk.Label(frm_param, text="fupshift:").grid(row=4, column=0, sticky="e", padx=5, pady=3)
    fupshift_var = tk.StringVar(value="1.0")
    tk.Entry(frm_param, textvariable=fupshift_var, width=8).grid(
        row=4, column=1, padx=5, pady=3
    )

    tk.Label(frm_param, text="xlh0 (r/a):").grid(row=4, column=2, sticky="e", padx=5, pady=3)
    xlh0_var = tk.StringVar(value="0.2")
    tk.Entry(frm_param, textvariable=xlh0_var, width=8).grid(
        row=4, column=3, padx=5, pady=3
    )

    tk.Label(frm_param, text="dlh0 (r/a):").grid(row=5, column=0, sticky="e", padx=5, pady=3)
    dlh0_var = tk.StringVar(value="0.3")
    tk.Entry(frm_param, textvariable=dlh0_var, width=8).grid(
        row=5, column=1, padx=5, pady=3
    )

    frm_btn = ttk.Frame(root)
    frm_btn.pack(fill="x", padx=10, pady=5)

    def run_button_clicked():
        try:
            shot = int(shot_var.get())
            time_val = float(time_var.get())
            source = source_var.get()
            txt_file = txt_file_var.get()

            gaz_val = int(gaz_var.get())
            freqlh_val = float(freqlh_var.get())
            npar0_val = float(npar0_var.get())
            wlh_val = float(wlh_var.get())
            etalh_val = float(etalh_var.get())
            eta_scale_val = float(eta_scale_var.get())
            lhmode_val = int(lhmode_var.get())
            upshiftmode_val = upshiftmode_var.get()
            fupshift_val = float(fupshift_var.get())
            xlh0_val = float(xlh0_var.get())
            dlh0_val = float(dlh0_var.get())

            run_metis_lh_calculation(
                shot=shot,
                time=time_val,
                data_source=source,
                txt_filename=txt_file,
                gaz=gaz_val,
                freqlh_GHz=freqlh_val,
                etalh=etalh_val,
                eta_scale=eta_scale_val,
                npar0=npar0_val,
                wlh=wlh_val,
                lhmode=lhmode_val,
                upshiftmode=upshiftmode_val,
                fupshift=fupshift_val,
                xlh0=xlh0_val,
                dlh0=dlh0_val,
            )

        except Exception as e:
            messagebox.showerror("错误", f"计算过程中出错:\n{e}")

    ttk.Button(frm_btn, text="运行 METIS-LH", command=run_button_clicked).pack(
        side="left", padx=10, pady=5
    )
    ttk.Button(frm_btn, text="退出", command=root.destroy).pack(
        side="right", padx=5, pady=5
    )

    root.mainloop()


if __name__ == "__main__":
    launch_gui()
