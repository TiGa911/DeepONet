# -*- coding: utf-8 -*-
"""
Python translation of METIS LH module:

- external_call_metis_lh_model.m
- z0lhacc2lobes.m
- z0lhacc.m

使用 scipy.constants 的物理常数，并保留与原 MATLAB 版本同样的输入/输出接口。

外部主接口：
    external_call_metis_lh_model(cons, profil, option)

输入：
    cons:
        temps : (nt,)  时间 [s]
        ip    : (nt,)  等离子体电流 [A]
        plh   : (nt,)  LH 输入功率 [W]

    profil:
        xli   : (nx,)      Lao 坐标 r/a
        Raxe  : (nt,nx)    每个磁面上的 R_axis [m]
        epsi  : (nt,nx)    反扁率 a(x)/Raxe(x)
        fdia  : (nt,nx)    R * B_T [T·m]
        qjli  : (nt,nx)    q(r)
        nep   : (nt,nx)    ne [m^-3]
        tep   : (nt,nx)    Te [eV]
        rmx   : (nt,nx)    toroidal flux 坐标 [m]
        spr   : (nt,nx)    dS/dxli [m^2] #
        vpr   : (nt,nx)    dV/dxli [m^3] 
        zeff  : (nt,nx)    Zeff(r) （可选）
        epar  : (nt,nx)    平行电场 [V/m]（可选）
        plh   : (nt,nx)    （仅用于 vloop_lh，和 fplh 同名不会冲突）

    option:
        gaz         : 1->H, 2->D, 3->DT, 4->He
        freqlh      : LH 频率 [GHz]
        etalh       : directivity（正波瓣功率占比）
        npar0       : 正波瓣发射 n∥
        wlh         : 天线有效宽度 [m]
        npar_neg    : 负波瓣 n∥（=0 或未给时用 -npar0 或 -π npar0）
        fupshift    : Landau 共振位置因子
        xlh         : 经验初始沉积 r/a
        dlh         : 经验初始宽度 r/a
        lhmode      : 影响 directivity 的模式（和 METIS 一样）
        upshiftmode : 'newmodel', 'newmodel + tail', '1/q', 'Bpol', 'x^2',
                      'sqrt(x)', 'null', 'step@edge', 或其他（走 old model 分支）
"""

import numpy as np
from scipy.constants import (
    c as C_LIGHT,
    h as H_PLANCK,
    e as E_CHARGE,
    mu_0 as MU0,
    epsilon_0 as EPS0,
    G as G_GRAV,
    k as K_BOLTZ,
    alpha as ALPHA,
    m_e as M_E,
    m_p as M_P,
    N_A as N_AVO,
    sigma as SIGMA_SB,
)


# =====================================================================
# 外部主接口：external_call_metis_lh_model
# =====================================================================

def external_call_metis_lh_model(cons, profil, option):
    """
    Python 版 METIS external_call_metis_lh_model.m

    返回：
        time       : (nt,)
        plh_tot    : (nt,)   LH 吸收总功率 [W]
        ilh        : (nt,)   LH 驱动电流 [A]
        x          : (nx,)   r/a
        plh        : (nt,nx) 功率沉积 [W/m^3]
        jlh        : (nt,nx) 电流驱动“密度”
                     ——本版本已乘上 2πRaxe，对应你要的带 2πR 的定义
        efficiency : (nt,)   归一化效率 [A/W/m^2]
    """
    # ---------- cons ----------
    temps = np.asarray(cons["temps"]).reshape(-1)
    ip = np.asarray(cons["ip"]).reshape(-1)
    plh0 = np.asarray(cons["plh"]).reshape(-1)  # [W]
    nt = temps.size

    # ---------- profil ----------
    x = np.asarray(profil["xli"]).reshape(-1)  # (nx,)
    nx = x.size
    ve = np.ones(nx)
    vt = np.ones(nt)

    def as_2d(key):
        arr = np.asarray(profil[key])
        if arr.ndim == 1:
            arr = np.tile(arr.reshape(1, -1), (nt, 1))
        return arr

    Raxe = as_2d("Raxe")
    epsi = as_2d("epsi")
    fdia = as_2d("fdia")
    qjli = as_2d("qjli")
    nep = as_2d("nep")
    tep = as_2d("tep")
    rmx = as_2d("rmx")
    spr = as_2d("spr")
    vpr = as_2d("vpr")

    # zeff 剖面
    if "zeff" in profil:
        zeff_prof = as_2d("zeff")
        zeff = np.trapz(zeff_prof, x, axis=1)
    else:
        zeff_prof = None
        zeff = np.zeros(nt)

    # epar 剖面（只用来算 vloop_lh）
    if "epar" in profil:
        epar = as_2d("epar")
    else:
        epar = None

    # ------------ internal METIS parameters ------------
    transitoire = 1
    # backup value
    option_xlh = option.get("xlh", 0.2)
    option_dlh = option.get("dlh", 0.3)
    option_lhmode = option.get("lhmode", 0)
    option_upshiftmode = option.get("upshiftmode", "newmodel")
    # no ripple
    friplh = 0.0

    # directivity (正波瓣功率占比, 0–1)
    etalh = abs(option.get("etalh", 0.75))
    if option_lhmode in (0, 3, 4):
        directivity = etalh
    else:
        directivity = 0.75
    directivity = np.clip(directivity, 0.0, 1.0)

    # 机器标定因子：整体放大效率 / 电流，不改形状
    eta_scale = float(option.get("eta_scale", 1.0))

    xlhin = option_xlh * vt
    dlhin = option_dlh * vt

    # geo
    geo_a = Raxe[:, -1] * epsi[:, -1]
    geo_R = Raxe[:, -1]
    geo_b0 = fdia[:, -1] / Raxe[:, -1]

    # gas
    gaz = option.get("gaz", 2)
    if gaz == 1:
        agaz, zgaz = 1.0, 1.0
    elif gaz == 2:
        agaz, zgaz = 2.0, 1.0
    elif gaz == 3:
        agaz, zgaz = 2.5, 1.0
    elif gaz == 4:
        agaz, zgaz = 4.0, 2.0
    else:
        agaz, zgaz = 2.0, 1.0
    agaz_arr = agaz * vt
    zgaz_arr = zgaz * vt

    # qcyl & upshift
    qcyl = 5.0 * geo_a**2 * geo_b0 / (ip / 1e6) / geo_R
    qbord = qjli[:, -1]

    fupshift = option.get("fupshift", 1.0)
    if option_upshiftmode in ("newmodel", "newmodel + tail"):
        upshift_amp = fupshift * vt
    else:
        upshift_amp = fupshift * np.maximum(np.finfo(float).eps, qbord / qcyl - 1.0)

    # 调用两波瓣模型
    freqlh_Hz = option["freqlh"] * 1e9
    npar0 = option["npar0"]
    wlh = option["wlh"]
    npar_neg = option.get("npar_neg", None)

    x_void, fplh, xlh, dlh, efficiency, rapnegpos, lc, hc, acc, landau = z0lhacc2lobes(
        flh=freqlh_Hz,
        npar0=npar0,
        width=wlh,
        agaz=agaz_arr,
        zgaz=zgaz_arr,
        temps=temps,
        x=x,
        nep=nep,
        tep=tep,
        qp=qjli,
        Raxe=Raxe,
        rmx=rmx,
        spr=spr,
        vpr=vpr,
        Bt=geo_b0,
        plh=np.maximum(1.0, plh0),
        xlhin=xlhin,
        dlhin=dlhin,
        transitoire=transitoire,
        directivity=directivity,
        friplh=friplh,
        kx=upshift_amp,
        plotonoff=1,
        upshiftmode=option_upshiftmode,
        npar_neg=npar_neg,
    )

    # fplh: 复数 -> 实部功率形状，虚部电流形状
    fplh_real = np.real(fplh)
    fplh_imag = np.imag(fplh)

    if zeff_prof is not None:
        fjlh = (
            fplh_imag
            / (5.0 + zeff_prof)
            * (1.0 - epsi ** ((5.0 + zeff_prof) / 2.0 / (1.0 + zeff_prof)))
        )
    else:
        fjlh = fplh_imag.copy()

    fplh_abs = np.abs(fplh_real)

    # 归一化功率剖面：∫ vpr*fplh dx = cons.plh
    trap_v_fplh = np.trapz(vpr * fplh_abs, x, axis=1)
    normplh = np.maximum(1.0, plh0) / np.maximum(1.0, trap_v_fplh)
    fplh_abs = fplh_abs * (normplh[:, None] * ve)

    # jlh 初始形状（含负瓣 rapnegpos）
    jlh = fplh_abs * (
        (fplh_abs > 0).astype(float)
        + (rapnegpos[:, None] * ve) * (fplh_abs <= 0)
    )

    # 若 fjlh 有定义，则用 fjlh 覆盖
    indjlh = np.where(np.any(fjlh != 0, axis=1))[0]
    if indjlh.size > 0:
        jlh[indjlh, :] = fjlh[indjlh, :]

    # 若某些时间片 ∫S' jlh dx 太小，则改用平均正功率
    trap_spr_jlh = np.trapz(spr * jlh, x, axis=1)
    indbadlh = np.where(trap_spr_jlh < np.finfo(float).eps)[0]
    if indbadlh.size > 0:
        mean_pos = np.mean(fplh_abs * (fplh_abs > 0), axis=1)
        jlh[indbadlh, :] = np.outer(mean_pos[indbadlh], np.ones(nx))

    # 效率 Zeff 修正
    efficiency = (
        efficiency
        / (5.0 + zeff)
        * (
            1.0
            - (xlh * geo_a / geo_R)
            ** ((5.0 + zeff) / 2.0 / (1.0 + zeff))
        )
    )

    # 若有 fjlh & zeff 剖面，再用 fjlh 算一次效率
    if zeff_prof is not None and indjlh.size > 0:
        num_eff_full = np.trapz(spr * fjlh, x, axis=1)
        den_eff_full = np.maximum(
            1.0, np.trapz(spr * fplh_abs, x, axis=1)
        )
        nbar = np.trapz(nep, x, axis=1)
        efficiency_full = num_eff_full / den_eff_full * nbar
        efficiency[indjlh] = efficiency_full[indjlh]

    # vloop_lh（可选）
    if epar is not None and "plh" in profil:
        plh_prof_for_vloop = as_2d("plh")
        num_v = 2.0 * np.pi * np.trapz(
            Raxe * epar * plh_prof_for_vloop * spr, x, axis=1
        )
        den_v = np.maximum(
            1.0, np.trapz(plh_prof_for_vloop * spr, x, axis=1)
        )
        vloop_lh = num_v / den_v
    else:
        vloop_lh = np.zeros(nt)

    # nbar, xlh0, rhot, Ilh
    nbar = np.trapz(nep, x, axis=1)
    xlh0 = np.maximum(1.0, plh0) / nbar / geo_R
    rhot = (
        8.0
        * nbar
        * geo_R
        / xlh0
        / (efficiency**2)
        * (3.0 + zeff)
        / (5.0 + zeff) ** 2
    )

    # ilh = (plh>1) .* (efficiency.*xlh0 + vloop_lh./rhot)
    mask_plh = (plh0 > 1.0).astype(float)
    ilh = mask_plh * (efficiency * xlh0 + vloop_lh / rhot)

    # jlh 归一化到总电流（这里用 spr 做归一化）
    trap_spr_jlh = np.trapz(spr * jlh, x, axis=1)
    scale = np.where(
        trap_spr_jlh > np.finfo(float).eps,
        ilh / np.maximum(np.finfo(float).eps, trap_spr_jlh),
        0.0,
    )
    jlh = jlh * (scale[:, None] * ve)

    # 机器标定因子：整体放大电流和效率（不改剖面形状）
    ilh *= eta_scale
    jlh *= eta_scale
    efficiency *= eta_scale

    # ★★★ 关键一步：把 jlh 换成“乘了 2πR 的版本” ★★★
    # 也就是 jlh_new = jlh_old * (2πRaxe)，对应你要的带 2πR 的驱动电流定义
    # Raxe: (nt, nx)，每个磁面的大半径
    jlh = jlh * (2.0 * np.pi * Raxe)

    # plh = abs(fplh)；plh_tot = trapz(x,vpr.*fplh,2)
    plh_prof = np.abs(fplh_abs)
    plh_tot = np.trapz(vpr * fplh_abs, x, axis=1)

    time = temps.copy()

    return time, plh_tot, ilh, x, plh_prof, jlh, efficiency


# 给一个别名，方便调用
metis_lh = external_call_metis_lh_model


# =====================================================================
# z0lhacc2lobes.m 的 Python 版
# =====================================================================

def z0lhacc2lobes(
    flh,
    npar0,
    width,
    agaz,
    zgaz,
    temps,
    x,
    nep,
    tep,
    qp,
    Raxe,
    rmx,
    spr,
    vpr,
    Bt,
    plh,
    xlhin,
    dlhin,
    transitoire,
    directivity,
    friplh,
    kx,
    plotonoff,
    upshiftmode,
    npar_neg=None,
):
    """
    Python 版 z0lhacc2lobes.m
    """
    nt, nx = nep.shape
    ve = np.ones(nx)

    # upshiftmode 对 npar_neg 的处理（跟 METIS 一致）
    if upshiftmode in ("newmodel", "newmodel + tail"):
        if npar_neg is None or (npar_neg > -1):
            npar_neg = -npar0
    else:
        if npar_neg is None or (npar_neg > -1):
            npar_neg = -npar0 * np.pi

    directivity = float(np.clip(abs(directivity), 0.0, 1.0))

    # ========= 正波瓣 (+n_par) =========
    (
        x_out,
        fpoutp,
        xlh,
        dlh,
        lc,
        hc,
        acc,
        landau,
        efficiency_p,
        effacc_p,
        fjlh_p,
    ) = z0lhacc(
        flh=flh,
        npar0=npar0,
        width=width,
        agaz=agaz,
        zgaz=zgaz,
        temps=temps,
        x=x,
        nep=nep,
        tep=tep,
        qp=qp,
        Raxe=Raxe,
        rmx=rmx,
        spr=spr,
        vpr=vpr,
        Bt=Bt,
        plh=plh * directivity,
        xlhin=xlhin,
        dlhin=dlhin,
        transitoire=transitoire,
        upshift=kx,
        plotonoff=plotonoff,
        upshiftmode=upshiftmode,
    )

    real_fpoutp = np.real(fpoutp)
    trap_v_p = np.trapz(vpr * real_fpoutp, x, axis=1)
    fjlhoutp = np.imag(fpoutp) * (
        (plh * directivity / np.maximum(1.0, trap_v_p))[:, None] * ve
    )
    fpoutp = real_fpoutp * (
        (plh * directivity / np.maximum(1.0, trap_v_p))[:, None] * ve
    )

    # ========= 负波瓣 (-n_par) =========
    (
        x_out2,
        fpoutn,
        xlh_n,
        dlh_n,
        lc_n,
        hc_n,
        acc_n,
        landau_n,
        efficiency_n,
        effacc_n,
        fjlh_n,
    ) = z0lhacc(
        flh=flh,
        npar0=npar_neg,
        width=width,
        agaz=agaz,
        zgaz=zgaz,
        temps=temps,
        x=x,
        nep=nep,
        tep=tep,
        qp=qp,
        Raxe=Raxe,
        rmx=rmx,
        spr=spr,
        vpr=vpr,
        Bt=Bt,
        plh=plh * np.maximum(0.01, 1.0 - directivity - friplh),
        xlhin=xlhin,
        dlhin=dlhin,
        transitoire=transitoire,
        upshift=kx,
        plotonoff=plotonoff,
        upshiftmode=upshiftmode,
    )

    real_fpoutn = np.real(fpoutn)
    trap_v_n = np.trapz(vpr * real_fpoutn, x, axis=1)
    fjlhoutn = np.imag(fpoutn) * (
        (
            plh * np.maximum(0.0, 1.0 - directivity - friplh)
            / np.maximum(1.0, trap_v_n)
        )[:, None]
        * ve
    )
    fpoutn = real_fpoutn * (
        (
            plh * np.maximum(0.0, 1.0 - directivity - friplh)
            / np.maximum(1.0, trap_v_n)
        )[:, None]
        * ve
    )

    # rapnegpos = min(1, efficiency_n ./ max(1,efficiency_p));
    rapnegpos = np.minimum(1.0, efficiency_n / np.maximum(1.0, efficiency_p))

    # 合并：正波瓣 - 负波瓣
    fpout = fpoutp - fpoutn + 1j * (fjlhoutp - fjlhoutn)

    # 全局效率：efficiency_p * ( directivity - rapnegpos*(1-directivity-friplh) )
    efficiency_global = efficiency_p * (
        directivity - rapnegpos * np.maximum(0.0, 1.0 - directivity - friplh)
    )

    return x_out, fpout, xlh, dlh, efficiency_global, rapnegpos, lc, hc, acc, landau


# =====================================================================
# z0lhacc.m 的 Python 版（完整 upshiftmode 分支）
# =====================================================================

def z0lhacc(
    flh,
    npar0,
    width,
    agaz,
    zgaz,
    temps,
    x,
    nep,
    tep,
    qp,
    Raxe,
    rmx,
    spr,
    vpr,
    Bt,
    plh,
    xlhin,
    dlhin,
    transitoire,
    upshift,
    plotonoff,
    upshiftmode="newmodel",
):
    """
    Python 版 z0lhacc.m（严格按原始逻辑翻译）
    """

    phys_c = C_LIGHT
    phys_e = E_CHARGE
    phys_eps0 = EPS0
    phys_me = M_E
    phys_mp = M_P

    temps = np.asarray(temps).reshape(-1)
    x = np.asarray(x).reshape(-1)
    nep = np.asarray(nep)
    tep = np.asarray(tep)
    qp = np.asarray(qp)
    Raxe = np.asarray(Raxe)
    rmx = np.asarray(rmx)
    spr = np.asarray(spr)
    vpr = np.asarray(vpr)
    Bt = np.asarray(Bt).reshape(-1)
    plh = np.asarray(plh).reshape(-1)
    xlhin = np.asarray(xlhin).reshape(-1)
    dlhin = np.asarray(dlhin).reshape(-1)
    upshift = np.asarray(upshift).reshape(-1)

    nt, nx = nep.shape
    ve = np.ones(nx)
    vt = np.ones(nt)

    # lobe 标志
    if npar0 < 0:
        lobe = -1.0
        npar0 = abs(npar0)
    else:
        lobe = 1.0

    npar0 = max(1.0, npar0)

    # B 场
    btor = Bt[:, None] * ve
    bpol = np.sign(vt[:, None] * x[None, :]) * rmx * btor / Raxe / qp
    btot = np.sqrt(btor**2 + bpol**2)

    qp = np.maximum(1.0, np.minimum(qp[:, -1][:, None] * ve, qp))

    # upshift 分布
    kinetic_factor = np.ones_like(qp)
    if upshiftmode == "1/q":
        upshift_mat = (
            (upshift[:, None] * ve)
            * (
                (1.0 / qp - 1.0 / (np.max(qp, axis=1)[:, None] * ve))
                / (
                    1.0 / (np.min(qp, axis=1)[:, None] * ve)
                    - 1.0 / (np.max(qp, axis=1)[:, None] * ve)
                )
            )
            ** 2
        )
    elif upshiftmode == "Bpol":
        upbp = -np.cumsum(
            (bpol[:, ::-1] / btor[:, ::-1]) * np.diff(x[::-1])[0],
            axis=1,
        )
        upbp = upbp[:, ::-1] / np.maximum(
            np.finfo(float).eps, np.max(upbp, axis=1)[:, None] * ve
        )
        upshift_mat = np.clip(
            (upshift[:, None] * ve) * upbp, -0.9, 10.0
        )
    elif upshiftmode == "x^2":
        upshift_mat = (upshift[:, None] * ve) * (
            1.0 - (rmx / (rmx[:, -1][:, None] * ve)) ** 2
        )
    elif upshiftmode == "sqrt(x)":
        upshift_mat = (upshift[:, None] * ve) * (
            1.0 - np.sqrt(rmx / (rmx[:, -1][:, None] * ve))
        )
    elif upshiftmode == "null":
        upshift_mat = np.zeros_like(qp)
    elif upshiftmode in ("newmodel", "newmodel + tail"):
        kinetic_factor = upshift[:, None] * ve
        kinetic_factor[kinetic_factor == 0] = 1.0
        upshift_mat = np.zeros_like(qp)
    elif upshiftmode == "step@edge":
        upshift_mat = upshift[:, None] * ve
    else:
        upshift_mat = (upshift[:, None] * ve) * (
            1.0 - rmx / (rmx[:, -1][:, None] * ve)
        )

    # log Lambda
    lnl = 14.9 - 0.5 * np.log(nep / 1e20) + np.log(tep / 1e3)

    # 频率与 Stix 参数
    w = 2.0 * np.pi * flh
    wpe = np.sqrt(nep * phys_e**2 / phys_me / phys_eps0)
    wpi = np.sqrt(
        nep / (zgaz[:, None] * ve)
        * phys_e**2
        / phys_mp
        / (agaz[:, None] * ve)
        / phys_eps0
    )
    wce = phys_e / phys_me * btot

    S = 1.0 + wpe**2 / wce**2 - wpi**2 / w**2
    P = 1.0 - wpe**2 / w**2 - wpi**2 / w**2

    # Landau 共振
    landau = 6.5 / np.sqrt(np.maximum(30.0, tep) / 1e3)

    # newmodel + tail 进一步修改 upshift_mat
    if upshiftmode == "newmodel":
        upshift_mat = np.zeros_like(qp)
    elif upshiftmode == "newmodel + tail":
        upshift_mat = np.maximum(0.0, landau[:, [0]] - npar0) * (1.0 - x[None, :])

    # lc, hc, acc
    sqrt_term = np.sqrt(
        np.maximum(np.finfo(float).eps, -P / S)
    )
    lc = (npar0 + upshift_mat) / (
        1.0 + rmx / qp / Raxe * sqrt_term
    )
    hc = (npar0 + upshift_mat) / (
        1.0 - rmx / qp / Raxe * sqrt_term
    )
    acc = wpe / wce + np.sqrt(
        1.0 + wpe**2 / wce**2 - wpi**2 / w**2
    )
    acc[np.iscomplex(acc)] = 1000.0

    indbad = np.where((hc < lc) | (hc > (6.5 / np.sqrt(0.03))))[0]
    hc[indbad, :] = 6.5 / np.sqrt(0.03)
    hc[np.iscomplex(hc)] = npar0
    hc[~np.isfinite(hc)] = npar0

    # 自然谱宽
    dn0 = phys_c / flh / width

    # ================= newmodel / newmodel+tail =================
    if upshiftmode in ("newmodel", "newmodel + tail"):
        effacc = np.minimum(
            1.0, np.exp((npar0 - acc[:, -1]) / dn0)
        )[:, None] * ve

        factacc = np.minimum(
            1.0, np.exp((landau - (acc + dn0)) / dn0)
        )

        if upshiftmode == "newmodel":
            dlhabs = (dn0 + npar0 * width / Raxe) * (1.0 + landau / npar0)
        else:
            dlhabs = dn0 + npar0 * width / Raxe

        factlc = np.minimum(
            1.0,
            np.exp((landau - (lc + dlhabs / 2.0)) / dlhabs),
        )
        facthc = np.minimum(
            1.0,
            np.exp(((hc - dlhabs / 2.0) - landau) / dlhabs),
        )

        pabs = np.exp(
            -(kinetic_factor * landau - (npar0 + dlhabs / 2.0 + upshift_mat)) ** 2
            / (dlhabs**2)
        ) * factacc * factlc * facthc

        max_p = np.max(pabs, axis=1)
        max_p[max_p <= 0] = 1.0
        pabs = pabs / (max_p[:, None] * ve)

    # ================= old model 分支 =================
    else:
        if lobe < 0:
            dn0 = np.pi * dn0

        effacc = np.minimum(
            1.0, np.exp((npar0 - acc[:, -1]) / dn0)
        )[:, None] * ve

        # 默认在 Landau 最小值吸收
        dhc = landau.copy()
        mask = dhc == (np.min(dhc, axis=1)[:, None] * ve)
        indmat = np.ones_like(dhc) * np.arange(1, nx + 1)[None, :]
        indlh = np.maximum(
            1,
            np.round(
                np.sum(mask * indmat, axis=1) / np.maximum(1.0, np.sum(mask, axis=1))
            ).astype(int),
        )

        vali = np.zeros(nt)
        landau2 = landau.copy()
        landau2[
            (landau < acc)
            | (landau < lc)
            | (landau > hc)
        ] = np.inf

        dd = np.abs(landau2 - npar0 - upshift_mat)
        ddmin = np.min(dd, axis=1)
        indlhs = np.argmin(dd, axis=1)

        indok = np.where(np.isfinite(ddmin))[0]
        if indok.size > 0:
            indlh[indok] = indlhs[indok]
        if nt > 5:
            indlh = np.round(
                (np.roll(indlh, 1) + indlh + np.roll(indlh, -1)) / 3.0
            ).astype(int)

        vali[indok] = 1.0
        vali_mat = vali[:, None] * ve

        xlh_old = x[np.clip(indlh - 1, 0, nx - 1)]

        nparabs = landau[np.arange(nt), np.clip(indlh - 1, 0, nx - 1)]
        nparwave = npar0 + upshift_mat
        nparu = nparwave[
            np.arange(nt), np.clip(indlh - 1, 0, nx - 1)
        ]

        indnonok = np.where(~np.isfinite(nparabs))[0]
        if indnonok.size > 0:
            nparabs[indnonok] = np.mean(nparabs[indnonok])
        nparabs_mat = nparabs[:, None] * ve
        nparu_mat = nparu[:, None] * ve

        nparabs_mat = np.maximum(nparu_mat, nparabs_mat)

        fact = 1.0 + 0.5 * np.maximum(
            0.0, (nparabs_mat - nparu_mat) / dn0
        )
        dlhabs = np.maximum(
            dn0 * fact + width / Raxe,
            ((1.0 - vali_mat) * npar0) * ve,
        )

        maskxmax = (xlh_old[:, None] * ve) > (vt[:, None] * x[None, :])

        pabs = np.exp(
            -0.5
            * (
                nparabs_mat
                - nparu_mat
                + npar0
                + upshift_mat
                - landau
            )
            ** 2
            / (dlhabs**2)
        )
        pabs_u = np.exp(
            -0.5
            * (
                nparabs_mat
                - nparu_mat
                + npar0
                + upshift_mat
                - np.minimum(hc, landau)
            )
            ** 2
            / (dlhabs**2)
        )

        indi = np.where(vali > 0)[0]
        if indi.size > 0:
            pabs[indi, :] = (
                pabs[indi, :] * (maskxmax[indi, :] == 0)
                + pabs_u[indi, :] * (maskxmax[indi, :] != 0)
            )

        max_p = np.max(pabs, axis=1)
        max_p[max_p <= 0] = 1.0
        pabs = pabs / (max_p[:, None] * ve)

    # 默认高斯
    fplh_default = np.exp(
        -(vt[:, None] * x[None, :] - (xlhin[:, None] * ve)) ** 2
        / ((np.maximum(dlhin[:, None], 0.05) * ve) ** 2)
        / 2.0
    )

    indni = np.where(
        (np.sum(pabs, axis=1) == 0) | (~np.isfinite(pabs).all(axis=1))
    )[0]
    if indni.size > 0:
        pabs[indni, :] = fplh_default[indni, :]

    max_p2 = np.max(pabs, axis=1)
    max_p2[max_p2 <= 0] = 1.0
    pabs = pabs / (max_p2[:, None] * ve)

    # 效率计算
    if upshiftmode in ("newmodel", "newmodel + tail"):
        w1 = np.minimum(1.0 / landau, 1.0 / npar0 - np.finfo(float).eps)
        w2 = 1.0 / npar0

        Dpar = (
            0.32
            * ((plh[:, None] / 1e6) * ve)
            / np.maximum(width, rmx)
            / Raxe
            * np.sqrt(tep / 1e3)
            / (nep / 1e19) ** 1.5
            / np.sqrt(S)
            * landau**2
            / np.maximum(dlhabs, np.finfo(float).eps)
        )

        quasi = 1.0 + 0.5 * (1.0 + np.tanh(np.log(np.maximum(np.finfo(float).eps, Dpar)) - np.log(0.1)))
        f1 = 31e20 / lnl
        efficiency_loc = (
            effacc
            * np.maximum(
                0.0,
                f1
                * quasi
                * (w2**2 - w1**2)
                / np.maximum(np.finfo(float).eps, np.log(w2 / w1)),
            )
        )

        jlh_loc = pabs * efficiency_loc / np.maximum(1.0, nep)

        efficiency = np.trapz(
            pabs * efficiency_loc * (vt[:, None] * x[None, :]), x, axis=1
        ) / np.maximum(
            np.finfo(float).eps,
            np.trapz(pabs * (vt[:, None] * x[None, :]), x, axis=1),
        )

    else:
        w1 = 1.0 / np.minimum(hc, np.maximum(npar0, landau))
        w2multi = 1.0 / np.maximum(acc, lc)
        w2single = 1.0 / npar0
        rap = np.zeros_like(pabs)
        w2 = w2multi * rap + w2single * (1.0 - rap)
        Dpar = (
            0.32
            * ((plh[:, None] / 1e6) * ve)
            / np.maximum(width, rmx)
            / Raxe
            * np.sqrt(tep / 1e3)
            / (nep / 1e19) ** 1.5
            / np.sqrt(S)
            * landau**2
            / np.maximum(dlhabs, np.finfo(float).eps)
        )
        quasi = 4.0 + 2.0 * (
            1.0 + np.tanh(np.log(np.maximum(np.finfo(float).eps, Dpar)) - np.log(0.1))
        )
        f1 = 0.5
        efficiency = (
            effacc[:, 0]
            * np.trapz(
                pabs
                * np.maximum(
                    0.0,
                    f1
                    * quasi
                    * (w2**2 - w1**2)
                    / np.maximum(
                        np.finfo(float).eps,
                        np.log(
                            np.maximum(w2, w1)
                            / np.maximum(w1, np.finfo(float).eps)
                        ),
                    ),
                )
                * (vt[:, None] * x[None, :]),
                x,
                axis=1,
            )
            / np.maximum(
                np.finfo(float).eps,
                np.trapz(pabs * (vt[:, None] * x[None, :]), x, axis=1),
            )
            * 1e20
        )
        jlh_loc = np.zeros_like(pabs)

    # xlh & dlh
    mask_max = pabs == (np.max(pabs, axis=1)[:, None] * ve)
    xlh = np.sum((vt[:, None] * x[None, :]) * mask_max, axis=1) / np.maximum(
        1.0, np.sum(mask_max, axis=1)
    )
    dlh = np.maximum(
        0.05,
        np.sqrt(
            np.trapz(
                np.abs(vt[:, None] * x[None, :])
                * pabs
                * (vt[:, None] * x[None, :] - xlh[:, None] * ve) ** 2,
                x,
                axis=1,
            )
            / np.maximum(
                np.finfo(float).eps,
                np.trapz(
                    np.abs(vt[:, None] * x[None, :]) * pabs, x, axis=1
                ),
            )
        ),
    )

    if transitoire == 1:
        fpout = pabs + 1j * jlh_loc
    else:
        fpout = vt[:, None] * np.mean(pabs, axis=0)[None, :]
        xlh = vt * np.mean(xlh)
        dlh = vt * np.mean(dlh)

    indp = np.where(plh < 1e3)[0]
    if indp.size > 0:
        fpout[indp, :] = 1.0 + 0j
        jlh_loc[indp, :] = 0.0

    fjlh = jlh_loc.copy()

    return x, fpout, xlh, dlh, lc, hc, acc, landau, efficiency, effacc, fjlh
