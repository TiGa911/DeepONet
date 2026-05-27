# -*- coding: utf-8 -*-
"""
run_81481_metis_onetwo_from_txt.py

只针对 EAST #81481 @ 5.300 s：
- 从 TXT 文件读取 Te / ne / Ti 剖面（数据wu1.txt）
- 用 METIS-LH Python 模块计算 LHCD 功率沉积与电流驱动
- 把这些剖面写入 inone（通过 Namelist3）
- 读取 gfile，生成 g0_input
- 在对应目录下调用 ONETWO (onetwo_129_201)

运行方式：
    python run_81481_metis_onetwo_from_txt.py
"""

import os
import numpy as np
import subprocess

from Namelist3 import Namelist
from MDSplus.connection import Connection
import geqdsk

# 低杂波模块
from LHW_main import build_east_metis_inputs_from_txt
from LHW_metis import external_call_metis_lh_model


def compute_lh_from_txt_for_onetwo(
    shot: int,
    time: float,
    txt_filename: str,
    eta_scale: float = 1.0,
    npar0: float = 2.0,
    wlh: float = 0.4,
    freqlh_GHz: float = 4.6,
    gaz: int = 2,
    etalh: float = 0.75,
    lhmode: int = 0,
    upshiftmode: str = "newmodel",
    fupshift: float = 1.0,
    xlh0: float = 0.2,
    dlh0: float = 0.3,
    z_eff_value: float = 2.0,
):
    """
    从 txt 读取剖面，调用 METIS-LH，返回给 ONETWO 用的剖面。

    返回:
        rho_lower : (201,)   归一化半径 rho
        p_MW_m3   : (201,)   功率密度 (MW/m^3)
        j_MA_m2   : (201,)   电流密度 (MA/m^2)
    """
    # 1) 构造 METIS-LH 的 cons / profil / option
    cons, profil, option, rho_in, real_time, PLH_W = build_east_metis_inputs_from_txt(
        shot=shot,
        time=time,
        txt_filename=txt_filename,
        npar0=npar0,
        wlh=wlh,
        freqlh_GHz=freqlh_GHz,
        gaz=gaz,
        etalh=etalh,
        eta_scale=eta_scale,
        lhmode=lhmode,
        upshiftmode=upshiftmode,
        fupshift=fupshift,
        xlh0=xlh0,
        dlh0=dlh0,
        z_eff_value=z_eff_value,
    )

    # 2) 调用 METIS-LH 主函数
    time_arr, plh_tot, ilh, x, plh_prof, jlh_prof, eff = external_call_metis_lh_model(
        cons, profil, option
    )

    # 只取单一时间片
    j_A_m2 = jlh_prof[0, :]   # A/m^2 (flux-surface averaged)
    p_W_m3 = plh_prof[0, :]   # W/m^3

    # 3) 插值到 ONETWO 使用的 201 点 rho 网格 [0, 1]
    rho_lower = np.linspace(0.0, 1.0, 201)
    j_interp_A_m2 = np.interp(rho_lower, x, j_A_m2, left=0.0, right=0.0)
    p_interp_W_m3 = np.interp(rho_lower, x, p_W_m3, left=0.0, right=0.0)

    # 4) 单位转换：A/m^2 -> MA/m^2； W/m^3 -> MW/m^3
    j_MA_m2 = j_interp_A_m2 / 1e6
    p_MW_m3 = p_interp_W_m3 / 1e6

    return rho_lower, p_MW_m3, j_MA_m2


def main():
    # ========= 基本设置 =========
    shot = 81481
    time = 5.300      # s
    txt_path = "数据wu1.txt"   # Te/ne/Ti 剖面 TXT 文件

    # 结果目录：results/81481/005.30000s/onetwo
    time_str = f"{time:07.5f}s"
    output_base = "results"
    time_dir = time_str
    output_dir = os.path.join(output_base, f"{shot}", time_dir)
    onetwo_result_dir = os.path.join(output_dir, "onetwo")
    os.makedirs(onetwo_result_dir, exist_ok=True)

    # ========= 1. 读 inone_template，准备 Namelist =========
    obj = Namelist()
    obj.read("inone_template")

    # ========= 2. 从 TXT 读取 Te / ne / Ti，写入 NAMELIS1 =========
    # TXT 格式: rho ne te(exp) ti(exp) rho te(simu) ti(simu)
    data = np.loadtxt(txt_path, skiprows=1)
    rho_txt = data[:, 0]
    ne_1e19 = data[:, 1]
    te_simu_keV = data[:, 5]   # 如果想用实验值就改成 data[:, 2]
    ti_simu_keV = data[:, 6]   # 或 data[:, 3]

    # 写入 Te：RTEIN (rho), TEIN (eV)
    obj["NAMELIS1"]["RTEIN"] = list(rho_txt)
    obj["NAMELIS1"]["TEIN"] = list(te_simu_keV * 1e3)  # keV -> eV

    # 写入 ne：RENEIN (rho), ENEIN (cm^-3)
    # 原 txt 中 ne 单位是 1e19 m^-3，因此 *1e13 -> cm^-3
    obj["NAMELIS1"]["RENEIN"] = list(rho_txt)
    obj["NAMELIS1"]["ENEIN"] = list(ne_1e19 * 1e13)

    # 写入 Ti：RTIIN (rho), TIIN (keV)
    obj["NAMELIS1"]["RTIIN"] = list(rho_txt)
    obj["NAMELIS1"]["TIIN"] = list(ti_simu_keV)

    # ========= 3. 用 METIS-LH 计算 LHCD 剖面，写入 NAMELIS2 =========
    rho_lh, power_MW_m3, current_MA_m2 = compute_lh_from_txt_for_onetwo(
        shot=shot,
        time=time,
        txt_filename=txt_path,
        eta_scale=1.0,   # 这里就是你之前的 eta_scale，总体电流放大因子
    )

    # ONETWO 的 extcurrf/extqerf 设置，逻辑和 outonetwo.py 相同
    obj["NAMELIS2"]["extcurrf"] = [1.0]
    obj["NAMELIS2"]["extcurrf_id"] = ["lhw"]
    obj["NAMELIS2"]["extcurrf_nj"] = [len(rho_lh)]
    obj["NAMELIS2"]["extcurrf_rho"] = list(rho_lh)
    # MA/m^2 -> A/cm^2：乘 1e6 / 1e4
    obj["NAMELIS2"]["extcurrf_curr"] = list(current_MA_m2 * 1e6 / 1e4)
    obj["NAMELIS2"]["extcurrf_amps"] = [0.0]  # 不再额外归一化

    obj["NAMELIS2"]["extqerf"] = [1.0]
    obj["NAMELIS2"]["extqerf_id"] = ["lhw"]
    obj["NAMELIS2"]["extqerf_nj"] = [len(rho_lh)]
    obj["NAMELIS2"]["extqerf_rho"] = list(rho_lh)
    # 这里维持和你原 outonetwo.py 一样：直接用 MW/m^3 数字
    obj["NAMELIS2"]["extqerf_qe"] = list(power_MW_m3)
    obj["NAMELIS2"]["extqerf_watts"] = [0.0]

    obj["NAMELIS2"]["extqirf"] = [0.0]
    obj["NAMELIS2"]["extqirf_id"] = ["lhw"]
    obj["NAMELIS2"]["extqirf_nj"] = [0]

    # ========= 4. 写 inone =========
    inone_path = os.path.join(onetwo_result_dir, "inone")
    obj.write(inone_path)
    print(f"inone 已写入: {inone_path}")

    # ========= 5. 从 MDSplus 读取 gfile，写 g0_input =========
    conn = Connection("202.127.204.42")
    TREE = "efit_east"
    conn.openTree(TREE, shot)

    # 读取 gfile 时间数组，找到离 time 最近的一个
    g_times = conn.get(r"data(\GTIME)").data()
    time_index = int(np.argmin(np.abs(g_times - time)))
    gg = geqdsk.read_from_MDS(conn, time_index)
    conn.closeTree(TREE, shot)

    gfile_path = os.path.join(onetwo_result_dir, "g0_input")
    geqdsk.save(gg, gfile_path)
    print(f"gfile 已写入: {gfile_path}")

    # ========= 6. 运行 ONETWO =========
    print("开始运行 ONETWO (onetwo_129_201) ...")
    subprocess.run(["onetwo_129_201"], cwd=onetwo_result_dir, check=True)
    print("ONETWO 运行完成。")


if __name__ == "__main__":
    main()
