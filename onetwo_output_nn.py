# -*- coding: utf-8 -*-
"""
onetwo_output_nn.py — 基于神经网络的 ONETWO 输运计算流程

与 onetwo_output.py 的唯一区别：用 ProfileNet 神经网络推理替代了 mtanh+spline 拟合链。
其余逻辑（MDSplus 数据读取、LHW 低杂波计算、gfile 平衡文件、ONETWO 执行）完全相同。

使用方法:
    python onetwo_output_nn.py
    （交互式输入炮号，与原版相同）
"""

import geqdsk
import numpy as np
import os
from MDSplus.connection import Connection
import matplotlib
matplotlib.use('Agg')          # 非交互式后端，避免 GUI 弹窗
import matplotlib.pyplot as plt
import multiprocessing
import subprocess
from Namelist3 import Namelist
import shutil
import glob
import time
from lower_view_final import lower_onetwo

import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator

# ---- 神经网络推理接口 ----
from profile_nn.infer import nn_fit_te, nn_fit_ne, nn_fit_ti, set_model_dir

# 指定训练好的模型目录（可修改为其他路径）
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'profile_nn_models')
set_model_dir(MODEL_DIR)


def format_time_dirname(t: float) -> str:
    """将浮点时间转换为目录命名格式，如 3.5s -> '003.50000s'"""
    time_str = "{:.5f}".format(t)
    if '.' not in time_str:
        integer_part = time_str
        decimal_part = "00000"
    else:
        integer_part, decimal_part = time_str.split('.')
    integer_part = integer_part.zfill(3)          # 整数部分补零到 3 位
    decimal_part = decimal_part.ljust(5, '0')[:5] # 小数部分补零到 5 位
    return f"{integer_part}.{decimal_part}s"


def read_lhcd_data(filename):
    """读取低杂波电流驱动数据文件（rho, power, current 三列）"""
    rho, power, current = [], [], []
    with open(filename, 'r') as f:
        next(f)  # 跳过标题行
        for line in f:
            data = line.strip().split('\t')
            rho.append(float(data[0]))
            power.append(float(data[1]))
            current.append(float(data[2]))
    return np.array(rho), np.array(power), np.array(current)


from readMDS_onetwo import readmds


def process_time_point(args):
    """处理单个时间片的完整 ONETWO 流程（使用神经网络拟合剖面）

    流程：
    1. 从 EAST MDSplus 读取该时间点的诊断数据
    2. 用 ProfileNet 分别拟合 Te、ne、Ti 剖面
    3. 计算 LHW 低杂波功率沉积和电流驱动
    4. 组装 ONETWO Namelist 输入文件
    5. 从 MDSplus 读取 gfile 平衡文件
    6. 执行 ONETWO 输运计算

    Args:
        args: (时间点, 炮号) 元组，用于 multiprocessing.Pool.map
    """
    i, shot = args
    time = float(i)

    output_base = "results"
    time_dir = format_time_dirname(time)
    output_dir = os.path.join(output_base, f"{shot}", time_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ---- 读取 MDSplus 诊断数据 ----
    data, status, real_time = readmds(shot, time)

    R_inone = np.linspace(0, 1, 201)  # ONETWO 剖面网格：rho = 0~1，共 201 点

    obj = Namelist()
    obj.read('inone_template')  # 读取 ONETWO 输入模板

    TE_STATUS = 0   # 0=成功, 1=失败/跳过
    NE_STATUS = 0

    te_ped_x = None  # Te 台基坐标（ρ ≥ 0.88），供 Ti 拟合使用
    te_ped_y = None  # Te 台基值

    # =====================================================================
    # 第一步：电子温度 Te 剖面拟合（汤姆逊散射 TS 数据 → ProfileNet）
    # =====================================================================
    if status['TS_status'] == 1:
        try:
            x, y, datatype = data['Te']['TS']['Rho'], data['Te']['TS']['data'], data['Te']['TS']['type']

            # ---- 神经网络推理：替代 fitting() + robust_interp() ----
            # 输入 x(ρ), y(Te[eV]) → 输出 x_201(201点ρ网格), y_201(Te[keV])
            x_201, y_201 = nn_fit_te(x, y)

            # 提取 Te 台基数据（ρ ≥ 0.88 区域），供后续 Ti 拟合使用
            mask = (x_201 >= 0.88)
            te_ped_x = x_201[mask]
            te_ped_y = y_201[mask]

            # ---- 绘制并保存拟合结果图 ----
            plt.figure(figsize=(8, 5))
            plt.scatter(x, y / 1000., marker='.', c='g', label='TS raw')   # 原始数据 eV→keV
            plt.plot(x_201, y_201, 'b-', linewidth=2, label='NN fitted')   # NN 拟合结果
            plt.xlabel(r'$\rho$')
            plt.ylabel('Te(TS) (keV)')
            plt.title(f'Shot {shot} @ {real_time:.3f}s [NN]')
            plt.xlim([0, 1])
            plt.legend()

            save_path = os.path.join(output_dir, f'Te_shot{shot}_time{time_dir}_nn.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f'(NN_Te) success at {real_time:.3f}s -> {save_path}')

            # 写入 ONETWO 的 NAMELIS1：TEIN 需要 eV 单位
            obj['NAMELIS1']['RTEIN'] = list(x_201)
            obj['NAMELIS1']['TEIN'] = list(y_201 * 1e3)  # keV -> eV
        except Exception as e:
            print(f'(NN_Te) Error at {time}s: {str(e)}')
            TE_STATUS = 1
    else:
        TE_STATUS = 1  # TS 诊断不可用，标记跳过

    # =====================================================================
    # 第二步：电子密度 ne 剖面拟合（反射计 Refl 数据 → ProfileNet）
    # =====================================================================
    if status['Refl_status'] == 1:
        try:
            x, y, datatype = data['ne']['Refl']['Rho'], data['ne']['Refl']['data'], data['ne']['Refl']['type']

            # ---- 神经网络推理：替代 fitting() + robust_interp() ----
            # 输入 x(ρ), y(ne[10^19 m^-3]) → 输出 x_201, y_201（同单位）
            x_201, y_201 = nn_fit_ne(x, y)

            # ---- 绘制并保存拟合结果图 ----
            plt.figure(figsize=(8, 5))
            plt.scatter(x, y, marker='.', c='g', label='Refl raw')
            plt.plot(x_201, y_201, 'b-', linewidth=2, label='NN fitted')
            plt.xlabel(r'$\rho$')
            plt.ylabel('ne(Refl) (10$^{19}$ m$^{-3}$)')
            plt.title(f'Shot {shot} @ {real_time:.3f}s [NN]')
            plt.xlim([0, 1])
            plt.legend()

            save_path = os.path.join(output_dir, f'ne_shot{shot}_time{time_dir}_nn.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f'(NN_ne) success at {real_time:.3f}s -> {save_path}')

            # 写入 ONETWO 的 NAMELIS1：ENEIN 需要 cm^-3 单位
            obj['NAMELIS1']['RENEIN'] = list(x_201)
            obj['NAMELIS1']['ENEIN'] = list(y_201 * 1e13)  # 10^19 m^-3 -> cm^-3
        except Exception as e:
            print(f'(NN_ne) Error at {time}s: {str(e)}')
            NE_STATUS = 1
    else:
        NE_STATUS = 1  # Refl 诊断不可用，标记跳过

    # =====================================================================
    # 第三步：离子温度 Ti 剖面拟合（TXCS X 射线晶体谱仪 → ProfileNet_Ti）
    # 需要 Te 台基数据作为辅助输入（TXCS 测点通常很稀疏，尤其在台基区域）
    # =====================================================================
    if status['TXCS_status'] == 1 and te_ped_x is not None and te_ped_y is not None:
        try:
            x, y, datatype = data['Ti']['TXCS']['Rho'], data['Ti']['TXCS']['data'], data['Ti']['type']

            # ---- 神经网络推理（双编码器，带 Te 台基辅助信息） ----
            # 替代 fitting() + robust_interp() + enforce_monotone_pchip()
            x_201, y_201 = nn_fit_ti(x, y, te_ped_x, te_ped_y)

            # ---- 绘制并保存拟合结果图 ----
            plt.figure(figsize=(8, 5))
            plt.scatter(x, y, marker='.', c='g', label='TXCS raw')
            plt.plot(x_201, y_201, 'b-', linewidth=2, label='NN fitted')
            plt.xlabel(r'$\rho$')
            plt.ylabel('Ti(TXCS) (keV)')
            plt.title(f'Shot {shot} @ {real_time:.3f}s [NN]')
            plt.xlim([0, 1])
            plt.legend()

            save_path = os.path.join(output_dir, f'Ti_shot{shot}_time{time_dir}_nn.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f'(NN_Ti) success at {real_time:.3f}s -> {save_path}')

            # 写入 ONETWO 的 NAMELIS1：TIIN 直接用 keV，无需单位转换
            obj['NAMELIS1']['RTIIN'] = list(x_201)
            obj['NAMELIS1']['TIIN'] = list(y_201)
        except Exception as e:
            print(f'(NN_Ti) Error at {time}s: {str(e)}')
    elif status['TXCS_status'] == 1:
        print(f'(NN_Ti) Skipped at {time}s: No Te pedestal data available')

    # =====================================================================
    # 第四步：LHW 低杂波功率沉积与电流驱动计算 + ONETWO 输运求解
    # 仅在 Te 和 ne 都成功拟合时才执行
    # =====================================================================
    if (TE_STATUS == 0 and NE_STATUS == 0):
        try:
            onetwo_result_dir = os.path.join(output_base, f"{shot}", time_dir, 'onetwo_nn')
            os.makedirs(onetwo_result_dir, exist_ok=True)

            # ---- 调用 lower_onetwo 计算 LHW 功率沉积和电流驱动 ----
            rho, power, current = lower_onetwo(shot, time, onetwo_result_dir)

            # ---- 组装 ONETWO Namelist ----
            # NAMELIS2：外部电流驱动（LHW）
            obj['NAMELIS2']['extcurrf'] = [1.0]
            obj['NAMELIS2']['extcurrf_id'] = ['lhw']
            obj['NAMELIS2']['extcurrf_nj'] = [len(rho)]
            obj['NAMELIS2']['extcurrf_rho'] = list(rho)
            obj['NAMELIS2']['extcurrf_curr'] = list(current * 1e6 / 1e4)  # MA/m^2 -> A/cm^2
            obj['NAMELIS2']['extcurrf_amps'] = [0.0]

            # NAMELIS2：外部功率沉积（LHW 电子加热）
            obj['NAMELIS2']['extqerf'] = [1.0]
            obj['NAMELIS2']['extqerf_id'] = ['lhw']
            obj['NAMELIS2']['extqerf_nj'] = [len(rho)]
            obj['NAMELIS2']['extqerf_rho'] = list(rho)
            obj['NAMELIS2']['extqerf_qe'] = list(power)
            obj['NAMELIS2']['extqerf_watts'] = [0.0]

            # NAMELIS2：外部离子加热源（LHW 对离子无直接加热，设为 0）
            obj['NAMELIS2']['extqirf'] = [0.0]
            obj['NAMELIS2']['extqirf_id'] = ['lhw']
            obj['NAMELIS2']['extqirf_nj'] = [0]

            # NAMELIS2：ECRH 电子回旋共振加热参数（当前不启用，占位配置）
            obj['NAMELIS2']['rfmode'] = ['ech', 'ech', 'ech', 'ech']
            obj['NAMELIS2']['genraydat'] = ['', '', '', '']
            obj['NAMELIS2']['rfon'] = [-40.0, -40.0, -40.0, -40.0]
            obj['NAMELIS2']['rftime'] = [20e3, 20e3, 20e3, 20e3]
            obj['NAMELIS2']['rfpow'] = [4e5, 4e5, 4e5, 4e5]
            obj['NAMELIS2']['freq'] = [140e9, 140e9, 140e9, 140e9]
            obj['NAMELIS2']['xec'] = [300.0, 300.0, 300.0, 300.0]
            obj['NAMELIS2']['zec'] = [-30, -30, -30, -30]
            obj['NAMELIS2']['thetec'] = [86.0, 80.0, 86.2, 86.0]
            obj['NAMELIS2']['phaiec'] = [200.0, 200.0, 200.0, 0.0]
            obj['NAMELIS2']['irfcur'] = [1.0, 1.0, 1.0, 1.0]
            obj['NAMELIS2']['wrfo'] = [0.0, 0.0, 0.0, 0.0]
            obj['NAMELIS2']['nray'] = [30.0, 30.0, 30.0, 30.0]
            obj['NAMELIS2']['idamp'] = [2.0, 2.0, 2.0, 2.0]
            obj['NAMELIS2']['hlwec'] = [1.2, 1.2, 2.0, 1.2]
            obj['NAMELIS2']['ratwec'] = [1.0, 1.0, 1.0, 1.0]

            # ---- 写入 ONETWO 输入文件 ----
            onetwo_save_path = os.path.join(onetwo_result_dir, 'inone')
            obj.write(onetwo_save_path)
            print('ecrh_written (NN)')

            # ---- 从 MDSplus 读取 gfile 平衡文件 ----
            conn = Connection('202.127.204.42')
            TREE = 'efit_east'
            conn.openTree(TREE, shot)
            g_times = conn.get(r'data(\GTIME)').data()
            timeid = np.argmin(abs(g_times - time))  # 找最近的 gfile 时间点
            gg = geqdsk.read_from_MDS(conn, timeid)
            conn.closeTree(TREE, shot)

            filename = os.path.join(onetwo_result_dir, "g0_input")
            geqdsk.save(gg, filename)

            # ---- 执行 ONETWO 输运计算 ----
            subprocess.run(["onetwo_129_201"], cwd=onetwo_result_dir, check=True)
            print(f'(NN_onetwo) success at {time}s')

        except Exception as Error_onetwo:
            print(f'(NN_onetwo) fail at {time}s: {str(Error_onetwo)}')


def main():
    """主函数：多进程并行处理该炮号的所有时间点

    流程：
    1. 读取 TS 诊断的所有时间点
    2. 用 multiprocessing.Pool 并行处理每个时间点
    3. 每个时间点独立完成 剖面拟合 → LHW 计算 → ONETWO 输运
    """
    start_time = time.time()
    shot = int(input('Shot number: '))

    # ---- 获取该炮号所有 TS 时间点 ----
    conn = Connection('202.127.204.42')
    conn.openTree('TS_EAST', shot)
    TS_times = conn.get(r'dim_of(\Te_coreTS)').data()
    conn.closeTree('TS_EAST', shot)

    print(f'Found {len(TS_times)} time points')

    os.makedirs("results", exist_ok=True)

    # ---- 多进程并行处理每个时间点 ----
    pool = multiprocessing.Pool()
    task_args = [(t, shot) for t in TS_times]

    pool.map(process_time_point, task_args)
    pool.close()
    pool.join()

    # ---- 输出总耗时 ----
    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nTotal execution time: {int(hours):0>2}h {int(minutes):0>2}m {seconds:05.2f}s")


if __name__ == '__main__':
    main()
