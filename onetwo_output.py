# -*- coding: utf-8 -*-
"""
该模块用于批量处理 EAST 托卡马克实验数据，
生成 ONETWO 输运程序所需的输入文件（inone），
并调用外部 ONETWO 可执行程序进行计算。

主要功能：
1. 从 EAST MDSplus 服务器读取 Te（TS）、ne（Refl）、Ti（TXCS）诊断数据
2. 对剖面数据进行拟合、清洗、插值，生成 201 点标准网格
3. 将结果写入 inone 模板（Namelist 格式）
4. 从 lower_view_final 获取 LHW 功率/电流沉积剖面，作为外部源加入 inone
5. 读取平衡 gfile 并保存到 ONETWO 工作目录
6. 调用 onetwo_129_201 进行输运计算
7. 支持多进程并行处理多个时间片
"""

import geqdsk               # EFIT 平衡文件（gfile）读写模块
import numpy as np          # 数值计算
import os                   # 文件/目录操作
from MDSplus.connection import Connection  # EAST MDSplus 数据连接
from fitting_mtanh import fitting          # 剖面拟合（mtanh + spline）
import matplotlib
matplotlib.use('Agg')       # 无图形界面后端，适合服务器/批处理
import matplotlib.pyplot as plt            # 绘图并保存拟合结果
import multiprocessing      # 多进程并行，加速批量时间片处理
import subprocess           # 调用外部 ONETWO 可执行文件
from Namelist3 import Namelist             # Fortran Namelist 格式读写类
import shutil               # 文件拷贝/移动（本脚本中保留备用）
import glob                 # 通配符文件查找（保留备用）
import time                 # 计时
from lower_view_final import lower_onetwo  # 计算 LHW 功率与电流沉积剖面

import numpy as np
from scipy.interpolate import interp1d     # 一维插值
from scipy.interpolate import PchipInterpolator  # 保形单调插值（PCHIP）


def enforce_monotone_pchip(x, y, num_points=201, decreasing=True):
    """
    用 PCHIP（保形分段三次 Hermite 插值）强制生成单调剖面。

    等离子体剖面（Ti/Te/ne）通常在径向上单调递减，
    本函数先将数据取反，插值后再翻转回来，从而保证单调性。

    参数:
        x          : array-like, 归一化半径 rho（0~1，需递增）
        y          : array-like, 剖面值（如 Ti, Te, ne）
        num_points : 输出点数，默认 201（与 ONETWO 标准网格一致）
        decreasing : True=强制单调递减（默认），False=单调递增

    返回:
        x_new, y_new : 处理后的单调剖面
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # 按 rho 排序，防止输入不是严格递增
    order = np.argsort(x)
    x, y = x[order], y[order]

    # 若要求递减，先将 y 取反；插值后再翻转回来即可保证递减
    if decreasing:
        y_proc = -y
    else:
        y_proc = y

    # PCHIP 插值：相比普通 cubic spline，不会在数据间产生非物理振荡
    pchip = PchipInterpolator(x, y_proc)
    x_new = np.linspace(0, 1, num_points)
    y_new = pchip(x_new)

    # 翻转回原来的物理量方向
    if decreasing:
        y_new = -y_new

    return x_new, y_new


def robust_interp(x, y, datatype, thresholds=None, offsets=None, num_points=201):
    """
    鲁棒性插值：自动剔除过小异常点，必要时整体抬升数据，最终输出固定 201 点。

    处理逻辑：
      1. 根据 datatype 设定阈值，剔除 y 值过小的异常点；
      2. 对清洗后的数据做线性插值；
      3. 若插值后仍有极小值，则对原始数据整体加一个偏移量后重新插值；
      4. 最终输出 num_points 个均匀网格点（默认 201，匹配 ONETWO 需求）。

    参数：
      x, y       : 原始剖面数据（数组）
      datatype   : 物理量标识，"ne" / "Te" / "Ti"
      thresholds : 各物理量的最小阈值字典
      offsets    : 各物理量的抬升量字典
      num_points : 输出点数

    返回：
      x_fit, y_fit : 插值后的均匀网格剖面
    """
    # 默认阈值与抬升量（防止插值后出现非物理的 0 或负值）
    if thresholds is None:
        thresholds = {"ne": 0.1, "Te": 0.05, "Ti": 0.05}
    if offsets is None:
        offsets = {"ne": 0.05, "Te": 0.05, "Ti": 0.05}

    th = thresholds.get(datatype, 0.0)
    off = offsets.get(datatype, 0.0)

    x, y = np.array(x), np.array(y)

    # 第一步：剔除小于阈值的异常点
    mask = y >= th
    x_clean, y_clean = x[mask], y[mask]
    x_del, y_del = x[~mask], y[~mask]  # 仅记录，调试用

    # 若全部点都被剔除，则直接整体抬升原始数据并插值
    if len(y_clean) == 0:
        y_lifted = y + off
        f = interp1d(x, y_lifted, kind='linear', fill_value="extrapolate")
        x_fit = np.linspace(0, 1.0, num_points)
        y_fit = f(x_fit)
        return x_fit, y_fit

    # 第二步：线性插值到均匀网格
    f = interp1d(x_clean, y_clean, kind='linear', fill_value="extrapolate")
    x_fit = np.linspace(0, 1.0, num_points)
    y_fit = f(x_fit)

    # 第三步：若插值结果仍低于阈值，说明边界可能外推到异常值，整体抬升后重新插值
    if np.min(y_fit) < th:
        y_lifted = y + off
        f = interp1d(x, y_lifted, kind='linear', fill_value="extrapolate")
        x_fit = np.linspace(0, 1.0, num_points)
        y_fit = f(x_fit)

    return x_fit, y_fit


def format_time_dirname(t: float) -> str:
    """
    将时间（秒）格式化为固定宽度的目录名字符串，例如 5.3 -> "005.30000s"。
    便于按时间片组织输出结果，避免浮点数作为目录名导致的排序混乱。
    """
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
    """
    读取 LHW（低杂波）功率与电流沉积剖面的文本文件。

    文件格式：三列，分别为 rho、power（功率密度）、current（电流密度）
    返回：numpy 数组 rho, power, current
    """
    rho, power, current = [], [], []
    with open(filename, 'r') as f:
        next(f)  # 跳过表头行
        for line in f:
            data = line.strip().split('\t')
            rho.append(float(data[0]))
            power.append(float(data[1]))
            current.append(float(data[2]))
    return np.array(rho), np.array(power), np.array(current)


# 导入专为 ONETWO 流程封装的 readmds（与主流程 readMDS.py 不同）
from readMDS_onetwo import readmds


def process_time_point(args):
    """
    单个时间片的完整处理函数，供多进程池调用。

    处理流程：
      1. 读取 EAST MDSplus 诊断数据（Te, ne, Ti）
      2. 拟合并清洗剖面，保存拟合结果图
      3. 将剖面写入 inone Namelist（ONETWO 输入文件）
      4. 获取 LHW 外部源剖面并写入 inone
      5. 读取 gfile 并保存
      6. 调用 onetwo_129_201 进行输运计算

    参数:
        args : 元组 (time, shot)
    """
    i, shot = args
    time = float(i)

    # 构造输出目录结构：results/{shot}/{formatted_time}/
    output_base = "results"
    time_dir = format_time_dirname(time)
    output_dir = os.path.join(output_base, f"{shot}", time_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 调用 readmds 读取该时间点的所有诊断数据
    data, status, real_time = readmds(shot, time)

    # ONETWO 标准径向网格：201 点，rho 从 0 到 1
    R_inone = np.linspace(0, 1, 201)

    # 初始化 Namelist 对象，读取 inone 模板
    obj = Namelist()
    obj.read('inone_template')

    # 状态标记：0=成功，1=失败（用于决定是否执行 ONETWO）
    TE_STATUS = 0
    NE_STATUS = 0

    # 用于存储 Te 台基数据，供 Ti 拟合时作为边界参考
    te_ped_x = None
    te_ped_y = None

    # ========== 处理 Te（汤姆逊散射 TS）==========
    if status['TS_status'] == 1:
        try:
            x, y, datatype = data['Te']['TS']['Rho'], data['Te']['TS']['data'], data['Te']['TS']['type']

            # 绘制原始数据散点图
            plt.figure(figsize=(8, 5))
            plt.scatter(x, y/1000., marker='.', c='g')

            # 调用 mtanh+spline 拟合，返回清洗后的数据及被剔除的点
            x_te, y_te, x_del, y_del, q = fitting(x, y, datatype)

            # 绘制被剔除的异常点和拟合曲线
            plt.scatter(x_del, y_del/1000., marker='x', s=60, c='r')
            plt.plot(x_te, y_te, 'g--', linewidth=2, label='Fitted')
            plt.xlabel(r'$\rho$')
            plt.ylabel('Te(TS) (keV)')
            plt.title(f'Shot {shot} @ {real_time:.3f}s')
            plt.xlim([0, 1])
            plt.legend()

            # 保存图像到输出目录
            save_path = os.path.join(output_dir, f'Te_shot{shot}_time{time_dir}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f'(TS_Te) success at {real_time:.3f}s -> {save_path}')

            # 提取 Te 台基区域数据（rho >= 0.88），供后续 Ti 拟合使用
            mask = (0.88 <= x_te)
            te_ped_x = x_te[mask]
            te_ped_y = y_te[mask]
            print(f"Te pedestal data - x: {te_ped_x}, y: {te_ped_y}")

            # 插值为 201 点并写入 inone 的 NAMELIS1 区块
            x_fit, y_interp = robust_interp(x_te, y_te, datatype)
            obj['NAMELIS1']['RTEIN'] = list(x_fit)
            obj['NAMELIS1']['TEIN'] = list(y_interp)
        except Exception as e:
            print(f'(TS_Te) Error at {time}s: {str(e)}')
            TE_STATUS = 1
    else:
        TE_STATUS = 1

    # ========== 处理 ne（反射计 Refl）==========
    if status['Refl_status'] == 1:
        try:
            x, y, datatype = data['ne']['Refl']['Rho'], data['ne']['Refl']['data'], data['ne']['Refl']['type']

            plt.figure(figsize=(8, 5))
            plt.scatter(x, y, marker='.', c='g')
            x, y, x_del, y_del, q = fitting(x, y, datatype)
            plt.scatter(x_del, y_del, marker='x', s=60, c='r')
            plt.plot(x, y, 'g--', linewidth=2, label='Fitted')
            plt.xlabel(r'$\rho$')
            plt.ylabel('ne(Refl) (10$^{19}$ m$^{-3}$)')
            plt.title(f'Shot {shot} @ {real_time:.3f}s')
            plt.xlim([0, 1])
            plt.legend()

            save_path = os.path.join(output_dir, f'ne_shot{shot}_time{time_dir}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f'(Refl_ne) success at {real_time:.3f}s -> {save_path}')

            # 插值并写入 inone；注意单位转换：1e19 m^-3 -> 1e13 cm^-3（ONETWO 内部单位）
            x_fit, y_interp = robust_interp(x, y, datatype)
            obj['NAMELIS1']['RENEIN'] = list(x_fit)
            obj['NAMELIS1']['ENEIN'] = list(y_interp * 1e13)
        except Exception as e:
            print(f'(Refl_ne) Error at {time}s: {str(e)}')
            NE_STATUS = 1
    else:
        NE_STATUS = 1

    # ========== 处理 Ti（X射线晶体谱仪 TXCS）==========
    # Ti 拟合需要 Te 台基作为边界条件；若无 Te 台基则跳过
    if status['TXCS_status'] == 1 and te_ped_x is not None and te_ped_y is not None:
        try:
            x, y, datatype = data['Ti']['TXCS']['Rho'], data['Ti']['TXCS']['data'], data['Ti']['type']
            plt.figure(figsize=(8, 5))
            plt.scatter(x, y, marker='.', c='g')

            # 将 Te 台基信息传入 fitting，使 Ti 剖面在台基区域更合理
            x, y, x_del, y_del, q = fitting(x, y, datatype, te_ped_x=te_ped_x, te_ped_y=te_ped_y)

            # 鲁棒插值 + 强制单调递减（PCHIP）
            x, y = robust_interp(x, y, datatype)
            x, y = enforce_monotone_pchip(x, y)

            plt.scatter(x_del, y_del, marker='x', s=60, c='r')
            plt.plot(x, y, 'g--', linewidth=2, label='Fitted')
            plt.xlabel(r'$\rho$')
            plt.ylabel('Ti(TXCS) (keV)')
            plt.title(f'Shot {shot} @ {real_time:.3f}s')
            plt.xlim([0, 1])
            plt.legend()

            save_path = os.path.join(output_dir, f'Ti_shot{shot}_time{time_dir}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f'(Ti_TXCS) success at {real_time:.3f}s -> {save_path}')

            # 写入 inone
            x_fit, y_interp = robust_interp(x, y, datatype)
            obj['NAMELIS1']['RTIIN'] = list(x_fit)
            obj['NAMELIS1']['TIIN'] = list(y_interp)
        except Exception as e:
            print(f'(Ti_TXCS) Error at {time}s: {str(e)}')
    elif status['TXCS_status'] == 1:
        print(f'(Ti_TXCS) Skipped at {time}s: No Te pedestal data available')

    # ========== 执行 ONETWO 输运计算 ==========
    # 只有当 Te 和 ne 都成功读取时，才构造完整的 inone 并运行 ONETWO
    if (TE_STATUS == 0 and NE_STATUS == 0):
        try:
            # ONETWO 工作目录
            onetwo_result_dir = os.path.join(output_base, f"{shot}", time_dir, 'onetwo')
            os.makedirs(onetwo_result_dir, exist_ok=True)

            # 获取 LHW 功率沉积与电流驱动剖面
            rho, power, current = lower_onetwo(shot, time, onetwo_result_dir)

            # --- 将 LHW 电流驱动剖面写入外部电流源（extcurrf）---
            obj['NAMELIS2']['extcurrf'] = [1.0]          # 启用外部电流源
            obj['NAMELIS2']['extcurrf_id'] = ['lhw']     # 源标识
            obj['NAMELIS2']['extcurrf_nj'] = [len(rho)]  # 径向网格点数
            obj['NAMELIS2']['extcurrf_rho'] = list(rho)
            # 单位转换：MA/m^2 -> A/cm^2（ONETWO 要求）
            obj['NAMELIS2']['extcurrf_curr'] = list(current * 1e6 / 1e4)
            obj['NAMELIS2']['extcurrf_amps'] = [0.0]     # 不归一化，使用实际剖面值

            # --- 将 LHW 功率沉积剖面写入外部电子热源（extqerf）---
            obj['NAMELIS2']['extqerf'] = [1.0]
            obj['NAMELIS2']['extqerf_id'] = ['lhw']
            obj['NAMELIS2']['extqerf_nj'] = [len(rho)]
            obj['NAMELIS2']['extqerf_rho'] = list(rho)
            obj['NAMELIS2']['extqerf_qe'] = list(power)  # MW/m^3 -> W/cm^3（已在 lower_onetwo 中转换）
            obj['NAMELIS2']['extqerf_watts'] = [0.0]     # 不归一化

            # --- 外部离子热源（本计算中 LHW 主要加热电子，离子源置 0）---
            obj['NAMELIS2']['extqirf'] = [0.0]
            obj['NAMELIS2']['extqirf_id'] = ['lhw']
            obj['NAMELIS2']['extqirf_nj'] = [0]

            # --- ECRH 参数占位（ONETWO Namelist 要求存在这些字段）---
            obj['NAMELIS2']['rfmode'] = ['ech','ech','ech','ech']
            obj['NAMELIS2']['genraydat'] = ['','','','']
            obj['NAMELIS2']['rfon'] = [-40.0,-40.0,-40.0,-40.0]
            obj['NAMELIS2']['rftime'] = [20e3,20e3,20e3,20e3]
            obj['NAMELIS2']['rfpow'] = [4e5,4e5,4e5,4e5]
            obj['NAMELIS2']['freq'] = [140e9,140e9,140e9,140e9]

            obj['NAMELIS2']['xec'] = [300.0,300.0,300.0,300.0]
            obj['NAMELIS2']['zec'] = [-30,-30,-30,-30]
            obj['NAMELIS2']['thetec'] = [86.0,80.0,86.2,86.0]
            obj['NAMELIS2']['phaiec'] = [200.0,200.0,200.0,0.0]

            obj['NAMELIS2']['irfcur'] = [1.0,1.0,1.0,1.0]
            obj['NAMELIS2']['wrfo'] = [0.0,0.0,0.0,0.0]
            obj['NAMELIS2']['nray'] = [30.0,30.0,30.0,30.0]
            obj['NAMELIS2']['idamp'] = [2.0,2.0,2.0,2.0]
            obj['NAMELIS2']['hlwec'] = [1.2,1.2,2.0,1.2]
            obj['NAMELIS2']['ratwec'] = [1.0,1.0,1.0,1.0]

            # 保存完整的 inone 文件到 ONETWO 工作目录
            onetwo_save_path = os.path.join(onetwo_result_dir, 'inone')
            obj.write(onetwo_save_path)
            print('ecrh_written')

            # --- 从 MDSplus 读取 gfile 并保存为 ONETWO 所需的 g0_input ---
            conn = Connection('202.127.204.42')  # EAST MDSplus 服务器地址
            TREE = 'efit_east'
            conn.openTree(TREE, shot)
            gg = geqdsk.read_from_MDS(conn, time)
            g_times = conn.get(r'data(\GTIME)').data()
            timeid = np.argmin(abs(g_times - time))
            gg = geqdsk.read_from_MDS(conn, timeid)
            conn.closeTree(TREE, shot)

            filename = os.path.join(onetwo_result_dir, "g0_input")
            geqdsk.save(gg, filename)

            # --- 调用 ONETWO 进行输运计算 ---
            subprocess.run(["onetwo_129_201"], cwd=onetwo_result_dir, check=True)
            print(f'(onetwo)success at {time}s')

        except Exception as Error_onetwo:
            print(f'(onetwo)fail at {time}s: {str(Error_onetwo)}')


def main():
    """
    主程序入口：
      1. 获取炮号（shot）
      2. 从 MDSplus 读取该炮的所有 TS 时间片
      3. 使用多进程池并行处理每个时间片
      4. 统计总耗时
    """
    start_time = time.time()
    shot = int(input('Shot number: '))

    # 连接 EAST MDSplus，获取 TS_EAST 树下的所有时间片
    conn = Connection('202.127.204.42')
    conn.openTree('TS_EAST', shot)
    TS_times = conn.get(r'dim_of(\Te_coreTS)').data()
    conn.closeTree('TS_EAST', shot)

    print(f'Found {len(TS_times)} time points')

    # 创建总结果目录
    os.makedirs("results", exist_ok=True)

    # 构造多进程任务列表：(time, shot) 元组列表
    pool = multiprocessing.Pool()
    task_args = [(t, shot) for t in TS_times]

    # 并行映射 process_time_point 到所有时间片
    pool.map(process_time_point, task_args)
    pool.close()
    pool.join()

    # 计算并输出总耗时
    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nTotal execution time: {int(hours):0>2}h {int(minutes):0>2}m {seconds:05.2f}s")


if __name__ == '__main__':
    main()
