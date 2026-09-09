# -*- coding: utf-8 -*-
"""
onetwo_output_mat.py — 使用 .mat 文件中的 Te/ne 剖面作为 ONETWO 输入

与 onetwo_output.py 的区别：
  - Te、ne 剖面从 .mat 文件直接读取（跳过 MDSplus 读取和 mtanh/NN 拟合）
  - Ti 剖面：优先使用 paper/override/ 下的 NPZ 文件（mtanh 拟合），否则从 MDSplus 读取
  - gfile：优先使用本地 paper/{shot}_{time}_gfile，否则从 MDSplus 读取
  - LHW 计算、Namelist 组装、ONETWO 执行与原版完全相同

.mat 文件格式（MATLAB v7.3 HDF5）:
  Fitdata.rho : (1, N) 归一化半径网格
  Fitdata.Te  : (1, N) 电子温度 [eV]
  Fitdata.ne  : (1, N) 电子密度 [1e19 m^-3]
  Fitdata.time: 时间点 [s]

用法:
  python onetwo_output_mat.py
  （交互式输入炮号，自动搜索 fitdata_{shot}_*.mat 文件）

  或指定时间点:
  python onetwo_output_mat.py 156005 4.0179 6.0178
"""

import numpy as np
import os
import sys
import glob
import h5py

# 确保脚本所在目录在 Python 搜索路径中（服务器部署兼容）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess
import time as time_module
import traceback as _traceback

from scipy.interpolate import interp1d, PchipInterpolator

import geqdsk
from MDSplus.connection import Connection
from fitting_mtanh import fitting
from Namelist3 import Namelist
from lower_view_final_nn import lower_onetwo_nn
from readMDS_onetwo import readmds

# ============================================================
# 路径配置
# ============================================================
_ONETWO_EXE = os.path.join(SCRIPT_DIR, 'onetwo_129_201')

# ============================================================
# 工具函数（从 onetwo_output.py 复制）
# ============================================================

def format_time_dirname(t: float) -> str:
    """将浮点时间转换为目录命名格式，如 3.5s -> '003.50000s'"""
    time_str = "{:.5f}".format(t)
    if '.' not in time_str:
        integer_part = time_str
        decimal_part = "00000"
    else:
        integer_part, decimal_part = time_str.split('.')
    integer_part = integer_part.zfill(3)
    decimal_part = decimal_part.ljust(5, '0')[:5]
    return f"{integer_part}.{decimal_part}s"


def robust_interp(x, y, datatype, thresholds=None, offsets=None, num_points=201):
    """鲁棒插值到 201 点均匀网格（从 onetwo_output.py 复制）"""
    if thresholds is None:
        thresholds = {"ne": 0.1, "Te": 0.05, "Ti": 0.05}
    if offsets is None:
        offsets = {"ne": 0.05, "Te": 0.05, "Ti": 0.05}

    th = thresholds.get(datatype, 0.0)
    off = offsets.get(datatype, 0.0)

    x, y = np.array(x), np.array(y)

    mask = y >= th
    x_clean, y_clean = x[mask], y[mask]

    if len(y_clean) == 0:
        y_lifted = y + off
        f = interp1d(x, y_lifted, kind='linear', fill_value="extrapolate")
        x_fit = np.linspace(0, 1.0, num_points)
        y_fit = f(x_fit)
        return x_fit, y_fit

    f = interp1d(x_clean, y_clean, kind='linear', fill_value="extrapolate")
    x_fit = np.linspace(0, 1.0, num_points)
    y_fit = f(x_fit)

    if np.min(y_fit) < th:
        y_lifted = y + off
        f = interp1d(x, y_lifted, kind='linear', fill_value="extrapolate")
        x_fit = np.linspace(0, 1.0, num_points)
        y_fit = f(x_fit)

    return x_fit, y_fit


def enforce_monotone_pchip(x, y, num_points=201, decreasing=True):
    """PCHIP 强制单调插值（从 onetwo_output.py 复制）"""
    x = np.asarray(x)
    y = np.asarray(y)
    order = np.argsort(x)
    x, y = x[order], y[order]

    if decreasing:
        y_proc = -y
    else:
        y_proc = y

    pchip = PchipInterpolator(x, y_proc)
    x_new = np.linspace(0, 1, num_points)
    y_new = pchip(x_new)

    if decreasing:
        y_new = -y_new

    return x_new, y_new


def _log_error(shot, time, stage, error_msg, exc_info=False):
    """记录错误日志"""
    os.makedirs(os.path.join("results", str(shot)), exist_ok=True)
    log_path = os.path.join("results", str(shot), "pipeline_errors.log")
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{ts}] time={time:.6f} | {stage} | {error_msg}"
    if exc_info:
        entry += f"\n{_traceback.format_exc()}"
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(entry + '\n')
    except Exception:
        pass
    print(entry)


# ============================================================
# .mat 文件读取
# ============================================================

def load_mat_profile(mat_path):
    """从 MATLAB v7.3 .mat 文件读取 Fitdata 结构体。

    返回:
        time_val : float, 时间点 [s]
        rho      : (N,) ndarray, 归一化半径
        Te       : (N,) ndarray, 电子温度 [eV]
        ne       : (N,) ndarray, 电子密度 [1e19 m^-3]
        pe       : (N,) ndarray, 电子压力
    """
    with h5py.File(mat_path, 'r') as f:
        fd = f['Fitdata']
        time_val = float(np.asarray(fd['time']).flatten()[0])
        rho = np.asarray(fd['rho']).flatten().astype(np.float64)
        Te = np.asarray(fd['Te']).flatten().astype(np.float64)
        ne = np.asarray(fd['ne']).flatten().astype(np.float64)
        pe = np.asarray(fd['pe']).flatten().astype(np.float64)

    return time_val, rho, Te, ne, pe


def discover_mat_files(shot, search_dir=None):
    """自动搜索 fitdata_{shot}_*.mat 文件。

    返回:
        {time_value: mat_file_path} 字典
    """
    if search_dir is None:
        search_dir = SCRIPT_DIR

    pattern = os.path.join(search_dir, f'fitdata_{shot}_*.mat')
    files = glob.glob(pattern)

    result = {}
    for f in files:
        time_val, _, _, _, _ = load_mat_profile(f)
        result[round(time_val, 6)] = f  # 四舍五入到微秒避免浮点误差

    return result


# ============================================================
# Ti 数据加载（override NPZ 或 MDSplus）
# ============================================================

def load_ti_override(shot, time_val):
    """尝试从 paper/override/ 加载 Ti 散点数据。

    返回:
        (x_ti, y_ti) 或 (None, None)
        x_ti: rho 数组
        y_ti: Ti [keV] 数组
    """
    override_dir = os.path.join(SCRIPT_DIR, 'paper', 'override')
    if not os.path.isdir(override_dir):
        return None, None

    # 生成可能的文件名: 156005_004.01790s_Ti.npz
    time_str = format_time_dirname(time_val)
    # 去掉末尾的 's' 再匹配（原始文件名没有 's'）
    time_str_no_s = time_str.replace('s', '')
    pattern = os.path.join(override_dir, f'{shot}_{time_str_no_s}_Ti.npz')
    if os.path.exists(pattern):
        data = np.load(pattern)
        x = np.asarray(data['rho'], dtype=np.float64).flatten()
        y = np.asarray(data['y'], dtype=np.float64).flatten()
        return x, y

    # 尝试更灵活的匹配：搜索同 shot 下最近时间的文件
    shot_pattern = os.path.join(override_dir, f'{shot}_*_Ti.npz')
    candidates = glob.glob(shot_pattern)
    best_match = None
    best_diff = float('inf')
    for c in candidates:
        basename = os.path.basename(c)
        try:
            parts = basename.replace(f'{shot}_', '').replace('_Ti.npz', '')
            t_str = parts.replace('s', '')
            t_val = float(t_str)
            diff = abs(t_val - time_val)
            if diff < best_diff and diff < 0.01:  # 10ms 容差
                best_diff = diff
                best_match = c
        except ValueError:
            continue

    if best_match is not None:
        data = np.load(best_match)
        x = np.asarray(data['rho'], dtype=np.float64).flatten()
        y = np.asarray(data['y'], dtype=np.float64).flatten()
        print(f'  Ti override: {best_match} (Δt={best_diff*1000:.1f}ms)')
        return x, y

    return None, None


# ============================================================
# gfile 加载（本地优先，MDSplus 作为 fallback）
# ============================================================

def load_gfile(shot, time_val):
    """加载 gfile，优先本地文件，否则从 MDSplus 读取。

    返回:
        gg : gfile 字典（geqdsk 格式）
    """
    # 策略1：本地 gfile — 先搜 fit/ 根目录，再搜 paper/ 子目录
    time_rounded = round(time_val, 3)
    local_candidates = [
        # 服务器上 gfile 直接放在 fit/ 根目录，命名格式 {shot}_{time}_gfile
        os.path.join(SCRIPT_DIR, f'{shot}_{time_rounded}_gfile'),
        os.path.join(SCRIPT_DIR, f'{shot}_{time_rounded:.3f}_gfile'),
        # paper/ 子目录（本地 repo 结构）
        os.path.join(SCRIPT_DIR, 'paper', f'{shot}_{time_rounded}_gfile'),
        os.path.join(SCRIPT_DIR, 'paper', f'{shot}_{time_rounded:.3f}_gfile'),
    ]
    for candidate in local_candidates:
        if os.path.exists(candidate):
            print(f'  gfile from local: {candidate}')
            return geqdsk.load(candidate)

    # 策略2：从 MDSplus 读取
    print(f'  gfile: reading from MDSplus efit_east...')
    conn = Connection('202.127.204.42')
    TREE = 'efit_east'
    try:
        conn.openTree(TREE, shot)
        g_times = np.asarray(conn.get(r'data(\GTIME)').data(), dtype=np.float64).flatten()
        if len(g_times) == 0:
            raise ValueError(f'efit_east tree has zero gfile time slices for shot {shot}')
        timeid = np.argmin(abs(g_times - time_val))
        gg = geqdsk.read_from_MDS(conn, timeid)
        conn.closeTree(TREE, shot)
        return gg
    except Exception:
        try:
            conn.closeTree(TREE, shot)
        except Exception:
            pass
        raise RuntimeError(
            f'Shot {shot} @ {time_val}s: gfile unavailable.'
        )


# ============================================================
# 主处理函数：单个时间点
# ============================================================

def process_time_point_mat(shot, time_val, mat_file):
    """处理单个时间点：.mat 数据 → ONETWO。

    参数:
        shot     : 炮号
        time_val : 时间点 [s]
        mat_file : .mat 文件路径
    """
    output_base = "results"
    time_dir = format_time_dirname(time_val)
    output_dir = os.path.join(output_base, str(shot), time_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ================================================================
    # 第一步：从 .mat 文件读取 Te 和 ne 剖面
    # ================================================================
    print(f'\n{"="*60}')
    print(f'Processing shot={shot}, time={time_val:.6f}s')
    print(f'  .mat file: {mat_file}')

    _, rho, Te_eV, ne_raw, pe = load_mat_profile(mat_file)
    print(f'  .mat rho: [{rho.min():.4f}, {rho.max():.4f}], {len(rho)} pts')
    print(f'  .mat Te:  [{Te_eV.min():.1f}, {Te_eV.max():.1f}] eV')
    print(f'  .mat ne:  [{ne_raw.min():.2e}, {ne_raw.max():.2e}] (1e19 m^-3)')

    # ---- 清理 ne：截断负值为 0 ----
    ne_clean = np.clip(ne_raw, 0.0, None)

    # ---- 截断 rho 到 [0, 1.05] 范围（.mat 可能超出 1） ----
    rho_clip_mask = (rho >= 0) & (rho <= 1.05)
    rho_clip = rho[rho_clip_mask]
    Te_clip = Te_eV[rho_clip_mask]
    ne_clip = ne_clean[rho_clip_mask]

    # ---- 关键：按 rho 升序排列（.mat 数据不是单调递增的，会导致 plot 线条锯齿） ----
    sort_idx = np.argsort(rho_clip)
    rho_clip = rho_clip[sort_idx]
    Te_clip = Te_clip[sort_idx]
    ne_clip = ne_clip[sort_idx]

    # ---- 去除 rho 重复值（两个 pass 在芯部附近有近似重复的点） ----
    _, unique_idx = np.unique(rho_clip, return_index=True)
    rho_clip = rho_clip[unique_idx]
    Te_clip = Te_clip[unique_idx]
    ne_clip = ne_clip[unique_idx]

    # ---- 插值到 ONETWO 标准 201 点网格 rho ∈ [0, 1] ----
    x_201 = np.linspace(0, 1, 201)

    # Te 插值（单位保持 eV）
    f_te = interp1d(rho_clip, Te_clip, kind='linear', fill_value='extrapolate')
    Te_201_eV = f_te(x_201)
    Te_201_eV = np.clip(Te_201_eV, 0.001, None)  # 确保非负
    Te_201_keV = Te_201_eV / 1000.0               # 同时准备 keV 版本供 LHW 计算

    # ne 插值（单位保持 1e19 m^-3）
    f_ne = interp1d(rho_clip, ne_clip, kind='linear', fill_value='extrapolate')
    ne_201 = f_ne(x_201)
    ne_201 = np.clip(ne_201, 0.0, None)           # 确保非负

    print(f'  Interpolated Te: [{Te_201_eV.min():.1f}, {Te_201_eV.max():.1f}] eV')
    print(f'  Interpolated ne: [{ne_201.min():.2e}, {ne_201.max():.2e}] (1e19 m^-3)')

    # ---- 绘制 Te 剖面 ----
    plt.figure(figsize=(8, 5))
    plt.plot(rho_clip, Te_clip / 1000., 'g.-', markersize=3, label='.mat Te (raw)')
    plt.plot(x_201, Te_201_keV, 'b-', linewidth=2, label='Interpolated (201 pts)')
    plt.xlabel('rho')
    plt.ylabel('Te (keV)')
    plt.title(f'Shot {shot} @ {time_val:.5f}s — Te from .mat')
    plt.xlim([0, 1])
    plt.legend()
    plt.grid(alpha=0.3)
    save_path = os.path.join(output_dir, f'Te_shot{shot}_time{time_dir}_mat.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Te plot saved: {save_path}')

    # ---- 绘制 ne 剖面 ----
    plt.figure(figsize=(8, 5))
    plt.plot(rho_clip, ne_clip, 'g.-', markersize=3, label='.mat ne (raw)')
    plt.plot(x_201, ne_201, 'b-', linewidth=2, label='Interpolated (201 pts)')
    plt.xlabel('rho')
    plt.ylabel('ne (1e19 m^-3)')
    plt.title(f'Shot {shot} @ {time_val:.5f}s — ne from .mat')
    plt.xlim([0, 1])
    plt.legend()
    plt.grid(alpha=0.3)
    save_path = os.path.join(output_dir, f'ne_shot{shot}_time{time_dir}_mat.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ne plot saved: {save_path}')

    # ================================================================
    # 第二步：获取 Ti 剖面（override NPZ → mtanh 拟合，或 MDSplus）
    # ================================================================
    TE_STATUS = 0
    NE_STATUS = 0
    TI_STATUS = 0

    te_ped_x = x_201[x_201 >= 0.88]
    te_ped_y = Te_201_keV[x_201 >= 0.88]  # keV

    obj = Namelist()
    obj.read('inone_template')

    # 写入 Te（与原版 onetwo_output.py 一致：keV；inone_template 按 keV 解析）
    obj['NAMELIS1']['RTEIN'] = list(x_201)
    obj['NAMELIS1']['TEIN'] = list(Te_201_keV)  # keV

    # 写入 ne（ONETWO 需要 cm^-3）
    obj['NAMELIS1']['RENEIN'] = list(x_201)
    obj['NAMELIS1']['ENEIN'] = list(ne_201 * 1e13)  # 1e19 m^-3 → cm^-3

    # ---- Ti 处理 ----
    x_ti_override, y_ti_override = load_ti_override(shot, time_val)

    if x_ti_override is not None and y_ti_override is not None:
        # 使用 mtanh 拟合 Ti 散点数据（与服务器流程一致）
        try:
            print(f'  Ti: fitting {len(x_ti_override)} override points with mtanh...')

            plt.figure(figsize=(8, 5))
            plt.scatter(x_ti_override, y_ti_override, marker='.', c='#2E86AB', s=30,
                        label='XCS Ti override')

            # mtanh 拟合 + 清洗
            x_fit, y_fit, x_del, y_del, q = fitting(
                x_ti_override, y_ti_override, 'Ti',
                te_ped_x=te_ped_x, te_ped_y=te_ped_y
            )

            if len(x_del) > 0:
                plt.scatter(x_del, y_del, marker='x', s=60, c='r',
                            label=f'Removed ({len(x_del)})')

            # 插值 + 单调化
            x_ti_201, y_ti_201 = robust_interp(x_fit, y_fit, 'Ti')
            x_ti_201, y_ti_201 = enforce_monotone_pchip(x_ti_201, y_ti_201)

            plt.plot(x_ti_201, y_ti_201, 'r-', linewidth=2, label='mtanh fit')
            plt.xlabel('rho')
            plt.ylabel('Ti (keV)')
            plt.title(f'Shot {shot} @ {time_val:.5f}s — Ti mtanh (override)')
            plt.xlim([0, 1])
            plt.legend()
            plt.grid(alpha=0.3)
            save_path = os.path.join(output_dir, f'Ti_shot{shot}_time{time_dir}_mat.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f'  Ti plot saved: {save_path}')

            # 写入 Namelist（Ti 直接用 keV）
            obj['NAMELIS1']['RTIIN'] = list(x_ti_201)
            obj['NAMELIS1']['TIIN'] = list(y_ti_201)
            print(f'  Ti fitted: core={y_ti_201[0]:.4f}, edge={y_ti_201[-1]:.4f} keV')

        except Exception as e:
            _log_error(shot, time_val, 'Ti_mtanh_override', str(e), exc_info=True)
            TI_STATUS = 1
    else:
        # 尝试从 MDSplus 读取 Ti
        print(f'  Ti: no override NPZ found, trying MDSplus...')
        try:
            data, status, real_time = readmds(shot, time_val)
            if status['TXCS_status'] == 1 and te_ped_x is not None and te_ped_y is not None:
                x, y, datatype = data['Ti']['TXCS']['Rho'], data['Ti']['TXCS']['data'], data['Ti']['type']

                plt.figure(figsize=(8, 5))
                plt.scatter(x, y, marker='.', c='g')

                x_fit, y_fit, x_del, y_del, q = fitting(
                    x, y, datatype,
                    te_ped_x=te_ped_x, te_ped_y=te_ped_y
                )
                x_ti, y_ti = robust_interp(x_fit, y_fit, datatype)
                x_ti, y_ti = enforce_monotone_pchip(x_ti, y_ti)

                plt.scatter(x_del, y_del, marker='x', s=60, c='r')
                plt.plot(x_ti, y_ti, 'g--', linewidth=2, label='mtanh fit')
                plt.xlabel('rho')
                plt.ylabel('Ti(TXCS) (keV)')
                plt.title(f'Shot {shot} @ {time_val:.5f}s — Ti MDSplus')
                plt.xlim([0, 1])
                plt.legend()
                plt.grid(alpha=0.3)
                save_path = os.path.join(output_dir, f'Ti_shot{shot}_time{time_dir}_mat.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f'  Ti from MDSplus: saved to {save_path}')

                obj['NAMELIS1']['RTIIN'] = list(x_ti)
                obj['NAMELIS1']['TIIN'] = list(y_ti)
                print(f'  Ti MDSplus: core={y_ti[0]:.4f}, edge={y_ti[-1]:.4f} keV')
            else:
                print(f'  Ti: TXCS not available on MDSplus, skipping Ti')
                TI_STATUS = 1
        except Exception as e:
            _log_error(shot, time_val, 'Ti_MDSplus', str(e))
            TI_STATUS = 1

    # ================================================================
    # 第三步：LHW 计算 + Namelist 组装 + ONETWO 执行
    # ================================================================
    if TE_STATUS == 0 and NE_STATUS == 0:
        try:
            onetwo_result_dir = os.path.join(output_base, str(shot), time_dir, 'onetwo_mat')
            os.makedirs(onetwo_result_dir, exist_ok=True)

            # ---- 调用 lower_onetwo_nn 计算 LHW ----
            # 传入 .mat 导出的 Te/ne 剖面
            print(f'  Running LHW calculation...')
            rho_lhw, power_lhw, current_lhw = lower_onetwo_nn(
                shot, time_val, onetwo_result_dir,
                te_profile_nn=Te_201_keV,      # keV
                ne_profile_nn=ne_201,           # 1e19 m^-3
                te_rho_nn=x_201,
                ne_rho_nn=x_201,
            )
            print(f'  LHW done: {len(rho_lhw)} rho points')

            # ---- 组装 ONETWO Namelist ----
            # NAMELIS2：外部电流驱动
            obj['NAMELIS2']['extcurrf'] = [1.0]
            obj['NAMELIS2']['extcurrf_id'] = ['lhw']
            obj['NAMELIS2']['extcurrf_nj'] = [len(rho_lhw)]
            obj['NAMELIS2']['extcurrf_rho'] = list(rho_lhw)
            obj['NAMELIS2']['extcurrf_curr'] = list(current_lhw * 1e6 / 1e4)  # MA/m^2 -> A/cm^2
            obj['NAMELIS2']['extcurrf_amps'] = [0.0]

            # NAMELIS2：外部功率沉积
            obj['NAMELIS2']['extqerf'] = [1.0]
            obj['NAMELIS2']['extqerf_id'] = ['lhw']
            obj['NAMELIS2']['extqerf_nj'] = [len(rho_lhw)]
            obj['NAMELIS2']['extqerf_rho'] = list(rho_lhw)
            obj['NAMELIS2']['extqerf_qe'] = list(power_lhw)
            obj['NAMELIS2']['extqerf_watts'] = [0.0]

            # NAMELIS2：外部离子加热源
            obj['NAMELIS2']['extqirf'] = [0.0]
            obj['NAMELIS2']['extqirf_id'] = ['lhw']
            obj['NAMELIS2']['extqirf_nj'] = [0]

            # NAMELIS2：ECRH 占位
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

            # ---- 写入 inone ----
            inone_path = os.path.join(onetwo_result_dir, 'inone')
            obj.write(inone_path)
            print(f'  inone written: {inone_path}')

            # ---- 加载并保存 gfile ----
            gg = load_gfile(shot, time_val)
            gfile_path = os.path.join(onetwo_result_dir, 'g0_input')
            geqdsk.save(gg, gfile_path)
            print(f'  gfile saved: {gfile_path}')

            # ---- 执行 ONETWO ----
            _env = os.environ.copy()
            _env['LD_LIBRARY_PATH'] = (
                '/usr/local/mdsplus/lib:'
                '/home/fusion/imd/onetwo5/lib:'
                '/home/fusion/imd/auto12/netcdf4.1.3/pgi-1410/lib:'
                '/home/fusion/imd/auto12/cfetr_bin/lib:'
                '/home/fusion/imd/auto12/hdf5/pgi-1410/lib:'
                + _env.get('LD_LIBRARY_PATH', '')
            )
            print(f'  Running ONETWO in {onetwo_result_dir}...')
            subprocess.run([_ONETWO_EXE], cwd=onetwo_result_dir, check=True, env=_env)
            print(f'  (ONETWO_mat) SUCCESS at time={time_val}s')

        except Exception as e:
            _log_error(shot, time_val, 'mat_ONETWO', str(e), exc_info=True)


# ============================================================
# 主入口
# ============================================================

def main():
    start = time_module.time()

    # 命令行参数支持：python onetwo_output_mat.py [shot] [time1 time2 ...]
    if len(sys.argv) >= 2:
        shot = int(sys.argv[1])
    else:
        shot = int(input('Shot number: '))

    # 自动发现 .mat 文件
    mat_files = discover_mat_files(shot)

    if not mat_files:
        print(f'ERROR: No fitdata_{shot}_*.mat files found in current directory.')
        print(f'Put the .mat files in: {SCRIPT_DIR}')
        sys.exit(1)

    print(f'Found {len(mat_files)} .mat file(s) for shot {shot}:')
    for t, f in sorted(mat_files.items()):
        print(f'  t={t:.6f}s -> {os.path.basename(f)}')

    # 如果命令行指定了时间点，只处理这些
    if len(sys.argv) >= 3:
        requested_times = [float(x) for x in sys.argv[2:]]
        # 匹配最近的时间点
        tasks = []
        for rt in requested_times:
            best_t = min(mat_files.keys(), key=lambda t: abs(t - rt))
            if abs(best_t - rt) < 0.1:  # 100ms 容差
                tasks.append((best_t, mat_files[best_t]))
            else:
                print(f'WARNING: No .mat file found for t={rt}s (closest: {best_t:.6f}s)')
    else:
        tasks = sorted(mat_files.items())

    print(f'\nProcessing {len(tasks)} time point(s)...')

    for time_val, mat_file in tasks:
        process_time_point_mat(shot, time_val, mat_file)

    elapsed = time_module.time() - start
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nTotal execution time: {int(hours):0>2}h {int(minutes):0>2}m {seconds:05.2f}s")


if __name__ == '__main__':
    main()
