# -*- coding: utf-8 -*-
import geqdsk
import numpy as np
import os
from MDSplus.connection import Connection
from fitting_mtanh import fitting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing
import subprocess
from Namelist3 import Namelist
import shutil
import glob
import time
from lower_view_final import lower_onetwo
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator

# =============================================================================
# 工具函数定义
# =============================================================================

def enforce_monotone_pchip(x, y, num_points=201, decreasing=True):
    """用 PCHIP 保证单调剖面"""
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

def robust_interp(x, y, datatype, thresholds=None, offsets=None, num_points=201):
    """自动处理过小数据并进行线性插值拟合"""
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
        return x_fit, f(x_fit)
    f = interp1d(x_clean, y_clean, kind='linear', fill_value="extrapolate")
    x_fit = np.linspace(0, 1.0, num_points)
    y_fit = f(x_fit)
    if np.min(y_fit) < th:
        y_lifted = y + off
        f = interp1d(x, y_lifted, kind='linear', fill_value="extrapolate")
        x_fit = np.linspace(0, 1.0, num_points)
        y_fit = f(x_fit)
    return x_fit, y_fit

def format_time_dirname(t: float) -> str:
    """时间格式化"""
    time_str = "{:.5f}".format(t)
    integer_part, decimal_part = time_str.split('.') if '.' in time_str else (time_str, "00000")
    return f"{integer_part.zfill(3)}.{decimal_part.ljust(5, '0')[:5]}s"

from readMDS_onetwo import readmds

# =============================================================================
# 核心处理函数 (包含计时逻辑)
# =============================================================================

def process_time_point(args):
    i, shot = args
    time_val = float(i)
    output_base = "results"
    time_dir = format_time_dirname(time_val)
    output_dir = os.path.join(output_base, f"{shot}", time_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 初始化计时统计字典
    stats = {'fit_io': 0.0, 'lhw': 0.0, 'onetwo': 0.0, 'total': 0.0}
    t_start_total = time.time()

    # --- 阶段 1: Data cleaning + fitting + I/O ---
    t0 = time.time()
    data, status, real_time = readmds(shot, time_val)
    obj = Namelist()
    obj.read('inone_template')
    TE_STATUS, NE_STATUS = 0, 0
    te_ped_x, te_ped_y = None, None

    # TS Te
    if status['TS_status'] == 1:
        try:
            x, y, dtype = data['Te']['TS']['Rho'], data['Te']['TS']['data'], data['Te']['TS']['type']
            x_te, y_te, x_del, y_del, _ = fitting(x, y, dtype)
            x_te, y_te = robust_interp(x_te, y_te, dtype)
            mask = (x_te >= 0.88)
            te_ped_x, te_ped_y = x_te[mask], y_te[mask]
            obj['NAMELIS1']['RTEIN'], obj['NAMELIS1']['TEIN'] = list(x_te), list(y_te)
            
            plt.figure(figsize=(8, 5))
            plt.scatter(x, y/1000., marker='.', c='g')
            plt.plot(x_te, y_te, 'g--')
            plt.savefig(os.path.join(output_dir, f'Te_{time_dir}.png'))
            plt.close()
        except: TE_STATUS = 1
    else: TE_STATUS = 1

    # Refl ne
    if status['Refl_status'] == 1:
        try:
            x, y, dtype = data['ne']['Refl']['Rho'], data['ne']['Refl']['data'], data['ne']['Refl']['type']
            x_f, y_f, _, _, _ = fitting(x, y, dtype)
            x_fit, y_interp = robust_interp(x_f, y_f, dtype)
            obj['NAMELIS1']['RENEIN'], obj['NAMELIS1']['ENEIN'] = list(x_fit), list(y_interp * 1e13)
        except: NE_STATUS = 1
    else: NE_STATUS = 1

    # TXCS Ti
    if status['TXCS_status'] == 1 and te_ped_x is not None:
        try:
            x, y, dtype = data['Ti']['TXCS']['Rho'], data['Ti']['TXCS']['data'], data['Ti']['type']
            x_ti, y_ti, _, _, _ = fitting(x, y, dtype, te_ped_x=te_ped_x, te_ped_y=te_ped_y)
            x_ti, y_ti = robust_interp(x_ti, y_ti, dtype)
            x_ti, y_ti = enforce_monotone_pchip(x_ti, y_ti)
            obj['NAMELIS1']['RTIIN'], obj['NAMELIS1']['TIIN'] = list(x_ti), list(y_ti)
        except: pass

    stats['fit_io'] = time.time() - t0

    # --- 阶段 2: LHW ---
    if (TE_STATUS == 0 and NE_STATUS == 0):
        try:
            onetwo_result_dir = os.path.join(output_dir, 'onetwo')
            os.makedirs(onetwo_result_dir, exist_ok=True)
            
            t1 = time.time()
            rho, power, current = lower_onetwo(shot, time_val, onetwo_result_dir)
            stats['lhw'] = time.time() - t1

            # --- 阶段 3: ECRH + ONETWO ---
            t2 = time.time()
            # 填充 ONETWO LHW 参数
            obj['NAMELIS2']['extcurrf'], obj['NAMELIS2']['extcurrf_rho'] = [1.0], list(rho)
            obj['NAMELIS2']['extcurrf_curr'] = list(current * 1e6 / 1e4)
            obj['NAMELIS2']['extqerf'], obj['NAMELIS2']['extqerf_rho'] = [1.0], list(rho)
            obj['NAMELIS2']['extqerf_qe'] = list(power)
            
            # 填充模板中的 ECRH 参数 (保持原逻辑)
            obj['NAMELIS2']['rfon'] = [-40.0]*4
            obj['NAMELIS2']['rfpow'] = [4e5]*4
            
            obj.write(os.path.join(onetwo_result_dir, 'inone'))
            
            # 平衡文件处理
            conn = Connection('202.127.204.42')
            try:
                conn.openTree('efit_east', shot)
                g_times = np.asarray(conn.get(r'data(\GTIME)').data(), dtype=np.float64).flatten()
                if len(g_times) == 0:
                    raise ValueError(f'efit_east tree has zero gfile time slices for shot {shot}')
                timeid = np.argmin(abs(g_times - time_val))
                gg = geqdsk.read_from_MDS(conn, timeid)
                conn.closeTree('efit_east', shot)
            except Exception as gfile_err:
                try:
                    conn.closeTree('efit_east', shot)
                except Exception:
                    pass
                raise RuntimeError(
                    f'Shot {shot} @ {time_val}s: gfile unavailable from MDSplus efit_east tree.'
                ) from gfile_err
            geqdsk.save(gg, os.path.join(onetwo_result_dir, "g0_input"))
            
            # 运行 ONETWO
            _env = os.environ.copy()
            _env['LD_LIBRARY_PATH'] = '/usr/local/mdsplus/lib:/home/fusion/imd/onetwo5/lib:/home/fusion/imd/auto12/netcdf4.1.3/pgi-1410/lib:/home/fusion/imd/auto12/cfetr_bin/lib:/home/fusion/imd/auto12/hdf5/pgi-1410/lib:' + _env.get('LD_LIBRARY_PATH', '')
            subprocess.run(["onetwo_129_201"], cwd=onetwo_result_dir, check=True, capture_output=True, env=_env)
            stats['onetwo'] = time.time() - t2
        except Exception as e:
            print(f'Fail at {time_val}s: {e}')

    stats['total'] = time.time() - t_start_total
    
    # 将计时结果追加写入文本文件
    with open(f"time_stats_{shot}.txt", "a") as f_log:
        f_log.write(f"{time_val:.5f}, {stats['fit_io']:.4f}, {stats['lhw']:.4f}, {stats['onetwo']:.4f}, {stats['total']:.4f}\n")
    
    return stats

# =============================================================================
# 分析汇总函数
# =============================================================================

def analyze_performance(filename):
    """读取计时文件并输出 Table II 格式报告"""
    try:
        data = np.loadtxt(filename, delimiter=',', comments='#')
        avg_fit = np.mean(data[:, 1])
        avg_lhw = np.mean(data[:, 2])
        avg_onetwo = np.mean(data[:, 3])
        avg_total = np.mean(data[:, 4])
        
        print("\n" + "="*60)
        print("Table II. Computational performance and time analysis")
        print("-" * 60)
        print(f"{'Module':<35} | {'Avg. Time (s)':<12} | {'Ratio (%)':<10}")
        print("-" * 60)
        print(f"{'ECRH + ONETWO':<35} | {avg_onetwo:<12.3f} | {avg_onetwo/avg_total*100:<10.1f}")
        print(f"{'LHW (METIS Fast Model)':<35} | {avg_lhw:<12.3f} | {avg_lhw/avg_total*100:<10.1f}")
        print(f"{'Data cleaning + fitting + I/O':<35} | {avg_fit:<12.3f} | {avg_fit/avg_total*100:<10.1f}")
        print("-" * 60)
        print(f"{'Total (Average per time slice)':<35} | {avg_total:<12.3f} | 100.0")
        print("="*60)
    except Exception as e:
        print(f"Performance analysis skipped: {e}")

# =============================================================================
# 主入口
# =============================================================================

if __name__ == '__main__':
    start_wall_clock = time.time()
    shot = int(input('Shot number: '))
    
    # 清理旧的统计文件并写入表头
    stat_file = f"time_stats_{shot}.txt"
    with open(stat_file, "w") as f:
        f.write("# time, data_fit_io, lhw, onetwo, total\n")

    conn = Connection('202.127.204.42')
    conn.openTree('TS_EAST', shot)
    TS_times = conn.get(r'dim_of(\Te_coreTS)').data()
    conn.closeTree('TS_EAST', shot)
    
    # 限制处理点数用于快速测试 (可选)
    # TS_times = TS_times[:5] 
    print(f'Processing {len(TS_times)} time points...')

    pool = multiprocessing.Pool()
    task_args = [(t, shot) for t in TS_times]
    pool.map(process_time_point, task_args)
    pool.close()
    pool.join()
    
    # 执行性能分析报告
    analyze_performance(stat_file)

    total_wall = time.time() - start_wall_clock
    print(f"\n[Parallel Summary] Processed {len(TS_times)} points in {total_wall:.2f}s (Wall Clock)")