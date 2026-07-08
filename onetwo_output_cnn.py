# -*- coding: utf-8 -*-
"""
onetwo_output_cnn.py — 基于 CNN-1D Baseline 的 ONETWO 输运计算流程

与 onetwo_output_nn.py（ProfileNet）和 onetwo_output_lstm.py（LSTM）的唯一区别：
用 CNN-1D ResNet 替代 ProfileNet/BiLSTM 做剖面拟合。
其余逻辑（MDSplus 数据读取、LHW 低杂波计算、gfile 平衡文件、ONETWO-Namelist 组装、
ONETWO 执行）完全一致。

输出独立存放在 onetwo_cnn/ 子目录下，不会与 ProfileNet / LSTM / 原始版本相互覆盖。

CNN-1D 特点：
  - 散点线性插值到 201 点固定网格 → 双通道输入（值+掩码）
  - 1D ResNet（残差卷积块）直接端到端输出平滑剖面
  - 天然保持局部光滑性，无需 LSTM 的序列建模或 SetEncoder 的排列不变

使用方法:
    python onetwo_output_cnn.py
    （交互式输入炮号）
"""

import geqdsk
import numpy as np
import os
from MDSplus.connection import Connection
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing
import subprocess
from Namelist3 import Namelist
import time
from lower_view_final_nn import lower_onetwo_nn

from profile_nn.infer_cnn import nn_fit_te_cnn, nn_fit_ne_cnn, nn_fit_ti_cnn, set_model_dir_cnn

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'profile_nn_models')
set_model_dir_cnn(MODEL_DIR)


def format_time_dirname(t: float) -> str:
    time_str = "{:.5f}".format(t)
    if '.' not in time_str:
        integer_part = time_str
        decimal_part = "00000"
    else:
        integer_part, decimal_part = time_str.split('.')
    integer_part = integer_part.zfill(3)
    decimal_part = decimal_part.ljust(5, '0')[:5]
    return f"{integer_part}.{decimal_part}s"


from readMDS_onetwo import readmds
import traceback as _traceback

# ONETWO 可执行文件路径（相对于脚本所在目录解析）
_ONETWO_EXE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'onetwo_129_201')

def _log_error(shot, time, stage, error_msg, exc_info=False):
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


def process_time_point(args):
    i, shot = args
    time = float(i)

    output_base = "results"
    time_dir = format_time_dirname(time)
    output_dir = os.path.join(output_base, f"{shot}", time_dir)
    os.makedirs(output_dir, exist_ok=True)

    data, status, real_time = readmds(shot, time)

    obj = Namelist()
    obj.read('inone_template')

    TE_STATUS = 0
    NE_STATUS = 0

    te_ped_x = None
    te_ped_y = None

    # =====================================================================
    # Te: TS → CNN-1D
    # =====================================================================
    if status['TS_status'] == 1:
        try:
            x, y = data['Te']['TS']['Rho'], data['Te']['TS']['data']
            x_te_201, y_te_201 = nn_fit_te_cnn(x, y)

            mask = (x_te_201 >= 0.88)
            te_ped_x = x_te_201[mask]
            te_ped_y = y_te_201[mask]

            plt.figure(figsize=(8, 5))
            plt.scatter(x, y / 1000., marker='.', c='g', label='TS raw')
            plt.plot(x_te_201, y_te_201, 'c-', linewidth=2, label='CNN-1D fitted')
            plt.xlabel(r'$\rho$')
            plt.ylabel('Te(TS) (keV)')
            plt.title(f'Shot {shot} @ {real_time:.3f}s [CNN-1D]')
            plt.xlim([0, 1])
            plt.legend()
            save_path = os.path.join(output_dir, f'Te_shot{shot}_time{time_dir}_cnn.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f'(CNN_Te) success at {real_time:.3f}s -> {save_path}')

            obj['NAMELIS1']['RTEIN'] = list(x_te_201)
            obj['NAMELIS1']['TEIN'] = list(y_te_201 * 1e3)
        except Exception as e:
            _log_error(shot, time, 'CNN_Te', str(e))
            TE_STATUS = 1
    else:
        TE_STATUS = 1

    # =====================================================================
    # ne: Refl → CNN-1D
    # =====================================================================
    if status['Refl_status'] == 1:
        try:
            x, y = data['ne']['Refl']['Rho'], data['ne']['Refl']['data']
            x_ne_201, y_ne_201 = nn_fit_ne_cnn(x, y)

            plt.figure(figsize=(8, 5))
            plt.scatter(x, y, marker='.', c='g', label='Refl raw')
            plt.plot(x_ne_201, y_ne_201, 'c-', linewidth=2, label='CNN-1D fitted')
            plt.xlabel(r'$\rho$')
            plt.ylabel('ne(Refl) (10$^{19}$ m$^{-3}$)')
            plt.title(f'Shot {shot} @ {real_time:.3f}s [CNN-1D]')
            plt.xlim([0, 1])
            plt.legend()
            save_path = os.path.join(output_dir, f'ne_shot{shot}_time{time_dir}_cnn.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f'(CNN_ne) success at {real_time:.3f}s -> {save_path}')

            obj['NAMELIS1']['RENEIN'] = list(x_ne_201)
            obj['NAMELIS1']['ENEIN'] = list(y_ne_201 * 1e13)
        except Exception as e:
            _log_error(shot, time, 'CNN_ne', str(e))
            NE_STATUS = 1
    else:
        NE_STATUS = 1

    # =====================================================================
    # Ti: TXCS + Te pedestal → CNN-1D (dual encoder)
    # =====================================================================
    if status['TXCS_status'] == 1 and te_ped_x is not None and te_ped_y is not None:
        try:
            x, y = data['Ti']['TXCS']['Rho'], data['Ti']['TXCS']['data']
            x_201, y_201 = nn_fit_ti_cnn(x, y, te_ped_x, te_ped_y)

            plt.figure(figsize=(8, 5))
            plt.scatter(x, y, marker='.', c='g', label='TXCS raw')
            plt.plot(x_201, y_201, 'c-', linewidth=2, label='CNN-1D fitted')
            plt.xlabel(r'$\rho$')
            plt.ylabel('Ti(TXCS) (keV)')
            plt.title(f'Shot {shot} @ {real_time:.3f}s [CNN-1D]')
            plt.xlim([0, 1])
            plt.legend()
            save_path = os.path.join(output_dir, f'Ti_shot{shot}_time{time_dir}_cnn.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f'(CNN_Ti) success at {real_time:.3f}s -> {save_path}')

            obj['NAMELIS1']['RTIIN'] = list(x_201)
            obj['NAMELIS1']['TIIN'] = list(y_201)
        except Exception as e:
            _log_error(shot, time, 'CNN_Ti', str(e))
    elif status['TXCS_status'] == 1:
        print(f'(CNN_Ti) Skipped at {time}s: No Te pedestal data available')

    # =====================================================================
    # LHW + ONETWO
    # =====================================================================
    if TE_STATUS == 0 and NE_STATUS == 0:
        try:
            onetwo_result_dir = os.path.join(output_base, f"{shot}", time_dir, 'onetwo_cnn')
            os.makedirs(onetwo_result_dir, exist_ok=True)

            rho, power, current = lower_onetwo_nn(
                shot, time, onetwo_result_dir,
                te_profile_nn=y_te_201, ne_profile_nn=y_ne_201,
                te_rho_nn=x_te_201, ne_rho_nn=x_ne_201,
            )

            # NAMELIS2 assembly
            obj['NAMELIS2']['extcurrf'] = [1.0]
            obj['NAMELIS2']['extcurrf_id'] = ['lhw']
            obj['NAMELIS2']['extcurrf_nj'] = [len(rho)]
            obj['NAMELIS2']['extcurrf_rho'] = list(rho)
            obj['NAMELIS2']['extcurrf_curr'] = list(current * 1e6 / 1e4)
            obj['NAMELIS2']['extcurrf_amps'] = [0.0]

            obj['NAMELIS2']['extqerf'] = [1.0]
            obj['NAMELIS2']['extqerf_id'] = ['lhw']
            obj['NAMELIS2']['extqerf_nj'] = [len(rho)]
            obj['NAMELIS2']['extqerf_rho'] = list(rho)
            obj['NAMELIS2']['extqerf_qe'] = list(power)
            obj['NAMELIS2']['extqerf_watts'] = [0.0]

            obj['NAMELIS2']['extqirf'] = [0.0]
            obj['NAMELIS2']['extqirf_id'] = ['lhw']
            obj['NAMELIS2']['extqirf_nj'] = [0]

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

            onetwo_save_path = os.path.join(onetwo_result_dir, 'inone')
            obj.write(onetwo_save_path)
            print('ecrh_written (CNN)')

            conn = Connection('202.127.204.42')
            TREE = 'efit_east'
            try:
                conn.openTree(TREE, shot)
                g_times = np.asarray(conn.get(r'data(\GTIME)').data(), dtype=np.float64).flatten()
                if len(g_times) == 0:
                    raise ValueError(f'efit_east tree has zero gfile time slices for shot {shot}')
                timeid = np.argmin(abs(g_times - time))
                gg = geqdsk.read_from_MDS(conn, timeid)
                conn.closeTree(TREE, shot)
            except Exception as gfile_err:
                try:
                    conn.closeTree(TREE, shot)
                except Exception:
                    pass
                raise RuntimeError(
                    f'Shot {shot} @ {time}s: gfile unavailable from MDSplus efit_east tree. '
                    f'This shot has no EFIT equilibrium data. ONETWO cannot run without gfile.'
                ) from gfile_err

            filename = os.path.join(onetwo_result_dir, "g0_input")
            geqdsk.save(gg, filename)

            _env = os.environ.copy()
            _env['LD_LIBRARY_PATH'] = '/usr/local/mdsplus/lib:/home/fusion/imd/onetwo5/lib:/home/fusion/imd/auto12/netcdf4.1.3/pgi-1410/lib:/home/fusion/imd/auto12/cfetr_bin/lib:/home/fusion/imd/auto12/hdf5/pgi-1410/lib:' + _env.get('LD_LIBRARY_PATH', '')
            subprocess.run([_ONETWO_EXE], cwd=onetwo_result_dir, check=True, env=_env)
            print(f'(CNN_onetwo) success at {time}s')

        except Exception as Error_onetwo:
            _log_error(shot, time, 'CNN_ONETWO', str(Error_onetwo), exc_info=True)


def main():
    start_time = time.time()
    shot = int(input('Shot number: '))

    conn = Connection('202.127.204.42')
    conn.openTree('TS_EAST', shot)
    TS_times = conn.get(r'dim_of(\Te_coreTS)').data()
    conn.closeTree('TS_EAST', shot)

    print(f'Found {len(TS_times)} time points')
    os.makedirs("results", exist_ok=True)

    pool = multiprocessing.Pool()
    task_args = [(t, shot) for t in TS_times]
    pool.map(process_time_point, task_args)
    pool.close()
    pool.join()

    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nTotal execution time: {int(hours):0>2}h {int(minutes):0>2}m {seconds:05.2f}s")


if __name__ == '__main__':
    main()
