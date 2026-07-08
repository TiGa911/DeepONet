# -*- coding: utf-8 -*-
"""
fetch_lhw_data.py — 从 MDSplus 获取 gfile + LHW 功率数据

对 paper_results/ 中已有 metadata.json 的每个时间点，通过 readmds() 下载
gfile 并读取 net LH power (PLHI2 - PLHR2)。

用法（服务器端）:
  pytorch_run paper/fetch_lhw_data.py

输出:
  paper_results/{shot}/
    gfiles/{real_time}_gfile     # gfile 平衡文件
    lhw_power.json               # {time_dir: {"p_in_kw": ..., "real_time": ...}, ...}
"""

import numpy as np
import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
RESULT_BASE = os.path.join(SCRIPT_DIR, 'paper_results')

from readMDS_onetwo import readmds
from MDSplus.connection import Connection
import geqdsk


def fetch_power(shot: int, time_val: float) -> float:
    """读取 net LH power (kW) = PLHI2 - PLHR2，窗口 [time-0.1, time]"""
    try:
        conn = Connection('202.127.204.12')
        conn.openTree('EAST', shot)

        PLHI2 = conn.get(r'data(\PLHI2)').data()
        PLHI2_times = conn.get(r'dim_of(\PLHI2)').data()
        PLHR2 = conn.get(r'data(\PLHR2)').data()
        PLHR2_times = conn.get(r'dim_of(\PLHR2)').data()

        conn.closeTree('EAST', shot)

        PLHI2_times = np.asarray(PLHI2_times, dtype=np.float64).flatten()
        PLHR2_times = np.asarray(PLHR2_times, dtype=np.float64).flatten()
        PLHI2 = np.asarray(PLHI2, dtype=np.float64).flatten()
        PLHR2 = np.asarray(PLHR2, dtype=np.float64).flatten()

        mask1 = (PLHI2_times >= time_val - 0.1) & (PLHI2_times <= time_val)
        mask2 = (PLHR2_times >= time_val - 0.1) & (PLHR2_times <= time_val)

        p_in = np.mean(PLHI2[mask1]) - np.mean(PLHR2[mask2])  # kW
        return float(np.nan_to_num(p_in, nan=0.0))
    except Exception as e:
        print(f'    Power read failed: {e}')
        return 0.0


def fetch_zeff(shot: int, time_val: float) -> float:
    """尝试从 readmds 获取 Zeff，失败返回 2.0"""
    try:
        data, status, _ = readmds(shot, time_val)
        z = data.get('zeff', 2.0)
        return float(z) if z is not None else 2.0
    except Exception:
        return 2.0


def main():
    print(f'{"="*60}')
    print(f'Fetch gfile + LHW power data')
    print(f'Scanning: {RESULT_BASE}/')
    print(f'{"="*60}')

    total = 0
    for shot_name in sorted(os.listdir(RESULT_BASE)):
        shot_path = os.path.join(RESULT_BASE, shot_name)
        if not os.path.isdir(shot_path):
            continue
        try:
            shot = int(shot_name)
        except ValueError:
            continue

        # 收集该炮号所有有 metadata 的时间点
        entries = []
        for td_name in os.listdir(shot_path):
            td_path = os.path.join(shot_path, td_name)
            if not os.path.isdir(td_path):
                continue
            meta_path = os.path.join(td_path, 'metadata.json')
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                entries.append({
                    'time_dir': td_name,
                    'time_val': meta.get('time'),
                    'real_time': meta.get('real_time'),
                })

        if not entries:
            continue

        print(f'\n--- Shot {shot}: {len(entries)} time points ---')

        # 读取 Zeff（一个炮号取一次即可，变化缓慢）
        t_mid = entries[len(entries) // 2]['time_val']
        zeff_val = fetch_zeff(shot, t_mid)
        print(f'  Zeff = {zeff_val:.2f}')

        # gfils 目录
        gfile_dir = os.path.join(shot_path, 'gfiles')
        os.makedirs(gfile_dir, exist_ok=True)

        power_data = {}

        for entry in entries:
            time_val = entry['time_val']
            # 读取 gfile
            try:
                data, status, real_time = readmds(shot, time_val)
                gfile_name = f'{shot}_{real_time:.3f}_gfile'
                gfile_path = os.path.join(gfile_dir, gfile_name)

                if not os.path.exists(gfile_path):
                    # readmds 已经自动保存了 gfile 到 CWD，移动过来
                    cwd_gfile = f'{shot}_{real_time}_gfile'
                    if os.path.exists(cwd_gfile):
                        os.rename(cwd_gfile, gfile_path)
                        print(f'  {entry["time_dir"]}: gfile saved as {gfile_name}')
                    else:
                        # 直接从 MDSplus 读并保存
                        conn = Connection('202.127.204.42')
                        conn.openTree('efit_east', shot)
                        g_times = np.asarray(
                            conn.get(r'data(\GTIME)').data(), dtype=np.float64
                        ).flatten()
                        timeid = np.argmin(np.abs(g_times - time_val))
                        gg = geqdsk.read_from_MDS(conn, timeid)
                        conn.closeTree('efit_east', shot)
                        geqdsk.save(gg, gfile_path)
                        print(f'  {entry["time_dir"]}: gfile saved (direct MDS)')
                else:
                    print(f'  {entry["time_dir"]}: gfile already exists')
            except Exception as e:
                print(f'  {entry["time_dir"]}: gfile FAILED - {e}')
                continue

            # 读取功率
            p_in = fetch_power(shot, time_val)
            power_data[entry['time_dir']] = {
                'time_val': time_val,
                'real_time': real_time,
                'p_in_kw': p_in,
                'zeff': zeff_val,
                'gfile': gfile_name,
            }
            print(f'    P_in = {p_in:.1f} kW')

            total += 1

        # 保存功率数据
        power_path = os.path.join(shot_path, 'lhw_power.json')
        with open(power_path, 'w') as f:
            json.dump(power_data, f, indent=2)
        print(f'  Power data saved: {power_path}')

    print(f'\n{"="*60}')
    print(f'Done: {total} time points processed')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
