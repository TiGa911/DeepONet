# -*- coding: utf-8 -*-
"""
label_h98.py — 为现有 NPZ 训练数据添加 H98 标签索引

扫描 profile_nn_data/ 中的所有 .npz 文件，提取 (shot, time) 对，
批量查询 MDSplus energy_east 树获取 H98 值，输出 CSV 索引文件。

用法:
  python3 paper/label_h98.py --data ./profile_nn_data --output ./profile_nn_data/h98_index.csv
  python3 paper/label_h98.py --data ./profile_nn_data --output ./profile_nn_data/h98_index.csv --incremental

输出 CSV 格式:
  filename,shot,time,datatype,h98,plasma_mode
  140087_Ti_5d10000.npz,140087,5.1,Ti,1.23,H

血浆模式分类（与 fitting_mtanh.judge_plasma_mode() 一致）:
  - h98 >= 0.7 → H (H-mode)
  - 0 < h98 < 0.7 → L (L-mode)
  - h98 < 0 或 NaN → unknown
"""

import os
import sys
import csv
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    from MDSplus.connection import Connection
    HAS_MDS = True
except ImportError:
    HAS_MDS = False


def classify_plasma_mode(h98, threshold=0.7):
    """按 H98 值分类等离子体模式。与 fitting_mtanh.judge_plasma_mode 一致。"""
    if h98 is None or np.isnan(h98) or h98 < 0:
        return 'unknown'
    return 'H' if h98 >= threshold else 'L'


def extract_files(data_dir):
    """扫描数据目录，提取所有 NPZ 文件的元信息。

    Returns:
        files: [(filename, shot, time, datatype), ...]
        unique_pairs: {(shot, time)} 唯一对集合（用于批量 H98 查询）
    """
    data_path = Path(data_dir)
    all_npz = sorted(data_path.glob('*.npz'))

    files = []
    unique_pairs = set()

    for npz_path in all_npz:
        fname = npz_path.name
        # 跳过合成数据（shot=-1）
        if fname.startswith('synthetic'):
            continue

        try:
            data = np.load(npz_path)
            shot = int(data.get('shot', -1))
            time = float(data.get('time', -1))
            datatype = str(data.get('datatype', 'unknown'))
            data.close()
        except Exception as e:
            print(f"  警告: 无法读取 {fname}: {e}")
            continue

        if shot < 0 or time < 0:
            continue

        files.append((fname, shot, time, datatype))
        unique_pairs.add((shot, time))

    return files, unique_pairs


def query_h98_batch(unique_pairs, conn=None):
    """批量查询 MDSplus 获取 (shot, time) 的 H98 值。

    优化：按 shot 分组，每炮只打开一次 energy_east 树。

    Returns:
        h98_map: {(shot, time): h98_value}
    """
    if conn is None:
        if not HAS_MDS:
            raise RuntimeError("MDSplus 不可用")
        conn = Connection('202.127.204.42')

    # 按 shot 分组
    shot_times = defaultdict(list)
    for shot, time in unique_pairs:
        shot_times[shot].append(time)

    h98_map = {}
    total = len(unique_pairs)
    done = 0

    for shot, times in sorted(shot_times.items()):
        try:
            conn.openTree('energy_east', shot)
            H98_times = conn.get(r'dim_of(\H98_MHD)').data()
            H98_data = conn.get(r'data(\H98_MHD)').data()
            conn.closeTree('energy_east', shot)

            H98_times = np.asarray(H98_times, dtype=np.float64).flatten()
            H98_data = np.asarray(H98_data, dtype=np.float64).flatten()

            # 对该炮的每个时间点，找最近的 H98 时间
            for t in times:
                timeid = np.argmin(np.abs(H98_times - t))
                h98_value = float(H98_data[timeid])
                h98_map[(shot, t)] = h98_value
                done += 1
        except Exception as e:
            print(f"  H98 读取失败 shot={shot}: {e}")
            for t in times:
                h98_map[(shot, t)] = np.nan
                done += 1

        if done % 500 == 0:
            print(f"  进度: {done}/{total}")

    return h98_map


def load_existing_csv(csv_path):
    """加载已有 CSV 索引，返回已处理的 filename 集合。"""
    existing = set()
    if os.path.exists(csv_path):
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.add(row['filename'])
    return existing


def main():
    parser = argparse.ArgumentParser(description='为 NPZ 训练数据添加 H98 标签索引')
    parser.add_argument('--data', type=str, default='./profile_nn_data',
                        help='NPZ 数据目录')
    parser.add_argument('--output', type=str, default='./profile_nn_data/h98_index.csv',
                        help='输出 CSV 路径')
    parser.add_argument('--incremental', action='store_true',
                        help='增量模式：跳过已有 CSV 中的文件，只处理新增')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='H/L 模式判定阈值（默认 0.7）')
    args = parser.parse_args()

    if not HAS_MDS:
        print("错误: MDSplus 不可用。请在服务器上运行此脚本。")
        sys.exit(1)

    print(f"扫描数据目录: {args.data}")
    files, unique_pairs = extract_files(args.data)

    print(f"  找到 {len(files)} 个 NPZ 文件")
    print(f"  唯一 (shot, time) 对: {len(unique_pairs)}")

    # 增量模式：跳过已处理的文件
    if args.incremental:
        existing = load_existing_csv(args.output)
        if existing:
            n_skip = len(existing)
            files = [f for f in files if f[0] not in existing]
            # 重新计算 unique_pairs
            unique_pairs = set()
            for fname, shot, time, _ in files:
                unique_pairs.add((shot, time))
            print(f"  增量模式: 跳过 {n_skip} 个已有文件, 剩余 {len(files)} 个")

    if not files:
        print("没有需要处理的新文件。")
        return

    # 连接 MDSplus 并查询 H98
    print("\n查询 MDSplus energy_east 树...")
    conn = Connection('202.127.204.42')
    h98_map = query_h98_batch(unique_pairs, conn=conn)
    print(f"  完成: {len(h98_map)} 个 H98 值")

    # 写 CSV
    mode = 'a' if (args.incremental and os.path.exists(args.output)) else 'w'
    with open(args.output, mode, newline='') as f:
        writer = csv.writer(f)
        if mode == 'w':
            writer.writerow(['filename', 'shot', 'time', 'datatype', 'h98', 'plasma_mode'])

        for fname, shot, time, datatype in files:
            h98 = h98_map.get((shot, time), np.nan)
            mode_label = classify_plasma_mode(h98, threshold=args.threshold)
            writer.writerow([fname, shot, f"{time:.5f}", datatype,
                             f"{h98:.6f}" if np.isfinite(h98) else "-1.0",
                             mode_label])

    # 统计摘要
    h98_values = [h98_map.get((shot, time), np.nan)
                  for _, shot, time, _ in files]
    valid = [h for h in h98_values if np.isfinite(h) and h >= 0]
    h_count = sum(1 for h in valid if h >= args.threshold)
    l_count = sum(1 for h in valid if h < args.threshold)
    unknown = len(h98_values) - len(valid)

    print(f"\n{'='*50}")
    print(f"CSV 已保存: {args.output}")
    print(f"  总样本: {len(files)}")
    print(f"  H 模 (H98>={args.threshold:.1f}): {h_count}")
    print(f"  L 模 (H98<{args.threshold:.1f}): {l_count}")
    print(f"  H98 不可用: {unknown}")

    # 按诊断类型统计
    for dt in ['Te', 'ne', 'Ti']:
        dt_files = [f for f in files if f[3] == dt]
        dt_h98 = [h98_map.get((s, t), np.nan) for _, s, t, _ in dt_files]
        dt_valid = [h for h in dt_h98 if np.isfinite(h) and h >= 0]
        dt_h = sum(1 for h in dt_valid if h >= args.threshold)
        dt_l = sum(1 for h in dt_valid if h < args.threshold)
        dt_u = len(dt_files) - len(dt_valid)
        print(f"  {dt}: total={len(dt_files)}, H={dt_h}, L={dt_l}, unknown={dt_u}")


if __name__ == '__main__':
    main()
