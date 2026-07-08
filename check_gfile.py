#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查 EAST 炮号在 MDSplus 上是否有 gfile (EFIT 平衡) 数据。

用途：在运行 ONETWO pipeline 前批量过滤无效炮号/时间点。
用法：
  python check_gfile.py --shots 81481 124092 140660
  python check_gfile.py --start 140000 --end 141000
  python check_gfile.py --shots 81481 --times 3.0 5.0 7.0
"""

import sys, io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import argparse
import numpy as np
from MDSplus.connection import Connection

MDSIP = '202.127.204.42'


def check_gfile(shot, times=None, max_time_gap=0.5):
    """检查某个炮号是否有 gfile 数据。

    Args:
        shot: 炮号
        times: 要检查的时间点列表。None 则只检查是否有任何 gfile 时间片
        max_time_gap: 允许的最大时间偏差（秒），超过此值视为不可用

    Returns:
        dict: {shot, available, gfile_times, matched_times, missing_times}
    """
    result = {
        'shot': shot,
        'available': False,
        'gfile_times': [],
        'gfile_count': 0,
        'matched_times': [],
        'missing_times': [],
        'time_gaps': [],
    }

    try:
        conn = Connection(MDSIP)
        try:
            conn.openTree('efit_east', shot)
        except Exception:
            result['available'] = False
            return result

        g_times = np.asarray(
            conn.get(r'data(\GTIME)').data(), dtype=np.float64
        ).flatten()
        conn.closeTree('efit_east', shot)

        result['gfile_times'] = list(g_times)
        result['gfile_count'] = len(g_times)
        result['available'] = len(g_times) > 0

        if times is not None and len(g_times) > 0:
            for t in times:
                dt = abs(g_times - t)
                min_dt = dt.min()
                if min_dt <= max_time_gap:
                    result['matched_times'].append(float(t))
                    result['time_gaps'].append(float(min_dt))
                else:
                    result['missing_times'].append(
                        (float(t), float(min_dt))
                    )

    except Exception as e:
        result['available'] = False

    return result


def main():
    parser = argparse.ArgumentParser(description='检查 EAST gfile 可用性')
    parser.add_argument('--shots', type=int, nargs='+', help='要检查的炮号列表')
    parser.add_argument('--start', type=int, help='起始炮号（批量扫描）')
    parser.add_argument('--end', type=int, help='结束炮号（批量扫描）')
    parser.add_argument('--times', type=float, nargs='+',
                        help='要检查的具体时间点（默认只检查有无任何时间片）')
    parser.add_argument('--max-gap', type=float, default=0.5,
                        help='允许的最大时间偏差（秒，默认0.5）')
    parser.add_argument('--summary-only', action='store_true',
                        help='只显示汇总统计')

    args = parser.parse_args()

    shots = []
    if args.shots:
        shots = args.shots
    elif args.start and args.end:
        import time as _time
        shots = list(range(args.start, args.end + 1))
    else:
        print('请指定 --shots 或 --start/--end')
        return

    print(f'检查 {len(shots)} 个炮号的 gfile 可用性...')
    if args.times:
        print(f'时间点: {args.times}')

    available = []
    unavailable = []

    for shot in shots:
        result = check_gfile(shot, args.times, args.max_gap)
        if result['available']:
            available.append(result)
            if not args.summary_only:
                gfr = result['gfile_times']
                t_range = f'[{gfr[0]:.1f}, {gfr[-1]:.1f}]s' if gfr else 'N/A'
                print(f'  ✓ {shot}: {result["gfile_count"]} 时间片 {t_range}')
                if args.times:
                    for t, gap in zip(result['matched_times'], result['time_gaps']):
                        print(f'      t={t:.3f}s: Δt={gap*1000:.0f}ms')
                    for t, gap in result['missing_times']:
                        print(f'      ✗ t={t:.3f}s: 最近 Δt={gap*1000:.0f}ms > {args.max_gap*1000:.0f}ms')
        else:
            print(f'  ✗ {shot}: 无 gfile 数据')

    # 汇总
    print(f'\n=== 汇总 ===')
    print(f'可用炮号: {len(available)}/{len(shots)}')
    print(f'不可用炮号: {len(unavailable)}/{len(shots)}')

    if unavailable:
        print(f'\n无 gfile 数据的炮号:')
        for r in unavailable:
            print(f'  {r["shot"]}')

    if available:
        total_slices = sum(r['gfile_count'] for r in available)
        print(f'\n可用炮号的 gfile 时间片总数: {total_slices}')

    # 导出可用炮号列表（方便后续使用）
    if available:
        avail_shots = [str(r['shot']) for r in available]
        print(f'\n可直接用于 pipeline 的炮号列表:')
        print(' '.join(avail_shots))


if __name__ == '__main__':
    main()
