# -*- coding: utf-8 -*-
"""
quick_scan.py — 快速扫描一小批候选炮号（仅扫描几十个，1-2 分钟完成）

策略：
  1. 从训练集空白区中手选代表性炮号区间
  2. 每区间只检查首尾 + 中间几个炮号
  3. 只检查树是否存在（openTree），不读数据

用法：
  python quick_scan.py          # 本地运行（需 MDSplus 联网）
  python quick_scan.py --server # 服务器上运行（用 python3）
"""

import sys
from MDSplus.connection import Connection

MDS_IP = '202.127.204.42'

# 训练集炮号
TRAINING = {
    60150, 62200, 62275, 62500, 62575, 62650, 62850, 62944, 62950,
    63425, 63486, 63487, 63488, 63496, 63500, 63501, 63504, 63505,
    63510, 63514, 63515, 63516, 63517, 63518, 63519, 63520, 63524,
    63525, 63548, 63575, 63625, 63650, 63800, 66675, 67425, 69100,
    70100, 70900, 73100, 75150, 81481,
    124092, 124100, 124102, 124104, 124105, 124107, 128200,
    137800, 137809, 137817, 137937, 137951, 138050, 138450, 138900, 138950, 139850,
    140087, 140088, 140089, 140090, 140091, 140092, 140093, 140094, 140095, 140096,
    140277, 140278, 140280, 140281, 140282, 140283, 140285, 140287, 140288, 140289,
    140290, 140291, 140292, 140293, 140295, 140296, 140299, 140300, 140301, 140302,
    140303, 140305,
    140661, 140664, 140665, 140671, 140672, 140673, 140675, 140676, 140677, 140678,
    140679, 140680, 140681, 140682, 140683, 140684, 140687, 140691, 140692, 140693,
    140694, 140695, 140696, 140699, 140707, 140709, 140711, 140712, 140713, 140714,
    140715, 140716, 140717, 140718, 140720, 140722, 140723, 140724, 140727, 140728,
    140733, 140734, 140737, 140738, 140739,
    143600, 143618, 143620, 143624, 143628, 143634,
    145000, 145100, 145200, 145400, 145500, 145600, 145800,
    146000, 146200, 146500, 146700, 146900,
    148000, 148100, 148600,
    149000, 149400, 149500, 149600,
}

# 精准候选：训练集空白区中已知有实验的炮号段
# (shot, label)
CANDIDATES = [
    # === 2015-2019 空白区 (82k-124k) ===
    (83000, '2015 campaign'),
    (85000, '2015 campaign'),
    (87000, '2015 campaign'),
    (90000, '2016 campaign'),
    (93000, '2016 campaign'),
    (96000, '2016 campaign'),
    (99000, '2017 campaign'),
    (102000, '2017 campaign'),
    (105000, '2017 campaign'),
    (108000, '2018 campaign'),
    (111000, '2018 campaign'),
    (114000, '2018 campaign'),
    (117000, '2019 campaign'),
    (120000, '2019 campaign'),
    (123000, '2019 campaign'),

    # === 2020-2021 空白区 (125k-137k) ===
    (126000, '2020 campaign'),
    (128000, '2020 campaign'),  # 128200 在训练集中但附近可能有
    (129000, '2020 campaign'),
    (131000, '2020 campaign'),
    (133000, '2021 campaign'),
    (135000, '2021 campaign'),
    (137000, '2021 campaign'),

    # === 2024-2025 新炮号 (>150k) ===
    (150000, '2024 campaign'),
    (151000, '2024 campaign'),
    (152000, '2024 campaign'),
    (153000, '2024 campaign'),
    (154000, '2024 campaign'),
    (155000, '2024 campaign'),
    (155500, '2024 campaign'),
    (156000, '2024 campaign'),
    (156500, '2025 campaign'),
    (157000, '2025 campaign'),
    (158000, '2025 campaign'),
    (159000, '2025 campaign'),
    (160000, '2025 campaign'),
]


def check_tree(conn, shot, tree_name):
    """快速检查树是否存在（只 openTree，不读数据）"""
    try:
        conn.openTree(tree_name, shot)
        conn.closeTree(tree_name, shot)
        return True
    except Exception:
        return False


def find_nearest_valid(conn, shot, direction='up'):
    """在候选炮号附近找有效炮号（±100 范围）"""
    step = 1 if direction == 'up' else -1
    for delta in range(0, 200):
        test_shot = shot + delta * step
        if test_shot in TRAINING:
            continue
        if test_shot < 60000 or test_shot > 170000:
            break
        # 快速检查 TS 树
        try:
            conn.openTree('TS_EAST', test_shot)
            conn.closeTree('TS_EAST', test_shot)
            return test_shot
        except Exception:
            continue
    return None


def main():
    print(f'Connecting to {MDS_IP} ...')
    conn = Connection(MDS_IP)
    print('Connected.\n')

    results = []
    for shot, label in CANDIDATES:
        # 跳过训练集
        if shot in TRAINING:
            continue

        # 快速检查四棵树
        ts = check_tree(conn, shot, 'TS_EAST')
        refl = check_tree(conn, shot, 'ReflJ_EAST')
        txcs = check_tree(conn, shot, 'TXCS_EAST')
        gfile = check_tree(conn, shot, 'efit_east')

        diag_count = sum([ts, refl, txcs, gfile])
        status = '★' if diag_count >= 3 else ('·' if diag_count >= 1 else '✗')
        print(f'  {status} {shot:<8} ({label:<18}) TS={ts} Refl={refl} TXCS={txcs} GF={gfile}')

        if ts and refl and gfile:  # 至少需要 TS + Refl + gfile
            results.append((shot, label, ts, refl, txcs, gfile))

    print(f'\n{"="*50}')
    print(f'Shots with TS + Refl + gfile: {len(results)}')

    # 找到的炮号
    good = [(s, l, tx) for s, l, ts, r, tx, g in results]
    good_3diag = [(s, l) for s, l, tx in good if tx]  # TS+Refl+TXCS

    print(f'  With TXCS (3 diagnostics): {len(good_3diag)}')
    print(f'  Without TXCS: {len(good) - len(good_3diag)}')

    if good_3diag:
        print(f'\n★★★ 3-diagnostic candidates (BEST for paper):')
        for s, l in good_3diag:
            print(f'  {s} ({l})')

    if good:
        print(f'\n★★ 2-diagnostic candidates (OK, Ti will be missing):')
        for s, l, tx in good:
            if not tx:
                print(f'  {s} ({l})')

    print(f'\nDone. If few candidates found, try finer scanning around promising ranges.')
    print(f'Example: python quick_scan.py --fine 150000 151000')


if __name__ == '__main__':
    fine_start = None
    fine_end = None
    if '--fine' in sys.argv:
        try:
            idx = sys.argv.index('--fine')
            fine_start = int(sys.argv[idx + 1])
            fine_end = int(sys.argv[idx + 2])
        except Exception:
            print('Usage: python quick_scan.py --fine <start_shot> <end_shot>')
            sys.exit(1)

    if fine_start:
        # 精细模式：逐炮扫描
        print(f'Fine scan: {fine_start} - {fine_end}')
        conn = Connection(MDS_IP)
        for shot in range(fine_start, fine_end + 1):
            if shot in TRAINING:
                continue
            ts = check_tree(conn, shot, 'TS_EAST')
            refl = check_tree(conn, shot, 'ReflJ_EAST')
            txcs = check_tree(conn, shot, 'TXCS_EAST')
            gfile = check_tree(conn, shot, 'efit_east')
            diag = sum([ts, refl, txcs, gfile])
            if diag >= 3:
                print(f'  ★ {shot}: TS={ts} R={refl} TX={txcs} GF={gfile}')
    else:
        main()
