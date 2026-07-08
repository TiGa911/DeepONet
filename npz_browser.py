# -*- coding: utf-8 -*-
"""
npz_browser.py — ProfileNet 训练数据 .npz 文件浏览器

用法:
    python npz_browser.py                              # 列出所有数据摘要
    python npz_browser.py --datatype ne                # 只看 ne 数据
    python npz_browser.py --shot 73100                 # 只看某炮号
    python npz_browser.py --file profile_nn_data/81481_Te_5d30000.npz  # 查看单个文件详情
    python npz_browser.py --plot ne                    # 随机绘制 5 个 ne 样本
    python npz_browser.py --plot Te --shot 81481       # 绘制某炮号的所有 Te 样本
    python npz_browser.py --bad                        # 检查并列出脏数据
"""

import os
import sys
import argparse
import numpy as np
from glob import glob
from collections import Counter

# 尝试导入绘图库
try:
    import matplotlib
    matplotlib.use('TkAgg')  # 交互式窗口
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

DATA_DIR = os.path.join(os.path.dirname(__file__), 'profile_nn_data')


def list_summary(datatype=None, shot=None, show_files=False):
    """列出所有 .npz 数据文件的摘要统计"""
    pattern = os.path.join(DATA_DIR, '*.npz')
    all_files = sorted(glob(pattern))

    if not all_files:
        print("ERROR: No .npz files found in", DATA_DIR)
        return

    # 按 datatype 分类
    for dt in ['Te', 'ne', 'Ti']:
        files = [f for f in all_files if f'_{dt}_' in os.path.basename(f) or f'synthetic_{dt}_' in f]
        if datatype and dt != datatype:
            continue
        if not files:
            continue

        real = [f for f in files if 'synthetic' not in os.path.basename(f)]
        synthetic = [f for f in files if 'synthetic' in os.path.basename(f)]

        if shot is not None:
            real = [f for f in real if f'{shot}_{dt}_' in os.path.basename(f)]
            synthetic = []  # synthetic don't have shot numbers

        filtered = real + synthetic
        if not filtered:
            print(f"\n{'='*60}")
            print(f"  {dt}: No files for shot {shot}")
            continue

        # 收集统计
        n_pts_list = []
        y_max_list = []
        y_min_list = []
        shots_set = set()

        for f in filtered:
            try:
                data = np.load(f)
                n_pts_list.append(len(data['X_rho']))
                y = data['Y']
                y_max_list.append(float(np.nanmax(y)))
                y_min_list.append(float(np.nanmin(y)))
                if 'synthetic' not in os.path.basename(f):
                    shots_set.add(os.path.basename(f).split(f'_{dt}_')[0])
            except Exception:
                continue

        n_pts_arr = np.array(n_pts_list)
        y_max_arr = np.array(y_max_list)
        y_min_arr = np.array(y_min_list)

        print(f"\n{'='*60}")
        print(f"  {dt} 数据集")
        print(f"{'='*60}")
        print(f"  样本数: {len(real)} 真实 + {len(synthetic)} 合成 = {len(filtered)} 总计")
        print(f"  炮号数: {len(shots_set)} 个")
        print(f"  散点数: {n_pts_arr.min():.0f} ~ {n_pts_arr.max():.0f} (均值 {n_pts_arr.mean():.1f})")
        print(f"  Y 范围: [{y_min_arr.min():.2f}, {y_max_arr.max():.2f}], 均值峰值 {y_max_arr.mean():.2f}")
        print(f"  路径: {DATA_DIR}")

        if show_files and len(filtered) <= 50:
            print(f"\n  文件列表:")
            for f in sorted(filtered):
                name = os.path.basename(f)
                print(f"    {name}")
        elif show_files:
            print(f"  (共 {len(filtered)} 个文件，太多不逐一列出)")


def view_file(filepath):
    """查看单个 .npz 文件的详细内容"""
    if not os.path.exists(filepath):
        # 尝试在 DATA_DIR 下搜索
        candidates = glob(os.path.join(DATA_DIR, f'*{filepath}*'))
        if len(candidates) == 1:
            filepath = candidates[0]
        elif len(candidates) > 1:
            print(f"找到 {len(candidates)} 个匹配文件，请指定完整路径:")
            for c in candidates[:10]:
                print(f"  {os.path.basename(c)}")
            return
        else:
            print(f"文件不存在: {filepath}")
            return

    data = np.load(filepath)
    name = os.path.basename(filepath)

    print(f"\n{'='*60}")
    print(f"  文件: {name}")
    print(f"{'='*60}")

    # 元数据
    for key in ['shot', 'time', 'datatype']:
        if key in data:
            val = data[key]
            if isinstance(val, np.ndarray):
                val = val.item()
            if key == 'time':
                print(f"  {key}: {float(val):.5f} s")
            else:
                print(f"  {key}: {val}")

    # 数组
    for key in ['X_rho', 'X_val', 'X_te_ped_rho', 'X_te_ped_val', 'Y']:
        if key in data:
            arr = data[key]
            print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
            print(f"         min={np.nanmin(arr):.4f}, max={np.nanmax(arr):.4f}, "
                  f"mean={np.nanmean(arr):.4f}, nans={np.isnan(arr).sum()}")

    # 物理合理性检查
    y = data['Y']
    xv = data['X_val']
    issues = []
    if np.nanmin(y) < -0.5:
        issues.append(f"Y 含负值 (min={np.nanmin(y):.2f})")
    if np.nanmax(y) > 20:
        issues.append(f"Y 峰值过高 (max={np.nanmax(y):.2f})")
    if np.nanmax(xv) > 20:
        issues.append(f"X_val 越界 (max={np.nanmax(xv):.2f})")
    if np.any(~np.isfinite(y)):
        issues.append("Y 含 NaN/Inf")
    d_y = np.diff(y)
    if np.any(d_y > 0.01):
        n_violations = (d_y > 0.01).sum()
        issues.append(f"Y 非单调 ({n_violations} 处上翘)")

    if issues:
        print(f"\n  [WARN] Issues:")
        for iss in issues:
            print(f"    - {iss}")
    else:
        print(f"\n  [OK] data clean")

    data.close()


def plot_samples(datatype, shot=None, n=5):
    """随机绘制 n 个样本的散点 + 拟合剖面"""
    if not HAS_MPL:
        print("需要 matplotlib: pip install matplotlib")
        return

    pattern = os.path.join(DATA_DIR, '*.npz')
    all_files = sorted(glob(pattern))

    if shot is not None:
        files = [f for f in all_files if f'{shot}_{datatype}_' in os.path.basename(f)]
    else:
        files = [f for f in all_files
                 if f'_{datatype}_' in os.path.basename(f) and 'synthetic' not in os.path.basename(f)]

        # 优先选真实数据，不够再用合成数据
        if len(files) < n:
            syn = [f for f in all_files if f'synthetic_{datatype}_' in os.path.basename(f)]
            files = files + syn

    if not files:
        print(f"没有找到 {datatype} 类型的文件")
        return

    files = files[:n]

    fig, axes = plt.subplots(1, min(n, len(files)), figsize=(4 * min(n, len(files)), 4))
    if len(files) == 1:
        axes = [axes]

    for i, (ax, f) in enumerate(zip(axes, files)):
        data = np.load(f)
        name = os.path.basename(f)

        x_rho = data['X_rho']
        x_val = data['X_val']
        y_201 = data['Y']

        # 散点
        ax.scatter(x_rho, x_val, c='green', s=20, alpha=0.7, label='Diagnostic')
        # 拟合剖面
        ax.plot(np.linspace(0, 1, len(y_201)), y_201, 'b-', lw=2, label='Fitted')

        # 标题
        if 'synthetic' in name:
            title = f"Synthetic {datatype} #{i}"
        else:
            title = os.path.splitext(name)[0].replace(f'_{datatype}_', f'\n{datatype} @ ')

        ax.set_title(title, fontsize=9)
        ax.set_xlabel(r'$\rho$')
        ax.set_xlim(0, 1)

        if datatype == 'Te':
            ax.set_ylabel('Te (keV)')
        elif datatype == 'ne':
            ax.set_ylabel('ne (10¹⁹ m⁻³)')
        else:
            ax.set_ylabel('Ti (keV)')

        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        data.close()

    plt.tight_layout()
    plt.show()


def check_bad_files():
    """检查所有 .npz 文件，列出数据异常的"""
    pattern = os.path.join(DATA_DIR, '*.npz')
    all_files = sorted(glob(pattern))

    bad = []
    for f in all_files:
        try:
            data = np.load(f)
            xv = data['X_val']
            yp = data['Y']
            x_max = float(np.nanmax(xv))
            y_max = float(np.nanmax(yp))
            y_min = float(np.nanmin(yp))

            reason = None
            if not np.isfinite(x_max) or x_max > 25.0 or x_max < -1.0:
                reason = f"X_val 越界 (max={x_max:.1f})"
            elif not np.isfinite(y_max) or y_max > 30.0:
                reason = f"Y 峰值异常 (max={y_max:.1f})"
            elif not np.isfinite(y_min) or y_min < -1.0:
                reason = f"Y 含负值 (min={y_min:.2f})"
            elif np.any(~np.isfinite(yp)):
                reason = "Y 含 NaN/Inf"

            if reason:
                bad.append((os.path.basename(f), reason))
            data.close()
        except Exception as e:
            bad.append((os.path.basename(f), f"读取失败: {e}"))

    if bad:
        print(f"发现 {len(bad)} 个异常文件:\n")
        for name, reason in bad:
            print(f"  {name}")
            print(f"    → {reason}\n")
    else:
        print(f"[OK] all {len(all_files)} files clean")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='ProfileNet 训练数据 .npz 文件浏览器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python npz_browser.py                          # 列出摘要
    python npz_browser.py --datatype ne            # 只看 ne
    python npz_browser.py --shot 81481             # 只看某炮号
    python npz_browser.py --file 81481_Te_5.30000.npz  # 单文件详情
    python npz_browser.py --plot Te                # 绘制 Te 样本
    python npz_browser.py --bad                    # 检查脏数据
        """)


    parser.add_argument('--datatype', '-d', choices=['Te', 'ne', 'Ti'],
                        help='按诊断类型过滤')
    parser.add_argument('--shot', '-s', type=int,
                        help='按炮号过滤')
    parser.add_argument('--file', '-f', type=str,
                        help='查看单个文件详情')
    parser.add_argument('--plot', '-p', choices=['Te', 'ne', 'Ti'],
                        help='绘制样本图')
    parser.add_argument('--bad', '-b', action='store_true',
                        help='检查异常数据')
    parser.add_argument('--list', '-l', action='store_true',
                        help='列出所有文件名')

    args = parser.parse_args()

    if args.bad:
        check_bad_files()
    elif args.file:
        view_file(args.file)
    elif args.plot:
        plot_samples(args.plot, shot=args.shot, n=5 if args.shot is None else 20)
    else:
        list_summary(datatype=args.datatype, shot=args.shot, show_files=args.list)


if __name__ == '__main__':
    main()
