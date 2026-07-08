# -*- coding: utf-8 -*-
"""
generate_paper_figures.py — 论文图表生成（Phase 2）

从 NPZ + metrics CSV 读取数据，生成 7 张论文图表。

用法：
  python3 paper/generate_paper_figures.py
  python3 paper/generate_paper_figures.py --output ./paper_results

输出：paper_results/figures/ 目录下 7 张 PNG + 1 个 CSV
"""

import numpy as np
import os
import sys
import json
import csv
import argparse
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from metrics_utils import (
    METHOD_SPECS, NN_METHODS, ALL_METHODS, DIAGNOSTICS, DIAGNOSTIC_SPECS,
    ARCHITECTURE_TABLE,
)

parser = argparse.ArgumentParser(description='Generate paper figures from NPZ + metrics')
parser.add_argument('--output', default=os.path.join(SCRIPT_DIR, 'paper_results'),
                    help='Output base directory')
args = parser.parse_args()
RESULT_BASE = args.output
FIG_DIR = os.path.join(RESULT_BASE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ================================================================
# 数据加载
# ================================================================

def load_metrics():
    """加载 all_metrics.csv 为 dict。"""
    csv_path = os.path.join(RESULT_BASE, 'all_metrics.csv')
    if not os.path.exists(csv_path):
        print(f'WARNING: {csv_path} not found — run compute_all_metrics.py first')
        return []
    with open(csv_path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_npz_metadata():
    """加载所有 metadata.json，返回 {(shot, time_dir): metadata}。"""
    metas = {}
    for shot_name in os.listdir(RESULT_BASE):
        shot_path = os.path.join(RESULT_BASE, shot_name)
        if not os.path.isdir(shot_path):
            continue
        try:
            int(shot_name)
        except ValueError:
            continue
        for td_name in os.listdir(shot_path):
            meta_path = os.path.join(shot_path, td_name, 'metadata.json')
            if os.path.exists(meta_path):
                try:
                    with open(meta_path) as f:
                        metas[(int(shot_name), td_name)] = json.load(f)
                except Exception:
                    pass
    return metas


def load_profile(shot, time_dir, method, diag):
    """加载单个剖面 NPZ，返回 (rho, y) 或 (None, None)。"""
    npz_path = os.path.join(RESULT_BASE, str(shot), time_dir,
                            'profiles', f'{method}_{diag}.npz')
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        return np.asarray(data['rho'], dtype=np.float64), np.asarray(data['y'], dtype=np.float64)
    return None, None


def load_scatter(shot, time_dir, diag):
    """加载原始散点 NPZ，返回 (x, y) 或 (None, None)。"""
    npz_path = os.path.join(RESULT_BASE, str(shot), time_dir, 'raw', f'scatter_{diag}.npz')
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        return np.asarray(data['rho'], dtype=np.float64), np.asarray(data['y'], dtype=np.float64)
    return None, None


# ================================================================
# 选择代表性时间点
# ================================================================

def select_representative_points(metas, n=2, mode='H'):
    """选 H98 最高/最低的 n 个时间点（优先来自不同炮号）。

    H-mode: H98 >= 0.7, 排除 H98==0（无效值），按 H98 降序。
    L-mode: H98 < 0.7, 排除 H98==0, 按 H98 升序。
    若有效点不足，用最低 H98 的 H-mode 点代替。
    """
    scored_all = []
    for (shot, td), meta in metas.items():
        h98 = meta.get('h98')
        if h98 is None:
            continue
        scored_all.append((shot, td, h98, meta))

    if not scored_all:
        return []

    # 排除 H98 == 0 的无效数据
    valid = [(s, td, h, m) for s, td, h, m in scored_all if h > 0.001]

    if mode == 'H':
        # H-mode: H98 >= 0.7, 降序
        hmode = [(s, td, h, m) for s, td, h, m in valid if h >= 0.7]
        if hmode:
            scored = sorted(hmode, key=lambda x: x[2], reverse=True)
        else:
            scored = sorted(valid, key=lambda x: x[2], reverse=True)
    else:
        # L-mode: H98 < 0.7, 升序
        lmode = [(s, td, h, m) for s, td, h, m in valid if h < 0.7]
        if lmode:
            scored = sorted(lmode, key=lambda x: x[2])
        else:
            # 无真 L-mode，用最低 H98 的 H-mode 代替
            scored = sorted(valid, key=lambda x: x[2])

    selected = []
    used_shots = set()
    for shot, td, h98, meta in scored:
        if shot not in used_shots:
            selected.append((shot, td, h98, meta))
            used_shots.add(shot)
        if len(selected) >= n:
            break
    return selected


# ================================================================
# Figure 2: H-mode 典型剖面叠加
# ================================================================

def fig_hmode_overlay(metas):
    """3 diag × 2 columns H-mode overlay."""
    selected = select_representative_points(metas, n=2, mode='H')
    if not selected:
        print('  No H-mode points found, using all available')
        scored = [(s, td, meta.get('h98', 0), meta) for (s, td), meta in metas.items()]
        scored.sort(key=lambda x: x[2], reverse=True)
        selected = [(s, td, h, m) for s, td, h, m in scored[:2]]

    fig, axes = plt.subplots(3, 2, figsize=(14, 16), dpi=300)
    fig.suptitle('Typical H-mode Profile Fitting — 5 Methods Comparison',
                 fontsize=14, fontweight='bold', y=0.995)

    for col, (shot, td, h98, meta) in enumerate(selected):
        for row, diag in enumerate(DIAGNOSTICS):
            ax = axes[row][col]
            spec = DIAGNOSTIC_SPECS[diag]

            # 原始散点
            sx, sy = load_scatter(shot, td, diag)
            if sx is not None and sy is not None:
                if diag == 'Te':
                    sy = sy / 1000.0  # eV → keV
                ax.scatter(sx, sy, marker='.', c='gray', alpha=0.3, s=6, zorder=1)

            # 各方法
            for method in ALL_METHODS:
                ms = METHOD_SPECS[method]
                x, y = load_profile(shot, td, method, diag)
                if x is not None and y is not None:
                    ax.plot(x, y, color=ms['color'], ls=ms['ls'], lw=ms['lw'],
                            label=ms['label'], zorder=2)

            ax.set_xlabel(r'$\rho$', fontsize=10)
            ax.set_ylabel(spec['ylabel'], fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_title(f'Shot {shot}  {td}s  (H98={h98:.2f})', fontsize=10)
            ax.grid(True, alpha=0.3)

    # 单一共用图例
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    path = os.path.join(FIG_DIR, 'fig2_hmode_overlay.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  fig2_hmode_overlay.png saved')


# ================================================================
# Figure 3: L-mode 典型剖面叠加
# ================================================================

def fig_lmode_overlay(metas):
    """3 diag × 2 columns L-mode overlay。若无 L-mode，用最低 H98 代表。"""
    selected = select_representative_points(metas, n=2, mode='L')
    if not selected:
        print('  No pure L-mode, using lowest-H98 points')
        scored = [(s, td, meta.get('h98', 99), meta) for (s, td), meta in metas.items()]
        scored.sort(key=lambda x: x[2])
        selected = [(s, td, h, m) for s, td, h, m in scored[:2]]

    fig, axes = plt.subplots(3, 2, figsize=(14, 16), dpi=300)
    # 检查是否是真 L-mode
    is_true_lmode = any(h < 0.7 for _, _, h, _ in selected)
    mode_label = 'L-mode' if is_true_lmode else 'Lowest-H98 Profiles'
    fig.suptitle(f'Typical {mode_label} Profile Fitting — 5 Methods Comparison',
                 fontsize=14, fontweight='bold', y=0.995)

    for col, (shot, td, h98, meta) in enumerate(selected):
        for row, diag in enumerate(DIAGNOSTICS):
            ax = axes[row][col]
            spec = DIAGNOSTIC_SPECS[diag]

            sx, sy = load_scatter(shot, td, diag)
            if sx is not None and sy is not None:
                if diag == 'Te':
                    sy = sy / 1000.0
                ax.scatter(sx, sy, marker='.', c='gray', alpha=0.3, s=6, zorder=1)

            for method in ALL_METHODS:
                ms = METHOD_SPECS[method]
                x, y = load_profile(shot, td, method, diag)
                if x is not None and y is not None:
                    ax.plot(x, y, color=ms['color'], ls=ms['ls'], lw=ms['lw'],
                            label=ms['label'], zorder=2)

            ax.set_xlabel(r'$\rho$', fontsize=10)
            ax.set_ylabel(spec['ylabel'], fontsize=10)
            ax.set_xlim(0, 1)
            mode_str = 'L' if h98 < 0.7 else f'H({h98:.2f})'
            ax.set_title(f'Shot {shot}  {td}s  H98={h98:.2f} [{mode_str}]', fontsize=10)
            ax.grid(True, alpha=0.3)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    path = os.path.join(FIG_DIR, 'fig3_lmode_overlay.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  fig3_lmode_overlay.png saved')


# ================================================================
# Figure 4: 台基区域放大
# ================================================================

def fig_pedestal_zoom(metas):
    """2×1 Te pedestal zoom (ρ=0.85-1.0)。"""
    selected = select_representative_points(metas, n=2, mode='H')
    if not selected:
        scored = [(s, td, meta.get('h98', 0), meta) for (s, td), meta in metas.items()]
        scored.sort(key=lambda x: x[2], reverse=True)
        selected = [(s, td, h, m) for s, td, h, m in scored[:2]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    fig.suptitle('Pedestal Region Zoom (Te, ρ = 0.85–1.0)',
                 fontsize=13, fontweight='bold')

    for col, (shot, td, h98, meta) in enumerate(selected):
        ax = axes[col]

        sx, sy = load_scatter(shot, td, 'Te')
        if sx is not None and sy is not None:
            sy = sy / 1000.0
            mask = sx >= 0.80
            ax.scatter(sx[mask], sy[mask], marker='.', c='gray', alpha=0.35, s=10, zorder=1)

        for method in ALL_METHODS:
            ms = METHOD_SPECS[method]
            x, y = load_profile(shot, td, method, 'Te')
            if x is not None and y is not None:
                mask = x >= 0.80
                ax.plot(x[mask], y[mask], color=ms['color'], ls=ms['ls'], lw=ms['lw'],
                        label=ms['label'], zorder=2)

        ax.set_xlabel(r'$\rho$', fontsize=11)
        ax.set_ylabel('Te (keV)', fontsize=11)
        ax.set_xlim(0.85, 1.0)
        ax.set_title(f'Shot {shot}  {td}s  H98={h98:.2f}', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0.95, color='gray', ls=':', lw=0.8, alpha=0.7)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    path = os.path.join(FIG_DIR, 'fig4_pedestal_zoom.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  fig4_pedestal_zoom.png saved')


# ================================================================
# Figure 5: 芯部峰值度散点
# ================================================================

def fig_peakedness_scatter(metrics_rows):
    """3×4 subplots: mtanh vs NN peakedness, colored by shot."""
    if not metrics_rows:
        print('  No metrics data, skipping fig5')
        return

    shot_colors = {156005: '#E63946', 156010: '#2A9D8F',
                   156100: '#457B9D', 156400: '#F4A261'}

    fig, axes = plt.subplots(3, 4, figsize=(18, 14), dpi=300)
    fig.suptitle('Core Peakedness: NN vs mtanh (Reference)',
                 fontsize=13, fontweight='bold')

    for row, diag in enumerate(DIAGNOSTICS):
        for col, method in enumerate(NN_METHODS):
            ax = axes[row][col]

            # 提取该 diagnostic+method 的所有数据点
            pts = []
            for r in metrics_rows:
                if r['diagnostic'] == diag and r['method'] == method:
                    try:
                        pts.append({
                            'shot': int(r['shot']),
                            'x': float(r['mtanh_peakedness']),
                            'y': float(r['nn_peakedness']),
                        })
                    except (ValueError, KeyError):
                        continue

            if not pts:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center',
                        fontsize=10, color='gray')
                ax.set_title(f'{diag} — {METHOD_SPECS[method]["label"]}', fontsize=10)
                continue

            for shot in sorted(shot_colors):
                shot_pts = [p for p in pts if p['shot'] == shot]
                if shot_pts:
                    ax.scatter([p['x'] for p in shot_pts], [p['y'] for p in shot_pts],
                               c=shot_colors[shot], label=f'{shot}', alpha=0.7, s=30)

            # y=x line
            all_x = [p['x'] for p in pts]
            all_y = [p['y'] for p in pts]
            xlim = [min(all_x) * 0.9, max(all_x) * 1.1]
            ylim = [min(all_y) * 0.9, max(all_y) * 1.1]
            lim = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
            ax.plot(lim, lim, 'k--', lw=0.8, alpha=0.4)
            ax.set_xlim(lim)
            ax.set_ylim(lim)

            # Pearson r
            if len(all_x) > 2:
                r = np.corrcoef(all_x, all_y)[0, 1]
                ax.text(0.05, 0.92, f'r={r:.3f}', transform=ax.transAxes,
                        fontsize=9, va='top')

            ax.set_xlabel('mtanh peakedness' if row == 2 else '', fontsize=9)
            ax.set_ylabel(f'{diag} NN peakedness' if col == 0 else '', fontsize=9)
            ax.set_title(f'{diag} — {METHOD_SPECS[method]["label"]}', fontsize=10)
            ax.grid(True, alpha=0.3)

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                          markersize=8, label=str(s))
               for s, c in shot_colors.items()]
    fig.legend(handles, [str(s) for s in shot_colors], loc='lower center',
               ncol=4, fontsize=10, title='Shot', bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    path = os.path.join(FIG_DIR, 'fig5_peakedness_scatter.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  fig5_peakedness_scatter.png saved')


# ================================================================
# Figure 6: MAE 箱线图
# ================================================================

def fig_mae_boxplot(metrics_rows):
    """3×1 subplots: 每个诊断下 4 个 NN 方法的 MAE 箱线图。"""
    if not metrics_rows:
        print('  No metrics data, skipping fig6')
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 14), dpi=300)
    fig.suptitle('MAE Distribution Across All Test Profiles (vs mtanh)',
                 fontsize=13, fontweight='bold')

    colors = [METHOD_SPECS[m]['color'] for m in NN_METHODS]
    labels = [METHOD_SPECS[m]['label'] for m in NN_METHODS]

    for row, diag in enumerate(DIAGNOSTICS):
        ax = axes[row]
        data_groups = []

        for method in NN_METHODS:
            vals = [float(r['MAE']) for r in metrics_rows
                    if r['diagnostic'] == diag and r['method'] == method]
            data_groups.append(vals)

        bp = ax.boxplot(data_groups, labels=labels, patch_artist=True,
                        widths=0.5, showfliers=True)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        # jittered scatter overlay
        for i, vals in enumerate(data_groups):
            if vals:
                jitter = np.random.normal(0, 0.04, len(vals))
                ax.scatter(np.full(len(vals), i + 1) + jitter, vals,
                           c=colors[i], alpha=0.4, s=15, edgecolors='none', zorder=3)

        ax.set_ylabel(DIAGNOSTIC_SPECS[diag]['ylabel'], fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='gray', lw=0.5)

        # 标注均值
        for i, vals in enumerate(data_groups):
            if vals:
                mean_val = np.mean(vals)
                ax.annotate(f'{mean_val:.3f}', xy=(i + 1, mean_val),
                            fontsize=8, ha='center', va='bottom',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(FIG_DIR, 'fig6_mae_boxplot.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  fig6_mae_boxplot.png saved')


# ================================================================
# Figure 8: 架构对比表
# ================================================================

def fig_architecture_table():
    """渲染架构对比表为 matplotlib table。"""
    col_labels = ['Model', 'Parameters', 'Te MAE', 'ne MAE', 'Ti MAE',
                  'Inference (ms)', 'Architecture']
    rows = []
    for model, params, te_mae, ne_mae, ti_mae, arch in ARCHITECTURE_TABLE:
        rows.append([model, f'{params:,}', f'{te_mae:.3f}', f'{ne_mae:.3f}',
                     f'{ti_mae:.3f}', 'N/A', arch])

    fig, ax = plt.subplots(figsize=(14, 3.5), dpi=200)
    ax.axis('off')
    table = ax.table(cellText=rows, colLabels=col_labels, loc='center',
                     cellLoc='center', colColours=['#E8E8E8'] * 7)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    ax.set_title('Architecture Comparison (Validation MAE from CLAUDE.md)',
                 fontsize=12, fontweight='bold', pad=20)

    path = os.path.join(FIG_DIR, 'fig8_architecture_table.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    # 同时导出 CSV
    csv_path = os.path.join(FIG_DIR, 'architecture_table.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(col_labels)
        for model, params, te_mae, ne_mae, ti_mae, arch in ARCHITECTURE_TABLE:
            writer.writerow([model, params, te_mae, ne_mae, ti_mae, '', arch])
    print(f'  fig8_architecture_table.png + csv saved')


# ================================================================
# Figure 10: 时序连续性
# ================================================================

def fig_temporal(metas):
    """选取数据最完整的炮号，绘制 core value vs time。"""
    # 选 TS 时间点最多的炮号
    shot_counts = {}
    for (shot, td), meta in metas.items():
        shot_counts[shot] = shot_counts.get(shot, 0) + 1
    if not shot_counts:
        print('  No metadata, skipping fig10')
        return

    best_shot = max(shot_counts, key=shot_counts.get)
    best_tds = sorted(
        [(td, meta) for (s, td), meta in metas.items() if s == best_shot],
        key=lambda x: x[1].get('time', 0)
    )

    fig, axes = plt.subplots(3, 1, figsize=(12, 14), dpi=300)
    fig.suptitle(f'Temporal Evolution — Shot {best_shot} ({len(best_tds)} time points)',
                 fontsize=13, fontweight='bold')

    for row, diag in enumerate(DIAGNOSTICS):
        ax = axes[row]
        spec = DIAGNOSTIC_SPECS[diag]

        times = []
        for td, meta in best_tds:
            t = meta.get('time')
            times.append(t) if t is not None else times.append(0)

        for method in ALL_METHODS:
            ms = METHOD_SPECS[method]
            core_vals = []
            valid_times = []
            for td, meta in best_tds:
                x, y = load_profile(best_shot, td, method, diag)
                if x is not None and y is not None:
                    core_vals.append(float(y[0]))
                    valid_times.append(meta.get('time', 0))

            if core_vals:
                ax.plot(valid_times, core_vals, color=ms['color'], ls=ms['ls'],
                        lw=ms['lw'], marker='o', markersize=4,
                        label=ms['label'], alpha=0.85)

        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel(spec['ylabel'], fontsize=10)
        ax.set_title(f'{diag} core value', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, ncol=3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(FIG_DIR, 'fig10_temporal.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  fig10_temporal.png saved (shot {best_shot})')


# ================================================================
# 主入口
# ================================================================

def main():
    print(f'{"="*60}')
    print(f'Paper Figure Generation — Phase 2')
    print(f'Input: {RESULT_BASE}/')
    print(f'Output: {FIG_DIR}/')
    print(f'{"="*60}\n')

    # 加载数据
    metas = load_npz_metadata()
    print(f'Loaded {len(metas)} metadata entries')

    metrics_rows = load_metrics()
    print(f'Loaded {len(metrics_rows)} metric rows\n')

    if not metas:
        print('ERROR: No metadata found. Run run_all_fits.py on server first.')
        sys.exit(1)

    # 生成图表
    print('Generating figures...')
    fig_hmode_overlay(metas)
    fig_lmode_overlay(metas)
    fig_pedestal_zoom(metas)
    fig_peakedness_scatter(metrics_rows)
    fig_mae_boxplot(metrics_rows)
    fig_architecture_table()
    fig_temporal(metas)

    print(f'\n{"="*60}')
    print(f'All figures saved to: {FIG_DIR}/')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
