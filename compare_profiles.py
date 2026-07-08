# -*- coding: utf-8 -*-
"""
compare_profiles.py — 对比五种方法的 TEIN / ENEIN / TIIN 剖面

将同一炮号同一时间点下 mtanh / ProfileNet / LSTM / CNN-1D / Transformer 五种方法
输出的 ONETWO Namelist 剖面提取出来, 绘制在同一张图上对比,
并计算各 NN 方法相对于 mtanh 基线的定量偏差。

使用方法:
    # 单时间点模式
    python compare_profiles.py <shot> <time>
    python compare_profiles.py 81481 5.3

    # 交互式单时间点
    python compare_profiles.py

    # 批量模式：自动扫描该炮所有时间点，有 inone 文件就对比
    python compare_profiles.py 81481 --batch

    # 批量 + 只生成 CSV 总表（跳过逐时间点 PNGB）
    python compare_profiles.py 81481 --batch --csv-only

输出:
    单时间点:  results/{shot}/{time}/compare/
                ├── compare_TEIN.png
                ├── compare_ENEIN.png
                ├── compare_TIIN.png
                └── compare_metrics.txt

    批量模式:  results/{shot}/{time}/compare/   (同上，每个有时间点一份)
              results/{shot}/compare_summary.csv  (全部时间点汇总表)

前提条件:
    该炮号在该时间点下已经跑过至少一个 ONETWO pipeline:
      results/{shot}/{time}/onetwo/       → mtanh (原始)
      results/{shot}/{time}/onetwo_nn/    → ProfileNet
      results/{shot}/{time}/onetwo_lstm/  → LSTM
      results/{shot}/{time}/onetwo_cnn/          → CNN-1D
      results/{shot}/{time}/onetwo_transformer/  → Transformer
"""

import numpy as np
from Namelist3 import Namelist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import csv
from glob import glob


# ================================================================
# 配置
# ================================================================

def _result_dir():
    """自动检测结果目录：优先 results/，回退到 result/"""
    for d in ['results', 'result']:
        if os.path.isdir(d):
            return d
    return 'results'  # 默认（服务器上）

RESULT_BASE = _result_dir()


METHOD_SPECS = {
    'mtanh': {
        'subdir': 'onetwo',
        'label': 'mtanh (baseline)',
        'color': 'black',
        'ls': '--', 'lw': 2.5,
        'marker': None,
    },
    'ProfileNet': {
        'subdir': 'onetwo_nn',
        'label': 'ProfileNet',
        'color': '#E63946',
        'ls': '-', 'lw': 2,
        'marker': None,
    },
    'LSTM': {
        'subdir': 'onetwo_lstm',
        'label': 'LSTM',
        'color': '#2A9D8F',
        'ls': '-', 'lw': 2,
        'marker': None,
    },
    'CNN': {
        'subdir': 'onetwo_cnn',
        'label': 'CNN-1D',
        'color': '#457B9D',
        'ls': '-', 'lw': 2,
        'marker': None,
    },
    'Transformer': {
        'subdir': 'onetwo_transformer',
        'label': 'Transformer',
        'color': '#F4A261',
        'ls': '-', 'lw': 2,
        'marker': None,
    },
}

CSV_COLUMNS = [
    'time_dir', 'time_parsed',
    'mtanh_available', 'ProfileNet_available', 'LSTM_available', 'CNN_available', 'Transformer_available',
    'TEIN_ProfileNet_MAE', 'TEIN_ProfileNet_RMSE', 'TEIN_ProfileNet_MaxAE',
    'TEIN_LSTM_MAE', 'TEIN_LSTM_RMSE', 'TEIN_LSTM_MaxAE',
    'TEIN_CNN_MAE', 'TEIN_CNN_RMSE', 'TEIN_CNN_MaxAE',
    'TEIN_Transformer_MAE', 'TEIN_Transformer_RMSE', 'TEIN_Transformer_MaxAE',
    'ENEIN_ProfileNet_MAE', 'ENEIN_ProfileNet_RMSE', 'ENEIN_ProfileNet_MaxAE',
    'ENEIN_LSTM_MAE', 'ENEIN_LSTM_RMSE', 'ENEIN_LSTM_MaxAE',
    'ENEIN_CNN_MAE', 'ENEIN_CNN_RMSE', 'ENEIN_CNN_MaxAE',
    'ENEIN_Transformer_MAE', 'ENEIN_Transformer_RMSE', 'ENEIN_Transformer_MaxAE',
    'TIIN_ProfileNet_MAE', 'TIIN_ProfileNet_RMSE', 'TIIN_ProfileNet_MaxAE',
    'TIIN_LSTM_MAE', 'TIIN_LSTM_RMSE', 'TIIN_LSTM_MaxAE',
    'TIIN_CNN_MAE', 'TIIN_CNN_RMSE', 'TIIN_CNN_MaxAE',
    'TIIN_Transformer_MAE', 'TIIN_Transformer_RMSE', 'TIIN_Transformer_MaxAE',
]


# ================================================================
# 工具函数
# ================================================================

def parse_time_from_dir(dirname: str):
    """从时间目录名解析浮点时间，如 '003.50000s' → 3.5"""
    try:
        t_str = dirname.replace('s', '').lstrip('0')
        if t_str == '' or t_str == '.':
            return 0.0
        return float(t_str)
    except ValueError:
        return None


def find_time_dir(shot: int, time: float) -> str:
    """在 results/{shot}/ 下找到与给定时间最接近的目录"""
    results_shot = f'{RESULT_BASE}/{shot}'
    if not os.path.isdir(results_shot):
        raise FileNotFoundError(f'未找到 {RESULT_BASE}/{shot}/ 目录，请先跑 ONETWO pipeline')

    dirs = [d for d in os.listdir(results_shot)
            if os.path.isdir(os.path.join(results_shot, d)) and d.endswith('s')]

    if not dirs:
        raise FileNotFoundError(f'{RESULT_BASE}/{shot}/ 下没有时间点目录')

    parsed = []
    for d in dirs:
        t = parse_time_from_dir(d)
        if t is not None:
            parsed.append((t, d))

    if not parsed:
        raise ValueError(f'无法解析任何时间点目录')

    parsed.sort(key=lambda x: abs(x[0] - time))
    matched_t, matched_dir = parsed[0]
    print(f'目标时间 {time:.5f}s → 匹配目录 {matched_dir} (t={matched_t:.5f}s)')
    return matched_dir


def discover_time_dirs_with_inone(shot: int):
    """扫描 results/{shot}/ 下所有时间点目录，返回含有至少一个 inone 文件的目录列表。

    Returns:
        list of (time_dir_name, parsed_time, available_methods_set)
        按时间升序排列
    """
    results_shot = f'{RESULT_BASE}/{shot}'
    if not os.path.isdir(results_shot):
        print(f'未找到 {RESULT_BASE}/{shot}/ 目录')
        return []

    all_dirs = [d for d in os.listdir(results_shot)
                if os.path.isdir(os.path.join(results_shot, d)) and d.endswith('s')]

    qualified = []
    for d in all_dirs:
        t = parse_time_from_dir(d)
        if t is None:
            continue

        # 检查哪些方法的 inone 文件存在
        available = set()
        for method_key, spec in METHOD_SPECS.items():
            inone_path = os.path.join(results_shot, d, spec['subdir'], 'inone')
            if os.path.isfile(inone_path):
                available.add(method_key)

        if available:
            qualified.append((d, t, available))

    qualified.sort(key=lambda x: x[1])
    return qualified


def read_namelist_profile(inone_path: str, normalize_tein_kev: bool = True):
    """从 inone 文件读取 TEIN / ENEIN / TIIN 剖面

    TEIN 单位自动归一化：mtanh 版（onetwo_output.py）写入 keV，
    NN 版写入 eV（×1000）。脚本自动检测并将 mtanh 的 keV 值统一转为 eV，
    确保对比在同一单位下进行。

    Returns:
        dict with keys 'RTEIN','TEIN','RENEIN','ENEIN','RTIIN','TIIN'
        每个元素是 np.ndarray; 如果某个剖面不存在则对应值为 None
    """
    if not os.path.isfile(inone_path):
        return None

    obj = Namelist()
    obj.read(inone_path)

    result = {}
    for key in ['RTEIN', 'TEIN', 'RENEIN', 'ENEIN', 'RTIIN', 'TIIN']:
        try:
            val = obj['NAMELIS1'][key]
            result[key] = np.array(val, dtype=float)
        except (KeyError, TypeError):
            result[key] = None

    # ---- TEIN 单位归一化 ----
    # mtanh (onetwo_output.py) 错误写入 keV，NN 版写入 eV。
    # 检测方法：若 TEIN max < 50，极可能是 keV（正常 Te < 10 keV）
    if normalize_tein_kev and result.get('TEIN') is not None:
        tein = result['TEIN']
        if tein.max() < 50.0:
            # 来自 mtanh，keV → eV
            result['TEIN'] = tein * 1000.0
            result['_tein_normalized'] = True
        else:
            result['_tein_normalized'] = False

    return result


def compute_metrics(ref: np.ndarray, pred: np.ndarray,
                    ref_rho: np.ndarray, pred_rho: np.ndarray):
    """计算 pred 相对 ref 的偏差指标。"""
    from scipy.interpolate import interp1d

    if not np.allclose(ref_rho, pred_rho, rtol=1e-4):
        interp = interp1d(pred_rho, pred, kind='linear',
                          bounds_error=False, fill_value='extrapolate')
        pred = interp(ref_rho)

    diff = pred - ref
    # 使用物理上有意义的阈值防止边缘近零值导致除法爆炸
    # 1e-3 keV = 1 eV（远低于任何物理等离子体边界温度）
    # 1e-3 * 1e13 cm^-3 = 1e10 cm^-3（远低于边界密度）
    eps = 1e-3
    rel_diff = np.where(np.abs(ref) > eps, diff / ref * 100, 0.0)
    # 裁剪异常相对误差，防止单个边缘点主导 MeanRel 指标
    rel_diff = np.clip(rel_diff, -1000.0, 1000.0)

    return {
        'MAE': float(np.mean(np.abs(diff))),
        'RMSE': float(np.sqrt(np.mean(diff ** 2))),
        'MaxAE': float(np.max(np.abs(diff))),
        'MeanRel%': float(np.mean(np.abs(rel_diff))),
        'MaxRel%': float(np.max(np.abs(rel_diff))),
    }


def plot_comparison(shot: int, time_label: str, profiles: dict,
                    kind: str, output_dir: str):
    """绘制四种方法在同一坐标轴上的对比图"""
    rho_key, val_key = f'R{kind}', kind

    unit_map = {
        'TEIN': ('Electron Temperature', 'eV'),
        'ENEIN': ('Electron Density', 'cm⁻³'),
        'TIIN': ('Ion Temperature', 'keV'),
    }
    title_base, unit = unit_map.get(kind, (kind, ''))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                             gridspec_kw={'width_ratios': [3, 2]})
    ax_main = axes[0]
    ax_diff = axes[1]

    # ---- 基线 (mtanh) ----
    baseline = profiles.get('mtanh')
    if baseline is not None:
        r = baseline.get(rho_key)
        v = baseline.get(val_key)
        if r is not None and v is not None:
            ax_main.plot(r, v, color='black', ls='--', lw=2.5,
                         label='mtanh (baseline)')
            ref_r, ref_v = r, v
        else:
            ref_r, ref_v = None, None
    else:
        ref_r, ref_v = None, None

    # ---- 四种 NN 方法 ----
    metrics_lines = []
    metrics_dict = {}
    for method_key in ['ProfileNet', 'LSTM', 'CNN', 'Transformer']:
        spec = METHOD_SPECS[method_key]
        pdata = profiles.get(method_key)
        if pdata is None:
            print(f'  [!] {method_key}: 找不到 inone 文件，跳过')
            continue

        r = pdata.get(rho_key)
        v = pdata.get(val_key)
        if r is None or v is None:
            print(f'  [!] {method_key}: {kind} 数据缺失，跳过')
            continue

        ax_main.plot(r, v, color=spec['color'], ls=spec['ls'], lw=spec['lw'],
                     label=spec['label'])

        if ref_r is not None and ref_v is not None:
            from scipy.interpolate import interp1d
            interp = interp1d(r, v, kind='linear',
                              bounds_error=False, fill_value='extrapolate')
            v_on_ref = interp(ref_r)
            diff = v_on_ref - ref_v
            ax_diff.plot(ref_r, diff, color=spec['color'], lw=1.5,
                         label=spec['label'])

            metrics = compute_metrics(ref_v, v, ref_r, r)
            metrics_dict[method_key] = metrics
            metrics_lines.append(
                f"  {spec['label']:15s}  MAE={metrics['MAE']:.4f}  "
                f"RMSE={metrics['RMSE']:.4f}  MaxAE={metrics['MaxAE']:.4f}  "
                f"MeanRel={metrics['MeanRel%']:.2f}%"
            )

    # ---- 美化 ----
    ax_main.set_xlabel(r'$\rho$ (normalized poloidal flux)', fontsize=12)
    ax_main.set_ylabel(f'{title_base} [{unit}]', fontsize=12)
    ax_main.set_title(f'Shot {shot}  {time_label}  —  {title_base} Profile Comparison',
                      fontsize=13)
    ax_main.legend(fontsize=10, loc='upper right')
    ax_main.set_xlim([0, 1])
    ax_main.grid(True, alpha=0.3)
    ax_main.set_ylim(bottom=0)

    ax_diff.axhline(y=0, color='black', ls='--', lw=1, alpha=0.5)
    ax_diff.set_xlabel(r'$\rho$', fontsize=12)
    if ref_r is not None and ref_v is not None:
        ax_diff.set_ylabel(f'Δ {title_base} [{unit}] (vs mtanh)', fontsize=12)
        ax_diff.set_title('Deviation from mtanh Baseline', fontsize=12)
        ax_diff.legend(fontsize=9, loc='upper right')
    else:
        ax_diff.text(0.5, 0.5, 'No mtanh baseline\n— deviation not computed',
                     transform=ax_diff.transAxes, ha='center', va='center',
                     fontsize=13, color='gray', style='italic')
        ax_diff.set_title('Deviation (unavailable)', fontsize=12)
    ax_diff.set_xlim([0, 1])
    ax_diff.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'compare_{kind}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [OK] 已保存: {save_path}')

    return metrics_lines, metrics_dict


def compare_single_time(shot: int, time_dir: str):
    """对单个时间点执行完整对比流程。

    Returns:
        profiles dict, 或 None（如果没有任何 inone 文件）
    """
    output_base = os.path.join(RESULT_BASE, str(shot), time_dir)
    compare_dir = os.path.join(output_base, 'compare')
    os.makedirs(compare_dir, exist_ok=True)

    # 读取所有方法
    profiles = {}
    for method_key, spec in METHOD_SPECS.items():
        inone_path = os.path.join(output_base, spec['subdir'], 'inone')
        profiles[method_key] = read_namelist_profile(inone_path)

    # 检查是否至少有一个有效剖面
    has_any = any(p is not None for p in profiles.values())
    if not has_any:
        return None

    available = [k for k, v in profiles.items() if v is not None]
    print(f'  可用方法: {", ".join(METHOD_SPECS[k]["label"] for k in available)}')

    # 逐项对比
    all_metrics = {}
    for kind in ['TEIN', 'ENEIN', 'TIIN']:
        lines, mdict = plot_comparison(shot, time_dir, profiles, kind, compare_dir)
        all_metrics[kind] = (lines, mdict)

    # 写入定量指标
    metrics_path = os.path.join(compare_dir, 'compare_metrics.txt')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write(f'Profile Comparison Metrics\n')
        f.write(f'Shot: {shot}  Time: {time_dir}\n')
        f.write(f'Baseline: mtanh (original fitting)\n')
        f.write(f'{"="*72}\n\n')
        for kind, (lines, _mdict) in all_metrics.items():
            f.write(f'[{kind}]\n')
            if lines:
                for line in lines:
                    f.write(line + '\n')
            else:
                f.write('  (no comparison data available)\n')
            f.write('\n')

    return profiles, all_metrics


def build_summary_csv(shot: int, results: list):
    """根据所有时间点的对比结果生成汇总 CSV。

    Args:
        results: list of (time_dir, parsed_time, available_set, all_metrics)
                 其中 all_metrics = {kind: (lines, mdict)}
    """
    csv_path = os.path.join(RESULT_BASE, str(shot), 'compare_summary.csv')

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction='ignore')
        writer.writeheader()

        for time_dir, parsed_t, available, all_metrics in results:
            row = {
                'time_dir': time_dir,
                'time_parsed': f'{parsed_t:.5f}',
                'mtanh_available': 1 if 'mtanh' in available else 0,
                'ProfileNet_available': 1 if 'ProfileNet' in available else 0,
                'LSTM_available': 1 if 'LSTM' in available else 0,
                'CNN_available': 1 if 'CNN' in available else 0,
                'Transformer_available': 1 if 'Transformer' in available else 0,
            }

            for kind in ['TEIN', 'ENEIN', 'TIIN']:
                if all_metrics is None:
                    continue
                kind_data = all_metrics.get(kind)
                if kind_data is None:
                    continue
                _lines, mdict = kind_data
                for nn in ['ProfileNet', 'LSTM', 'CNN', 'Transformer']:
                    m = mdict.get(nn, {})
                    row[f'{kind}_{nn}_MAE'] = m.get('MAE', '')
                    row[f'{kind}_{nn}_RMSE'] = m.get('RMSE', '')
                    row[f'{kind}_{nn}_MaxAE'] = m.get('MaxAE', '')

            writer.writerow(row)

    print(f'\n[OK] 汇总 CSV 已保存: {csv_path}')
    return csv_path


def print_batch_summary(shot: int, results: list):
    """打印批量对比的摘要统计"""
    total = len(results)
    has_mtanh = sum(1 for _, _, avail, _ in results if 'mtanh' in avail)

    print(f'\n{"="*70}')
    print(f'Shot {shot} 批量对比完成')
    print(f'  扫描时间点总数:     {len(discover_time_dirs_with_inone(shot))}')
    print(f'  成功对比时间点数:   {total}')
    print(f'  其中含 mtanh 基线:  {has_mtanh}/{total}')

    # 各方法的平均 MAE（仅统计有 mtanh 基线的时间点）
    for kind in ['TEIN', 'ENEIN', 'TIIN']:
        for nn in ['ProfileNet', 'LSTM', 'CNN', 'Transformer']:
            vals = []
            for _, _, _, all_metrics in results:
                if all_metrics is None:
                    continue
                kd = all_metrics.get(kind)
                if kd is None:
                    continue
                _lines, mdict = kd
                m = mdict.get(nn, {})
                if 'MAE' in m:
                    vals.append(m['MAE'])
            if vals:
                print(f'  {kind:6s} {nn:12s}  avg_MAE={np.mean(vals):.4f}  '
                      f'min={np.min(vals):.4f}  max={np.max(vals):.4f}  n={len(vals)}')


# ================================================================
# 主入口
# ================================================================

def main():
    batch_mode = '--batch' in sys.argv
    csv_only = '--csv-only' in sys.argv

    # 清理标志参数
    args = [a for a in sys.argv[1:] if not a.startswith('--')]

    if len(args) >= 2:
        shot = int(args[0])
        time = float(args[1])
    elif len(args) == 1:
        shot = int(args[0])
        if not batch_mode:
            time = float(input('Time (s): '))
    else:
        shot = int(input('Shot number: '))
        if not batch_mode:
            time = float(input('Time (s): '))

    # ================================================================
    # 批量模式
    # ================================================================
    if batch_mode:
        print(f'\n扫描 Shot {shot} 下所有含 inone 文件的时间点...\n')
        qualified = discover_time_dirs_with_inone(shot)

        if not qualified:
            print(f'未找到任何含 inone 文件的时间点。请先跑 ONETWO pipeline。')
            return

        print(f'找到 {len(qualified)} 个时间点有待对比数据:\n')
        for time_dir, parsed_t, available in qualified:
            methods_str = ', '.join(
                METHOD_SPECS[m]['label'] for m in
                ['mtanh', 'ProfileNet', 'LSTM', 'CNN', 'Transformer'] if m in available
            )
            print(f'  {time_dir:20s}  t={parsed_t:.5f}s  [{methods_str}]')

        batch_results = []
        for idx, (time_dir, parsed_t, available) in enumerate(qualified):
            print(f'\n[{idx+1}/{len(qualified)}] {time_dir}  (t={parsed_t:.5f}s)')
            print('-' * 50)
            try:
                result = compare_single_time(shot, time_dir)
                if result is not None:
                    profiles, all_metrics = result
                    batch_results.append((time_dir, parsed_t, available, all_metrics))
                else:
                    print(f'  [!] 跳过：未能读取任何有效剖面')
            except Exception as e:
                print(f'  [X] 失败: {str(e)}')
                continue

        if batch_results:
            # 生成汇总 CSV
            build_summary_csv(shot, batch_results)
            # 打印统计摘要
            print_batch_summary(shot, batch_results)

        return

    # ================================================================
    # 单时间点模式
    # ================================================================
    time_dir = find_time_dir(shot, time)

    print(f'\n读取 {shot} / {time_dir} 下各方法的 inone 文件...\n')
    output_base = os.path.join(RESULT_BASE, str(shot), time_dir)

    profiles = {}
    for method_key, spec in METHOD_SPECS.items():
        inone_path = os.path.join(output_base, spec['subdir'], 'inone')
        pdata = read_namelist_profile(inone_path)
        profiles[method_key] = pdata
        status = '[OK]' if pdata is not None else '[X] (文件不存在)'
        print(f'  {spec["label"]:20s}  {status}')

    result = compare_single_time(shot, time_dir)
    if result is None:
        print('\n[!] 未找到任何有效的 inone 文件，无法对比。')
        return

    profiles, all_metrics = result

    print(f'\n{"="*70}')
    print(f'对比完成 — 输出目录: {RESULT_BASE}/{shot}/{time_dir}/compare/')
    for fname in ['compare_TEIN.png', 'compare_ENEIN.png', 'compare_TIIN.png']:
        fpath = os.path.join(RESULT_BASE, str(shot), time_dir, 'compare', fname)
        if os.path.isfile(fpath):
            print(f'  {fname}')


if __name__ == '__main__':
    main()
