# -*- coding: utf-8 -*-
"""
benchmark_speed.py — NN 推理速度基准测试

对 4 架构 × 3 诊断测量单次推理耗时（不含模型加载），输出 CSV。

用法:
  pytorch_run paper/benchmark_speed.py          # 服务器
  python paper/benchmark_speed.py               # 本地（有 torch 即可）

输出:
  paper_results/figures/inference_speed.csv
"""

import numpy as np
import os
import sys
import csv
import time
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
MODEL_DIR = os.path.join(SCRIPT_DIR, 'profile_nn_models')
FIG_DIR = os.path.join(SCRIPT_DIR, 'paper_results', 'figures')

N_WARMUP = 10
N_BENCH = 200

# ================================================================
# 生成典型诊断散点（匹配 EAST TS 数据分布）
# ================================================================

def make_test_input(diag='Te'):
    """生成模拟的 EAST 诊断散点数据，放入 GPU（若可用）"""
    rng = np.random.RandomState(42)
    if diag == 'Te':
        n_pts = 25
        rho = np.sort(rng.uniform(0.01, 1.0, n_pts))
        val = 3000 * np.exp(-3.0 * rho) + 100 * (1 - rho)  # eV, 3 keV 芯部
    elif diag == 'ne':
        n_pts = 20
        rho = np.sort(rng.uniform(0.01, 1.0, n_pts))
        val = 2.5 * np.exp(-2.0 * rho) + 0.3 * (1 - rho)  # 1e19 m^-3
    else:  # Ti
        n_pts = 15
        rho = np.sort(rng.uniform(0.05, 0.85, n_pts))
        val = 1.5 * np.exp(-2.5 * rho) + 0.1 * (1 - rho)  # keV
    return rho.astype(np.float32), val.astype(np.float32)


# ================================================================
# 加载模型
# ================================================================

NN_FULL = {}

# ProfileNet
try:
    from profile_nn.infer import nn_fit_te, nn_fit_ne, nn_fit_ti, set_model_dir
    set_model_dir(MODEL_DIR)
    NN_FULL['ProfileNet'] = {'te': nn_fit_te, 'ne': nn_fit_ne, 'ti': nn_fit_ti}
    print('✓ ProfileNet')
except Exception as e:
    print(f'✗ ProfileNet: {e}')

# LSTM
try:
    from profile_nn.infer_lstm import (nn_fit_te_lstm, nn_fit_ne_lstm,
                                       nn_fit_ti_lstm, set_model_dir_lstm)
    set_model_dir_lstm(MODEL_DIR)
    NN_FULL['LSTM'] = {'te': nn_fit_te_lstm, 'ne': nn_fit_ne_lstm, 'ti': nn_fit_ti_lstm}
    print('✓ LSTM')
except Exception as e:
    print(f'✗ LSTM: {e}')

# CNN-1D
try:
    from profile_nn.infer_cnn import (nn_fit_te_cnn, nn_fit_ne_cnn,
                                      nn_fit_ti_cnn, set_model_dir_cnn)
    set_model_dir_cnn(MODEL_DIR)
    NN_FULL['CNN-1D'] = {'te': nn_fit_te_cnn, 'ne': nn_fit_ne_cnn, 'ti': nn_fit_ti_cnn}
    print('✓ CNN-1D')
except Exception as e:
    print(f'✗ CNN-1D: {e}')

# Transformer
try:
    from profile_nn.infer_transformer import (nn_fit_te_transformer, nn_fit_ne_transformer,
                                              nn_fit_ti_transformer, set_model_dir_transformer)
    set_model_dir_transformer(MODEL_DIR)
    NN_FULL['Transformer'] = {'te': nn_fit_te_transformer, 'ne': nn_fit_ne_transformer,
                              'ti': nn_fit_ti_transformer}
    print('✓ Transformer')
except Exception as e:
    print(f'✗ Transformer: {e}')


# ================================================================
# 基准测试
# ================================================================

def benchmark_one(label, func, rho, val, extra_args=()):
    """预热后计时 N_BENCH 次，返回 (mean_ms, std_ms)。"""
    # 预热
    for _ in range(N_WARMUP):
        func(rho.copy(), val.copy(), *extra_args)

    times = []
    for _ in range(N_BENCH):
        t0 = time.perf_counter()
        func(rho.copy(), val.copy(), *extra_args)
        times.append((time.perf_counter() - t0) * 1000.0)  # ms

    return np.mean(times), np.std(times)


def main():
    print(f'Models loaded: {list(NN_FULL.keys())}')
    print(f'Warmup: {N_WARMUP}, Bench: {N_BENCH} iterations\n')

    # 预生成 Ti 共用输入
    ti_rho, ti_val = make_test_input('Ti')
    _, te_for_ped = make_test_input('Te')
    te_ped_rho = np.array([0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.0], dtype=np.float32)
    te_ped_val = te_for_ped[-7:].astype(np.float32) * 0.001  # eV → keV

    rows = []
    print(f'{"Method":15s} {"Diag":5s} {"Mean(ms)":>10s} {"Std(ms)":>10s}')
    print('-' * 42)

    for method_name, fitters in NN_FULL.items():
        for diag in ['Te', 'ne', 'Ti']:
            rho, val = make_test_input(diag)
            extra = ()
            if diag == 'Ti':
                rho, val = ti_rho, ti_val
                extra = (te_ped_rho, te_ped_val)

            try:
                mean_ms, std_ms = benchmark_one(
                    method_name, fitters[diag.lower()], rho, val, extra
                )
            except Exception as e:
                mean_ms, std_ms = float('nan'), float('nan')
                print(f'  {method_name:15s} {diag:5s} ERROR: {e}')

            print(f'  {method_name:15s} {diag:5s} {mean_ms:10.3f} {std_ms:10.3f}')
            rows.append([method_name, diag, f'{mean_ms:.3f}', f'{std_ms:.3f}'])

    # 保存 CSV
    csv_path = os.path.join(FIG_DIR, 'inference_speed.csv')
    os.makedirs(FIG_DIR, exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Model', 'Diagnostic', 'Mean_ms', 'Std_ms'])
        w.writerows(rows)
    print(f'\nSaved: {csv_path}')


if __name__ == '__main__':
    main()
