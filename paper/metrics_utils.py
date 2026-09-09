# -*- coding: utf-8 -*-
"""
metrics_utils.py — 论文指标计算与绘图配置共享模块

被 run_all_fits.py / compute_all_metrics.py / generate_paper_figures.py 共同引用。
"""

import numpy as np


# ================================================================
# 指标计算
# ================================================================

def compute_metrics(ref: np.ndarray, pred: np.ndarray,
                    ref_rho: np.ndarray, pred_rho: np.ndarray):
    """计算 pred 相对 ref 的偏差指标。

    复现自 compare_profiles.py:240-265。若两个 rho 网格不一致，
    先将 pred 插值到 ref 网格再做逐点比较。
    """
    from scipy.interpolate import interp1d

    if not np.allclose(ref_rho, pred_rho, rtol=1e-4):
        interp = interp1d(pred_rho, pred, kind='linear',
                          bounds_error=False, fill_value='extrapolate')
        pred = interp(ref_rho)

    diff = pred - ref
    # 物理阈值：防止边缘近零值导致除法爆炸
    # 1e-3 keV = 1 eV（远低于等离子体边界温度）
    # 1e-3 * 1e13 cm^-3 = 1e10 cm^-3（远低于边界密度）
    eps = 1e-3
    rel_diff = np.where(np.abs(ref) > eps, diff / ref * 100, 0.0)
    # 裁剪异常相对误差
    rel_diff = np.clip(rel_diff, -1000.0, 1000.0)

    return {
        'MAE': float(np.mean(np.abs(diff))),
        'RMSE': float(np.sqrt(np.mean(diff ** 2))),
        'MaxAE': float(np.max(np.abs(diff))),
        'MeanRel%': float(np.mean(np.abs(rel_diff))),
        'MaxRel%': float(np.max(np.abs(rel_diff))),
    }


def compute_peakedness(y: np.ndarray, edge_avg: int = 10):
    """芯部-边缘比值 peakedness = core / edge。

    论文中衡量剖面陡峭度的指标。core = y[0]（磁轴值），
    edge = 最后 edge_avg 个点的均值（抑制单点噪声）。
    """
    core = float(y[0])
    n_edge = min(edge_avg, len(y))
    edge = float(np.mean(y[-n_edge:])) if n_edge > 0 else float(y[-1])
    return core / max(edge, 1e-10)  # 防零除


def compute_core_bias(y_nn: np.ndarray, y_mtanh: np.ndarray):
    """芯部值相对偏差 (%)。"""
    core_nn = float(y_nn[0])
    core_mtanh = float(y_mtanh[0])
    return (core_nn - core_mtanh) / max(abs(core_mtanh), 1e-10) * 100


# ================================================================
# 方法规格（绘图用）
# ================================================================

METHOD_SPECS = {
    'mtanh': {
        'label': 'mtanh (baseline)',
        'color': '#2ECC40',   # 绿色
        'ls': '--', 'lw': 2.5,
    },
    'nn': {
        'label': 'ProfileNet',
        'color': '#E63946',   # 红色
        'ls': '-', 'lw': 2.0,
    },
    'lstm': {
        'label': 'LSTM',
        'color': '#2A9D8F',   # 青色
        'ls': '-', 'lw': 2.0,
    },
    'cnn': {
        'label': 'CNN-1D',
        'color': '#457B9D',   # 蓝色
        'ls': '-', 'lw': 2.0,
    },
    'cnn_deeponet': {
        'label': 'CNN-DeepONet',
        'color': '#8E44AD',   # 紫色
        'ls': '-', 'lw': 2.0,
    },
    'transformer': {
        'label': 'Transformer',
        'color': '#F4A261',   # 橙色
        'ls': '-', 'lw': 2.0,
    },
}

# NN 方法列表（不含 mtanh）
NN_METHODS = ['nn', 'lstm', 'cnn', 'cnn_deeponet', 'transformer']
ALL_METHODS = ['mtanh'] + NN_METHODS

# ================================================================
# 诊断规格（绘图用）
# ================================================================

DIAGNOSTIC_SPECS = {
    'Te': {
        'ylabel': 'Te (keV)',
        'unit': 'keV',
        'inone_name': 'TEIN',
    },
    'ne': {
        'ylabel': 'ne (10$^{19}$ m$^{-3}$)',
        'unit': '1e19 m^-3',
        'inone_name': 'ENEIN',
    },
    'Ti': {
        'ylabel': 'Ti (keV)',
        'unit': 'keV',
        'inone_name': 'TIIN',
    },
}

DIAGNOSTICS = ['Te', 'ne', 'Ti']

# ================================================================
# 架构对比表数据（来自 CLAUDE.md）
# ================================================================

# === 修改版：扩充测试集（5 炮 24 时间点：8H/10L/6unk，Te/ne n=24、Ti n=20）===
# 由 make_latex_tables.py 从 paper_results/all_metrics.csv 生成
# 注：Params 为 Te 模型参数量（实测于 profile_nn_models/*_Te.pt 的 state_dict）
ARCHITECTURE_TABLE = [
    # (Model, Params, Te_MAE, ne_MAE, Ti_MAE, Architecture)
    ('ProfileNet',    67000, 0.512, 0.057, 0.091, 'SetEncoder + CoordDecoder'),
    ('LSTM',          76000, 0.347, 0.060, 0.074, 'BiLSTM + CoordDecoder'),
    ('CNN-1D',        66000, 0.342, 0.067, 0.141, 'Pure ResNet (no DeepONet)'),
    ('CNN-DeepONet',  79000, 0.393, 0.075, 0.082, 'CNN encoder + CoordDecoder'),
    ('Transformer',  136000, 0.350, 0.112, 0.089, 'TransformerEncoder + CoordDecoder'),
]

# V5 H-mode 测试集指标（8 时间点）
ARCHITECTURE_TABLE_H = [
    ('ProfileNet',    67000, 0.639, 0.058, 0.097, 'SetEncoder + CoordDecoder'),
    ('LSTM',          76000, 0.525, 0.065, 0.075, 'BiLSTM + CoordDecoder'),
    ('CNN-1D',        66000, 0.660, 0.067, 0.110, 'Pure ResNet (no DeepONet)'),
    ('CNN-DeepONet',  79000, 0.755, 0.073, 0.076, 'CNN encoder + CoordDecoder'),
    ('Transformer',  136000, 0.694, 0.120, 0.090, 'TransformerEncoder + CoordDecoder'),
]

# V5 L-mode 测试集指标（10 时间点，排除 unknown）
ARCHITECTURE_TABLE_L = [
    ('ProfileNet',    67000, 0.298, 0.058, 0.073, 'SetEncoder + CoordDecoder'),
    ('LSTM',          76000, 0.224, 0.063, 0.046, 'BiLSTM + CoordDecoder'),
    ('CNN-1D',        66000, 0.206, 0.068, 0.150, 'Pure ResNet (no DeepONet)'),
    ('CNN-DeepONet',  79000, 0.199, 0.071, 0.054, 'CNN encoder + CoordDecoder'),
    ('Transformer',  136000, 0.185, 0.110, 0.041, 'TransformerEncoder + CoordDecoder'),
]

# === 原始版本（保留参考）===
# # 4 炮号测试集
# TEST_SHOTS = [156005, 156010, 156100, 156400]

# === 修改版：扩充测试集，加入 156900（156200 因 EFIT 覆盖不足整体排除）===
# 5 炮号测试集
TEST_SHOTS = [156005, 156010, 156100, 156400, 156900]
