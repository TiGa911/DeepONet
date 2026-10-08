"""CNN-1D Baseline 推理模块：替代 mtanh + 样条拟合法链的即插即用函数。

与 profile_nn/infer.py（ProfileNet）和 infer_lstm.py（LSTM）的接口完全一致。

CNN-1D 架构特点：
  - 散点 → 线性插值到 201 点固定网格
  - 双通道输入：[插值值, 有效掩码]
  - 1D ResNet（残差卷积块）提取多尺度径向特征
  - 端到端输出平滑剖面，无需中间编码-解码

每个函数匹配 onetwo_output.py 中原始处理流程的函数签名。

使用示例：
    from profile_nn.infer_cnn import nn_fit_te_cnn, nn_fit_ne_cnn, nn_fit_ti_cnn, set_model_dir_cnn

    set_model_dir_cnn('./profile_nn_models')

    rho_201, te_201 = nn_fit_te_cnn(rho_ts, te_ev)
    rho_201, ne_201 = nn_fit_ne_cnn(rho_refl, ne_1e19)
    rho_201, ti_201 = nn_fit_ti_cnn(rho_txcs, ti_keV, te_ped_x, te_ped_y)
"""

from profile_nn.cnn_baseline import (
    cnn_fit_te,
    cnn_fit_ne,
    cnn_fit_ti,
    _load_cnn_model,
    scatter_to_grid,
)

# === 原始版本（保留参考）：直接重导出，无后处理 ===
# nn_fit_te_cnn = cnn_fit_te
# nn_fit_ne_cnn = cnn_fit_ne
# nn_fit_ti_cnn = cnn_fit_ti
# _load_model = _load_cnn_model  # alias for internal use

# === 修改版：CNN-1D 输出追加少节点单调 PCHIP 平滑（2026-10-08）===
# 原因：纯卷积解码器的原始输出存在平台状台阶（见论文计划 §十五），
# 与其他架构的 PCHIP 后处理声明不一致；本修改将台阶平滑为单调光滑曲线，
# MAE 变化 < 1%（knots=16 时 33 点均值 0.337→0.337）。
import numpy as _np
from scipy.interpolate import PchipInterpolator as _PchipInterpolator

_KNOTS = 16
_FLOOR = {'Te': 0.005, 'ne': 0.001, 'Ti': 0.005}


def _postprocess_cnn(y, diag):
    """少节点单调 PCHIP 平滑：消除平台台阶，保持单调递减与整体形状。"""
    y = _np.asarray(y, dtype=_np.float64)
    idx = _np.linspace(0, len(y) - 1, _KNOTS).astype(int)
    v = y[idx]
    v = _np.maximum.accumulate(v[::-1])[::-1]  # 后缀最大投影：保证单调递减
    pchip = _PchipInterpolator(idx / (len(y) - 1), v)
    yy = pchip(_np.linspace(0, 1, len(y)))
    return _np.maximum(yy, _FLOOR[diag])


def nn_fit_te_cnn(rho_scattered, te_scattered):
    rho_201, te_201 = cnn_fit_te(rho_scattered, te_scattered)
    return rho_201, _postprocess_cnn(te_201, 'Te')


def nn_fit_ne_cnn(rho_scattered, ne_scattered):
    rho_201, ne_201 = cnn_fit_ne(rho_scattered, ne_scattered)
    return rho_201, _postprocess_cnn(ne_201, 'ne')


def nn_fit_ti_cnn(rho_txcs, ti_keV, te_ped_x=None, te_ped_y=None):
    rho_201, ti_201 = cnn_fit_ti(rho_txcs, ti_keV, te_ped_x, te_ped_y)
    return rho_201, _postprocess_cnn(ti_201, 'Ti')


_load_model = _load_cnn_model  # alias for internal use


def set_model_dir_cnn(path):
    """设置包含训练好的 CNN-1D .pt 模型文件的目录路径。

    调用后缓存被清空，下一次调用会重新加载模型。

    Args:
        path: 模型目录的绝对或相对路径
    """
    import profile_nn.cnn_baseline as _cnn
    _cnn._model_dir_default = path
    _cnn._cache.clear()
