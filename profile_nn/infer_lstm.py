"""LSTM Baseline 推理模块：替代 mtanh + 样条拟合法链的即插即用函数。

与 profile_nn/infer.py（ProfileNet）的接口完全一致，仅底层模型不同。
LSTM（BiLSTM encoder）对径向序列结构更敏感，目前在 Te/ne 上略优于 ProfileNet。

每个函数匹配 onetwo_output.py 中原始处理流程的函数签名：

  原始 Te 流程:                              → nn_fit_te_lstm()
    fitting(x,y,'Te') + robust_interp()

  原始 ne 流程:                              → nn_fit_ne_lstm()
    fitting(x,y,'Refl') + robust_interp()

  原始 Ti 流程:                              → nn_fit_ti_lstm()
    fitting(x,y,'Ti',te_ped_x,te_ped_y) + robust_interp() + enforce_monotone_pchip()

核心设计原则：
  - 输入输出单位与原始拟合链完全一致，确保即插即用替换
  - 模型自动从磁盘加载并缓存在进程内存中（首次加载后后续调用零开销）
  - 所有输出均经过 PCHIP 单调性安全网后处理

使用示例：
    from profile_nn.infer_lstm import nn_fit_te_lstm, nn_fit_ne_lstm, nn_fit_ti_lstm, set_model_dir_lstm

    set_model_dir_lstm('./profile_nn_models')  # 可选：指定模型目录

    rho_201, te_201 = nn_fit_te_lstm(rho_ts, te_ev)           # Te: eV 输入, keV 输出
    rho_201, ne_201 = nn_fit_ne_lstm(rho_refl, ne_1e19)       # ne: 1e19 m^-3 输入输出
    rho_201, ti_201 = nn_fit_ti_lstm(rho_txcs, ti_keV,        # Ti: keV 输入输出
                                      te_ped_x, te_ped_y)
"""

import os
import numpy as np
import torch

# 模型文件默认路径：相对于本文件的 ../profile_nn_models/
_model_dir = os.path.join(os.path.dirname(__file__), '..', 'profile_nn_models')
_cache = {}  # 进程级模型缓存：{datatype: model}，避免重复加载


def _load_lstm_model(datatype, model_dir=None):
    """从磁盘加载训练好的 LSTM Baseline 模型。按诊断类型缓存在进程内存中。

    加载流程：
      1. 检查缓存 —— 同一进程内多次调用只加载一次
      2. 读取 .pt 检查点文件，提取模型配置（hidden 维度等）
      3. 实例化对应架构类（BaseLSTMProfileNet / LSTMProfileNet_Ti）
      4. 加载权重并设为评估模式（eval）

    Args:
        datatype: 'Te', 'ne', 或 'Ti'
        model_dir: 模型文件目录，默认为 ../profile_nn_models/

    Returns:
        训练好的 PyTorch 模型实例（已在 CPU 上，处于 eval 模式）

    Raises:
        FileNotFoundError: 模型文件不存在时，提示训练命令
    """
    if datatype in _cache:
        return _cache[datatype]

    if model_dir is None:
        model_dir = _model_dir

    # 延迟导入：仅在需要加载模型时才导入架构类
    from profile_nn.lstm_baseline import BaseLSTMProfileNet, LSTMProfileNet_Ti

    model_path = os.path.join(model_dir, f'LSTM_{datatype}.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"LSTM 模型未找到: {model_path}\n"
            f"请先训练模型: python -m profile_nn.lstm_baseline --data <data_dir> --datatype {datatype}"
        )

    # weights_only=False 是因为检查点包含非张量数据（config dict, history 等）
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', {})
    hidden = config.get('hidden', 64)
    lstm_layers = config.get('lstm_layers', 1)

    if datatype == 'Ti':
        model = LSTMProfileNet_Ti(hidden=hidden, lstm_layers=lstm_layers, num_output=201)
    else:
        model = BaseLSTMProfileNet(hidden=hidden, lstm_layers=lstm_layers, num_output=201)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 评估模式：关闭 dropout 等训练行为（LSTM 无 dropout，但确保一致性）

    _cache[datatype] = model
    return model


def _predict_lstm_profile(model, x, mask, device='cpu'):
    """在单个样本上运行 LSTM 模型推理。

    输入为 numpy 数组，内部转为 PyTorch 张量，前向传播后转回 numpy。
    batch 维度为 1（单样本推理）。

    Args:
        model: LSTM 模型实例
        x: (N, 2) numpy 数组 —— (ρ, y) 散点对
        mask: (N,) bool numpy 数组 —— 有效点标记
        device: 推理设备，默认 'cpu'

    Returns:
        rho_out: (201,) float64 numpy 数组 —— 均匀 ρ 网格 [0, 1]
        y_out:   (201,) float64 numpy 数组 —— 预测剖面值
    """
    model = model.to(device)
    x_t = torch.from_numpy(x).unsqueeze(0).to(device)  # (1, N, 2)
    if mask is not None:
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)  # (1, N)
    else:
        mask_t = None

    with torch.no_grad():  # 推理不需要计算梯度，节省内存和计算
        y = model(x_t, mask_t)  # (1, 201)

    rho_out = np.linspace(0, 1, 201, dtype=np.float64)
    y_out = y.squeeze(0).cpu().numpy().astype(np.float64)
    return rho_out, y_out


# ===========================================================================
# 公开推理 API
# ===========================================================================

def nn_fit_te_lstm(rho_scattered, te_scattered, model_dir=None):
    """LSTM 替代 Te 拟合法：TS 散点 → Te 连续剖面。

    替代原始调用链:
        x_fit, y_fit, _, _, _ = fitting(x, y_keV, datatype)
        x_201, y_201 = robust_interp(x_fit, y_fit, datatype)

    Args:
        rho_scattered: 一维数组，Thomson 散射 ρ 坐标（归一化径向位置 0~1）
        te_scattered:  一维数组，Te 值 [eV]，来自 MDSplus 原始数据
        model_dir:     可选，模型文件目录路径

    Returns:
        rho_201: (201,) 均匀 ρ 网格 [0, 1]
        te_201:  (201,) Te 剖面 [keV] —— 注意输出单位是 keV
    """
    rho = np.asarray(rho_scattered, dtype=np.float32)
    te = np.asarray(te_scattered, dtype=np.float32)

    # 单位归一化：eV → keV（匹配训练数据分布）
    te_keV = te / 1000.0

    # 构造 (N, 2) 输入张量：第 0 列 ρ，第 1 列 Te[keV]
    x = np.stack([rho, te_keV], axis=-1)
    mask = np.ones(len(rho), dtype=bool)

    model = _load_lstm_model('Te', model_dir)
    rho_201, te_201 = _predict_lstm_profile(model, x, mask)

    # ---- 单调性安全网 ----
    # PCHIP 确保输出严格单调递减，然后逐点取累积最小值消除任何残留上升
    try:
        from scipy.interpolate import PchipInterpolator
        pchip = PchipInterpolator(rho_201, -te_201)
        te_201 = np.maximum(-pchip(rho_201), 0.005)
        # 保证严格非增：从左到右，截断任何上升
        te_201 = np.minimum.accumulate(te_201)
    except Exception:
        pass  # PCHIP 失败时退回到原始 LSTM 输出

    # 边缘下界保护
    te_201 = np.maximum(te_201, 0.005)
    te_201 = np.minimum.accumulate(te_201)
    return rho_201, te_201


def nn_fit_ne_lstm(rho_scattered, ne_scattered, model_dir=None):
    """LSTM 替代 ne 拟合法：Refl 散点 → ne 连续剖面。

    替代原始调用链:
        x_fit, y_fit, _, _, _ = fitting(x, y, datatype)
        x_201, y_201 = robust_interp(x_fit, y_fit, datatype)

    Args:
        rho_scattered: 一维数组，反射计 ρ 坐标（归一化径向位置 0~1）
        ne_scattered:  一维数组，ne 值 [1e19 m^-3]，来自 MDSplus
        model_dir:     可选，模型文件目录路径

    Returns:
        rho_201: (201,) 均匀 ρ 网格 [0, 1]
        ne_201:  (201,) ne 剖面 [1e19 m^-3]
    """
    rho = np.asarray(rho_scattered, dtype=np.float32)
    ne = np.asarray(ne_scattered, dtype=np.float32)

    x = np.stack([rho, ne], axis=-1)
    mask = np.ones(len(rho), dtype=bool)

    model = _load_lstm_model('ne', model_dir)
    rho_201, ne_201 = _predict_lstm_profile(model, x, mask)

    # ---- 单调性安全网 ----
    try:
        from scipy.interpolate import PchipInterpolator
        pchip = PchipInterpolator(rho_201, -ne_201)
        ne_201 = np.maximum(-pchip(rho_201), 0.001)
        # 保证严格非增
        ne_201 = np.minimum.accumulate(ne_201)
    except Exception:
        pass

    # 边缘下界保护
    ne_201 = np.maximum(ne_201, 0.001)
    ne_201 = np.minimum.accumulate(ne_201)
    return rho_201, ne_201


def nn_fit_ti_lstm(rho_scattered, ti_scattered, te_ped_x, te_ped_y, model_dir=None):
    """LSTM 替代 Ti 拟合法：TXCS 散点 + Te 台基 → Ti 连续剖面。

    替代原始调用链:
        x_fit, y_fit, _, _, _ = fitting(x, y, datatype, te_ped_x=..., te_ped_y=...)
        x_201, y_201 = robust_interp(x_fit, y_fit, datatype)
        x_201, y_201 = enforce_monotone_pchip(x_201, y_201)

    与 Te/ne 的关键区别：
      - 需要额外的 Te 台基数据（ρ ≥ 0.88 的 Te 拟合值）
      - 使用双 BiLSTM 编码器架构：TXCS 散点编码器 + Te 台基编码器
      - 推理时需要同时传入两组数据

    Args:
        rho_scattered: 一维数组，TXCS ρ 坐标
        ti_scattered:  一维数组，Ti 值 [keV]，来自 MDSplus TXCS 诊断
        te_ped_x:      一维数组，Te 台基 ρ 坐标（ρ ≥ 0.88 区域）
        te_ped_y:      一维数组，Te 台基值 [keV]
        model_dir:     可选，模型文件目录路径

    Returns:
        rho_201: (201,) 均匀 ρ 网格 [0, 1]
        ti_201:  (201,) Ti 剖面 [keV]
    """
    from scipy.interpolate import PchipInterpolator

    rho = np.asarray(rho_scattered, dtype=np.float32)
    ti = np.asarray(ti_scattered, dtype=np.float32)
    ped_x = np.asarray(te_ped_x, dtype=np.float32)
    ped_y = np.asarray(te_ped_y, dtype=np.float32)

    # 构造主输入 (TXCS 散点) 和台基输入 (Te pedestal 散点)
    x_txcs = np.stack([rho, ti], axis=-1)
    mask_txcs = np.ones(len(rho), dtype=bool)
    x_ped = np.stack([ped_x, ped_y], axis=-1)
    mask_ped = np.ones(len(ped_x), dtype=bool)

    # Ti 模型有双输入，不能直接用 _predict_lstm_profile 辅助函数
    model = _load_lstm_model('Ti', model_dir)
    model = model.to('cpu')

    x_t = torch.from_numpy(x_txcs).unsqueeze(0)    # (1, N_txcs, 2)
    m_t = torch.from_numpy(mask_txcs).unsqueeze(0)  # (1, N_txcs)
    p_t = torch.from_numpy(x_ped).unsqueeze(0)      # (1, N_ped, 2)
    mp_t = torch.from_numpy(mask_ped).unsqueeze(0)  # (1, N_ped)

    with torch.no_grad():
        y = model(x_t, m_t, p_t, mp_t)  # (1, 201)

    rho_201 = np.linspace(0, 1, 201, dtype=np.float64)
    ti_201 = y.squeeze(0).cpu().numpy().astype(np.float64)

    # ---- 单调性安全网 ----
    try:
        pchip = PchipInterpolator(rho_201, -ti_201)
        ti_201_mono = -pchip(rho_201)
        ti_201 = np.maximum(ti_201_mono, 0.01)
        # 保证严格非增
        ti_201 = np.minimum.accumulate(ti_201)
    except Exception:
        pass  # PCHIP 失败时退回到原始 LSTM 输出

    # 边缘下界保护
    ti_201 = np.maximum(ti_201, 0.005)
    ti_201 = np.minimum.accumulate(ti_201)
    return rho_201, ti_201


def set_model_dir_lstm(path):
    """设置包含训练好的 LSTM .pt 模型文件的目录路径。

    调用此函数后：
      - 后续的 nn_fit_te_lstm/ne_lstm/ti_lstm 调用将从新路径加载模型
      - 缓存被清空，下一次调用会重新加载模型

    Args:
        path: 模型目录的绝对或相对路径
    """
    global _model_dir
    _model_dir = path
    _cache.clear()
