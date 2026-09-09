"""CNN-DeepONet 推理模块：CNN 编码器 + 共享 CoordDecoder（DeepONet trunk）。

与 profile_nn/infer_cnn.py（纯 CNN-1D）的接口一致。CNN-DeepONet 与
CNN-1D 的唯一区别在解码端：
  - CNN-1D       : 纯 Conv1d 解码器，直接输出 201 点（无 branch-trunk 点积）
  - CNN-DeepONet : CNN 编码器 → 潜向量 z → 共享 CoordDecoder（trunk）→ z·b(ρ) → Softplus

该变体用于消解「CNN-1D 是唯一不使用 DeepONet 的架构」这一实验混淆：
现在 ProfileNet/LSTM/Transformer/CNN-DeepONet 四个编码器共享同一 trunk 解码器，
形成干净的「仅编码器对比」，而 CNN-1D 单独作为无 trunk 结构的对照组。

使用示例：
    from profile_nn.infer_cnn_deeponet import (
        nn_fit_te_cnn_deeponet, nn_fit_ne_cnn_deeponet, nn_fit_ti_cnn_deeponet,
        set_model_dir_cnn_deeponet,
    )

    set_model_dir_cnn_deeponet('./profile_nn_models')

    rho_201, te_201 = nn_fit_te_cnn_deeponet(rho_ts, te_ev)           # eV → keV
    rho_201, ne_201 = nn_fit_ne_cnn_deeponet(rho_refl, ne_1e19)       # 1e19 m^-3
    rho_201, ti_201 = nn_fit_ti_cnn_deeponet(rho_txcs, ti_keV, te_ped_x, te_ped_y)
"""

from profile_nn.cnn_deeponet import (
    cnn_deeponet_fit_te,
    cnn_deeponet_fit_ne,
    cnn_deeponet_fit_ti,
)

# Re-export with consistent naming convention
nn_fit_te_cnn_deeponet = cnn_deeponet_fit_te
nn_fit_ne_cnn_deeponet = cnn_deeponet_fit_ne
nn_fit_ti_cnn_deeponet = cnn_deeponet_fit_ti


def set_model_dir_cnn_deeponet(path):
    """设置包含训练好的 CNN-DeepONet .pt 模型文件的目录路径。

    调用后缓存被清空，下一次调用会重新加载模型。

    Args:
        path: 模型目录的绝对或相对路径
    """
    import profile_nn.cnn_deeponet as _cd
    _cd._model_dir_default = path
    _cd._cache.clear()
