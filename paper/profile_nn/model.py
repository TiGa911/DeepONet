"""DeepONet 风格的集合编码器 + 坐标解码器，用于等离子体剖面拟合。

三种模型变体：
  - ProfileNet_Te: TS 散点 → Te 201点剖面
  - ProfileNet_ne: Refl 散点 → ne 201点剖面
  - ProfileNet_Ti: TXCS 散点 + Te 台基 → Ti 201点剖面

架构：
  SetEncoder（集合编码器）: 逐点 MLP(2→64→128→128) + 全局最大池化 → 隐向量 z
  CoordDecoder（坐标解码器）: MLP(1→128→128) 作用于 rho → 基函数 b(ρ)
  输出: 隐向量 z 与基函数 b(ρ) 在 201 个网格点上的点积

总参数量: 每个模型约 50K，可在 CPU 上训练。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# 共享模块
# ---------------------------------------------------------------------------

class PerChannelSoftplus(nn.Module):
    """可学习的逐通道 Softplus 激活函数。

    标准 Softplus(x) = ln(1+e^x) 对大范围值输出接近线性，对负值输出接近零。
    通过引入可学习的 beta 参数：softplus(x; beta_i) = (1/beta_i) * ln(1+exp(beta_i*x))

    更大的 beta → 更陡峭的零到线性过渡 → 有利于表示台基处的陡峭下降。
    每个 rho 通道独立学习 beta，允许芯部平坦、台基陡峭的剖面形态。

    Args:
        num_channels: 输出通道数（= 201，ρ 网格点数）
        beta_init: beta 初始值（默认 2.0，比标准 Softplus beta=1 更陡）
    """
    def __init__(self, num_channels=201, beta_init=2.0):
        super().__init__()
        # 使用 raw_beta + softplus 确保 beta 始终 > 0
        raw_init = math.log(math.exp(beta_init) - 1)  # inverse softplus
        self.raw_beta = nn.Parameter(torch.full((num_channels,), raw_init))

    def forward(self, x):
        beta = F.softplus(self.raw_beta).clamp(0.5, 20.0)  # 限制范围防止不稳定
        return F.softplus(x * beta.unsqueeze(0)) / beta.unsqueeze(0)


# ---------------------------------------------------------------------------
# 共享模块
# ---------------------------------------------------------------------------

class SetEncoder(nn.Module):
    """PointNet 风格的集合编码器：处理可变长度的 (ρ, y) 散点集合。

    核心思想：每个诊断散点独立通过 MLP 提取特征，再通过全局最大池化
    将所有散点信息聚合为一个固定长度的隐向量 z。这种方法天然处理
    可变数量的输入点（TS ~20-25 个，Refl ~15-20 个，TXCS ~3-15 个），
    且对输入点的排列顺序不敏感（置换不变性）。
    """

    def __init__(self, input_dim=2, hidden=128):
        """
        Args:
            input_dim: 输入维度，2 表示 (ρ, y) 坐标对
            hidden: 隐层维度，也是输出隐向量 z 的维度
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),   # 2 → 64，低维特征提取
            nn.ReLU(),
            nn.Linear(64, hidden),      # 64 → 128，中维特征
            nn.ReLU(),
            nn.Linear(hidden, hidden),  # 128 → 128，高维特征
            nn.ReLU(),
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, N_max, 2) — 填充后的 (ρ, y) 散点，B=批次大小，N_max=批次内最大点数
            mask: (B, N_max) — True 表示有效点，False 表示填充位
        Returns:
            z: (B, hidden) — 全局隐向量，编码了整个剖面形状信息
        """
        feat = self.mlp(x)  # 逐点特征提取: (B, N, 2) → (B, N, hidden)
        if mask is not None:
            # 将有效点的特征保留，填充位清零
            feat = feat * mask.unsqueeze(-1).float()
            # 将填充位置设为极大负值，确保最大池化时忽略这些位置
            feat = feat.masked_fill(~mask.unsqueeze(-1), -1e9)
        z, _ = feat.max(dim=1)  # 全局最大池化: (B, N, hidden) → (B, hidden)
        return z


class CoordDecoder(nn.Module):
    """干路网络（Trunk Net）：将标量坐标 ρ 映射为基函数 b(ρ)。

    在 DeepONet 框架中，干路网络负责学习一组连续基函数，
    这些基函数定义在输出坐标空间上。对于每个目标 ρ 值，
    网络输出一个 hidden 维度的基函数向量。

    v2 增强：添加 Fourier 特征编码（sin/cos 投影），
    使 MLP 能表示台基处 ρ≈0.9 的高频陡峭梯度。
    原理参考 NeRF 位置编码：将低频 1D 坐标映射到高频空间，
    让线性组合即可表达尖锐过渡。
    """

    def __init__(self, hidden=128, output_dim=128, n_freq=32):
        """
        Args:
            hidden: 隐层宽度
            output_dim: 基函数维度，需与 SetEncoder 输出的 z 维度一致
            n_freq: Fourier 频率数量（默认 32，产生 64 维特征）
        """
        super().__init__()
        self.n_freq = n_freq
        # 固定的随机频率基（不可训练），用于 sin/cos 投影
        self.register_buffer('fourier_B', torch.randn(1, n_freq) * 2.0)
        input_dim = 1 + 2 * n_freq  # 原始 ρ + sin 分量 + cos 分量
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),      # 1+2n_freq → 128，Fourier 增强输入
            nn.SiLU(),                       # SiLU 激活，比 ReLU 更平滑
            nn.Linear(128, 128),            # 128 → 128
            nn.SiLU(),
            nn.Linear(128, output_dim),     # 128 → hidden，输出基函数
        )

    def forward(self, rho_grid):
        """
        Args:
            rho_grid: (201,) 或 (B, 201) — 目标 ρ 网格坐标，范围 [0, 1]
        Returns:
            basis: (201, output_dim) 或 (B, 201, output_dim) — 基函数
        """
        single = rho_grid.dim() == 1
        if single:
            rho_grid = rho_grid.unsqueeze(-1)  # (..., 1)
        else:
            rho_grid = rho_grid.unsqueeze(-1)  # (..., 1)

        # Fourier 特征展开: ρ → [ρ, sin(2π·B·ρ), cos(2π·B·ρ)]
        proj = 2 * math.pi * rho_grid @ self.fourier_B  # (..., n_freq)
        ff = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (..., 2*n_freq)
        rho_aug = torch.cat([rho_grid, ff], dim=-1)  # (..., 1 + 2*n_freq)

        basis = self.net(rho_aug)  # (..., hidden)
        return basis


# ---------------------------------------------------------------------------
# 基础剖面网络
# ---------------------------------------------------------------------------

class BaseProfileNet(nn.Module):
    """DeepONet 剖面网络骨架：集合编码 → 基函数展开 → 点积 → Softplus。

    数学形式: y(ρ) = softplus( z · b(ρ) )
    其中 z 是编码器从散点中提取的隐向量（支路输出），
    b(ρ) 是解码器在坐标 ρ 处生成的基函数（干路输出）。

    Softplus 确保输出恒为正，满足 Te/ne/Ti 的物理约束。
    """

    def __init__(self, hidden=128, num_output=201):
        """
        Args:
            hidden: 隐层维度，编码器和解码器共用
            num_output: 输出剖面点数，默认 201（ρ 从 0 到 1 均匀分布）
        """
        super().__init__()
        self.encoder = SetEncoder(input_dim=2, hidden=hidden)
        self.decoder = CoordDecoder(hidden=hidden, output_dim=hidden)
        self.hidden = hidden
        self.num_output = num_output
        # 注册为 buffer（非可训练参数），随模型一起移动到设备
        self.register_buffer("rho_grid", torch.linspace(0, 1, num_output))
        # 可学习逐通道 Softplus：让模型自适应调整每个 ρ 位置的陡峭度
        self.activation = PerChannelSoftplus(num_output, beta_init=2.0)

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, N_max, 2) — 填充后的 (ρ, y) 诊断散点
            mask: (B, N_max) — True 表示有效散点
        Returns:
            y: (B, num_output) — 预测的连续剖面，正值（keV 或 1e19 m^-3）
        """
        B = x.shape[0]

        # 1. 集合编码：散点 → 隐向量 z
        z = self.encoder(x, mask)  # (B, hidden)

        # 2. 坐标解码：ρ 网格 → 基函数 b(ρ)
        rho = self.rho_grid.unsqueeze(0).expand(B, -1)  # (B, num_output)
        basis = self.decoder(rho)  # (B, num_output, hidden)

        # 3. 点积合成：y(ρ) = z · b(ρ)
        y = (z.unsqueeze(1) * basis).sum(dim=-1)  # (B, num_output)

        # 4. 可学习 Softplus 正性约束（允许台基区域更陡峭的过渡）
        y = self.activation(y)
        return y


# ---------------------------------------------------------------------------
# 各诊断专用模型
# ---------------------------------------------------------------------------

class ProfileNet_Te(BaseProfileNet):
    """电子温度剖面网络：从 Thomson 散射 (TS) 散点预测 Te 剖面。

    输入:  (B, N_ts, 2)  — (ρ, Te) 散点对，Te 单位 keV
    输出:  (B, 201)       — 归一化 ρ 网格上的 Te 剖面 [keV]
    用途:  替代 fitting(x,y,'Te') + robust_interp() 拟合链
    """
    pass


class ProfileNet_ne(BaseProfileNet):
    """电子密度剖面网络：从反射计 (Refl) 散点预测 ne 剖面。

    输入:  (B, N_refl, 2) — (ρ, ne) 散点对，ne 单位 1e19 m^-3
    输出:  (B, 201)       — 归一化 ρ 网格上的 ne 剖面 [1e19 m^-3]
    用途:  替代 fitting(x,y,'Refl') + robust_interp() 拟合链
    """
    pass


class ProfileNet_Ti(BaseProfileNet):
    """离子温度剖面网络：从 TXCS 散点 + Te 台基信息预测 Ti 剖面。

    Ti 数据通常非常稀疏（TXCS 只有 3-15 个通道），仅凭自身信息难以
    准确重建完整剖面。因此引入 Te 台基（ρ ≥ 0.88 区域的 Te 拟合值）
    作为辅助输入，提供边界约束——这与原始 mtanh 拟合中 Ti 依赖
    Te pedestal 的逻辑一致。

    双编码器结构：
      - 主编码器：处理 TXCS Ti 散点
      - 台基编码器：处理 Te 台基散点
      - 融合 MLP：拼接两个隐向量后降维，得到最终隐向量 z

    输入:  (B, N_txcs, 2) — (ρ, Ti) 散点对 [keV]
           (B, N_ped, 2)  — (ρ, Te) 台基散点对，ρ ≥ 0.88 [keV]
    输出:  (B, 201)       — 归一化 ρ 网格上的 Ti 剖面 [keV]
    """

    def __init__(self, hidden=128, num_output=201):
        super().__init__(hidden, num_output)
        # 台基分支的独立集合编码器
        self.pedestal_encoder = SetEncoder(input_dim=2, hidden=hidden)
        # 融合 MLP：将两个隐向量拼接后投影到统一隐空间
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden),  # 2×hidden → hidden
            nn.ReLU(),
            nn.Linear(hidden, hidden),      # hidden → hidden
        )

    def forward(self, x_txcs, mask_txcs, x_ped, mask_ped):
        """双编码器前向传播。

        Args:
            x_txcs:  (B, N_txcs, 2) — TXCS 诊断散点 (ρ, Ti)
            mask_txcs: (B, N_txcs) — TXCS 有效掩码
            x_ped:   (B, N_ped, 2)  — Te 台基散点 (ρ, Te)，仅 ρ ≥ 0.88
            mask_ped: (B, N_ped) — 台基有效掩码
        Returns:
            y: (B, num_output) — Ti 剖面 [keV]
        """
        B = x_txcs.shape[0]

        # 1. 两个编码器独立处理各自输入
        z_txcs = self.encoder(x_txcs, mask_txcs)          # TXCS Ti → 隐向量 (B, hidden)
        z_ped = self.pedestal_encoder(x_ped, mask_ped)   # Te 台基 → 隐向量 (B, hidden)

        # 2. 拼接并融合
        z = self.fusion(torch.cat([z_txcs, z_ped], dim=-1))  # (B, 2h) → (B, hidden)

        # 3. 解码 + 点积 + 可学习 Softplus（与基类一致）
        rho = self.rho_grid.unsqueeze(0).expand(B, -1)
        basis = self.decoder(rho)  # (B, num_output, hidden)
        y = (z.unsqueeze(1) * basis).sum(dim=-1)
        y = self.activation(y)
        return y


# ---------------------------------------------------------------------------
# 物理约束损失函数
# ---------------------------------------------------------------------------

class PhysicsConstrainedLoss(nn.Module):
    """带物理约束的加权损失函数。

    总损失 = MSE + λ₁·单调性 + λ₂·边界条件 + λ₃·平滑性

    各项物理意义：
      - L_MSE（数据保真）: 预测剖面与 mtanh 标注剖面的均方误差
      - L_mono（单调性）: 惩罚随 ρ 增大而上升的片段。Te/ne/Ti 剖面
        在等离子体中应单调递减（芯部高、边缘低），因此 dy/dρ ≤ 0
      - L_bdy（边界条件）: 在磁轴（ρ ≈ 0）附近，剖面应对称，
        导数应趋近于 0，即 dy/dρ |_{ρ→0} ≈ 0
      - L_smooth（平滑性）: 惩罚大的二阶导数，抑制剖面振荡和高频噪声
    """

    def __init__(self, w_mono=0.05, w_bdy=0.05, w_smooth=0.01,
                 pedestal_weight=25.0, pedestal_rho_start=0.82,
                 core_weight=3.0, core_rho_end=0.30,
                 w_log=0.1, log_eps=0.01):
        """
        Args:
            w_mono: 单调性约束权重（默认 0.05）
            w_bdy: 边界导数约束权重（默认 0.05）
            w_smooth: 平滑性约束权重（默认 0.01）
            pedestal_weight: 台基区域 MSE 加权倍率（默认 25×）
            pedestal_rho_start: 台基区域起始 ρ（默认 0.82）
            core_weight: 芯部区域 MSE 加权倍率（默认 3×，防止被台基高权重稀释）
            core_rho_end: 芯部区域结束 ρ（默认 0.30）
            w_log: 对数空间损失权重（默认 0.1）
            log_eps: 对数空间下界（默认 0.01）
        """
        super().__init__()
        self.w_mono = w_mono
        self.w_bdy = w_bdy
        self.w_smooth = w_smooth
        self.pedestal_weight = pedestal_weight
        self.pedestal_rho_start = pedestal_rho_start
        self.core_weight = core_weight
        self.core_rho_end = core_rho_end
        self.w_log = w_log
        self.log_eps = log_eps
        self.mse = nn.MSELoss()
        self.rho_grid = torch.linspace(0, 1, 201)

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: (B, 201) — 网络预测的剖面
            y_true: (B, 201) — mtanh 拟合的标注剖面（训练目标）
        Returns:
            total_loss: 标量，加权总损失
            loss_dict: 字典，包含各项损失的数值（用于日志记录）
        """
        device = y_pred.device
        rho = self.rho_grid.to(device)
        drho = rho[1] - rho[0]  # 网格间距 ≈ 0.005

        # ---- 1. 数据保真度（芯部+台基区域加权） ----
        # 芯部 (rho < core_rho_end): 加权促进峰值度，防止被台基高权重稀释
        # 台基 (rho > pedestal_rho_start): 高权重捕捉陡峭梯度
        # 中间区域: 权重 1.0
        weights = torch.ones(201, device=device)
        core_mask = rho < self.core_rho_end
        ped_mask = rho > self.pedestal_rho_start
        weights[core_mask] = self.core_weight
        weights[ped_mask] = self.pedestal_weight
        weights = weights / weights.mean()  # 归一化：保持总 MSE 尺度不变
        loss_mse = ((y_pred - y_true) ** 2 * weights.unsqueeze(0)).mean()

        # ---- 1b. 对数空间损失（比值敏感性） ----
        # 对数空间对边缘低值区域的相对误差更敏感
        # 防止 Te=0.05→0.01 keV 这种 80% 相对误差被 MSE 忽略
        y_pred_safe = torch.clamp(y_pred, min=self.log_eps)
        y_true_safe = torch.clamp(y_true, min=self.log_eps)
        loss_log = ((torch.log(y_pred_safe) - torch.log(y_true_safe)) ** 2).mean()

        # ---- 2. 单调性约束 ----
        # 剖面应随 ρ 单调递减，因此正导数（递增段）需被惩罚
        dy = torch.diff(y_pred, dim=1) / drho  # 一阶前向差分: (B, 200)
        loss_mono = F.relu(dy).mean()  # ReLU 只保留正值（递增），取平均

        # ---- 3. 磁轴对称边界条件 ----
        # 在 ρ < 0.05 区域，导数应趋近于 0
        boundary_idx = (rho < 0.05).sum().item()  # 约前 10 个网格点
        dy_boundary = torch.diff(y_pred[:, :boundary_idx], dim=1) / drho
        loss_bdy = (dy_boundary ** 2).mean()  # 导数的平方和（驱动导数 → 0）

        # ---- 4. 剖面平滑性 ----
        # 二阶导数绝对值均值，抑制振荡
        d2y = torch.diff(y_pred, n=2, dim=1) / (drho ** 2)  # 二阶中心差分: (B, 199)
        loss_smooth = d2y.abs().mean()

        # ---- 加权总损失 ----
        total = (loss_mse
                 + self.w_log * loss_log
                 + self.w_mono * loss_mono
                 + self.w_bdy * loss_bdy
                 + self.w_smooth * loss_smooth)

        loss_dict = {
            'mse': loss_mse.item(),
            'log': loss_log.item(),
            'mono': loss_mono.item(),
            'bdy': loss_bdy.item(),
            'smooth': loss_smooth.item(),
            'total': total.item(),
        }
        return total, loss_dict
