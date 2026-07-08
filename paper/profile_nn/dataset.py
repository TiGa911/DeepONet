"""ProfileDataset: 从 .npz 文件加载训练数据对，处理变长散点的填充与批处理。

数据格式：
  每个 .npz 文件包含：
    X_rho, X_val  — 诊断散点的 ρ 坐标和物理值（变长）
    Y             — mtanh 拟合的 201 点连续剖面（训练标签）
    Ti 样本额外包含：X_te_ped_rho, X_te_ped_val（Te 台基辅助信息）

文件命名规范：{炮号}_{诊断类型}_{时间}.npz，如 81481_Te_5d30000.npz
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class ProfileDataset(Dataset):
    """剖面数据集：管理 (诊断散点, mtanh标注剖面) 训练对。

    职责：
      1. 扫描数据目录，按文件名模式匹配找到所有 .npz 文件
      2. 使用固定随机种子 (42) 对文件进行确定性划分（训练/验证/测试）
         —— 避免同一炮号的不同时间片同时出现在训练和验证集中造成数据泄漏
      3. 按需加载 .npz 文件并转为 PyTorch 张量

    数据集划分：
      - 训练集: 70%（默认）
      - 验证集: 15%（默认，用于早停和超参选择）
      - 测试集: 15%（默认，用于最终评估）
    """

    def __init__(self, data_dir, datatype, split='train', split_ratio=(0.7, 0.15, 0.15)):
        """
        Args:
            data_dir: 存放 .npz 文件的目录路径
            datatype: 诊断类型，'Te'（电子温度）、'ne'（电子密度）或 'Ti'（离子温度）
            split: 数据集划分，'train'（训练）、'val'（验证）或 'test'（测试）
            split_ratio: (训练比例, 验证比例, 测试比例) 三元组，默认 7:1.5:1.5
        """
        self.data_dir = Path(data_dir)
        self.datatype = datatype

        # 扫描目录，找到该诊断类型的所有 .npz 文件
        # 模式示例: '*_Te_*.npz' 匹配 '81481_Te_5d30000.npz', 'synthetic_Te_00150.npz' 等
        pattern = f'*_{datatype}_*.npz'
        all_files = sorted(self.data_dir.glob(pattern))

        if not all_files:
            raise FileNotFoundError(
                f"在 {data_dir} 中未找到匹配模式 '{pattern}' 的 .npz 文件"
            )

        # 基于文件名哈希的确定性划分（固定随机种子 42）
        # 目的：同一文件总被分到同一子集，可复现，且避免时间序列泄漏
        np.random.seed(42)
        indices = np.random.permutation(len(all_files))
        n = len(all_files)
        train_end = int(n * split_ratio[0])          # 训练集截止索引
        val_end = train_end + int(n * split_ratio[1]) # 验证集截止索引

        if split == 'train':
            idx = indices[:train_end]
        elif split == 'val':
            idx = indices[train_end:val_end]
        else:
            idx = indices[val_end:]

        self.files = [all_files[i] for i in idx]
        self.is_ti = (datatype == 'Ti')  # Ti 需要额外加载 Te 台基数据

    def __len__(self):
        """返回该子集的样本总数。"""
        return len(self.files)

    def __getitem__(self, index):
        """加载单个 .npz 文件并返回张量字典。

        Args:
            index: 样本索引

        Returns:
            对于 Te/ne: {'X': (N,2) 张量, 'Y': (201,) 张量}
            对于 Ti:    {'X': (N,2), 'X_ped': (N_ped,2), 'Y': (201,)}
        """
        data = np.load(self.files[index])

        # 读取散点坐标和物理值，转为 float32 以匹配 PyTorch 默认精度
        X_rho = data['X_rho'].astype(np.float32)  # ρ 坐标（0~1 归一化径向位置）
        X_val = data['X_val'].astype(np.float32)  # 物理量值（Te: keV, ne: 1e19 m^-3, Ti: keV）
        Y = data['Y'].astype(np.float32)           # mtanh 标注的 201 点剖面

        # 将 (rho, val) 两个一维数组堆叠为 (N, 2) 的二维数组
        # 第 0 列 = ρ 坐标，第 1 列 = 物理值
        X = np.stack([X_rho, X_val], axis=-1)  # (N, 2)

        if self.is_ti:
            # Ti 样本额外包含 Te 台基信息（ρ ≥ 0.88 区域的 Te 拟合值）
            # 这些数据作为辅助输入，帮助模型在 Ti 散点稀疏时确定边界行为
            X_ped_rho = data['X_te_ped_rho'].astype(np.float32)
            X_ped_val = data['X_te_ped_val'].astype(np.float32)
            X_ped = np.stack([X_ped_rho, X_ped_val], axis=-1)  # (N_ped, 2)
            return {
                'X': torch.from_numpy(X),
                'X_ped': torch.from_numpy(X_ped),
                'Y': torch.from_numpy(Y),
            }
        else:
            return {
                'X': torch.from_numpy(X),
                'Y': torch.from_numpy(Y),
            }


def collate_fn(batch):
    """自定义批处理函数：将批次内变长散点填充到统一长度。

    由于不同诊断时间片的散点数量不同（TS ~20-25 个，Refl ~15-20 个，TXCS ~3-15 个），
    需要将同一批次内的样本填充到该批次的最大点数 N_max。
    填充位置用全零向量表示，并由布尔掩码标记有效位置。

    Args:
        batch: list of dict，每个元素是 __getitem__ 返回的字典

    Returns:
        Te/ne 批次: X_padded (B,N_max,2), mask (B,N_max), Y (B,201)
        Ti 批次:    X_padded, mask, X_ped_padded, mask_ped, Y
    """
    is_ti = 'X_ped' in batch[0]

    # 找到本批次的最大散点数
    N_max = max(item['X'].shape[0] for item in batch)
    B = len(batch)

    # 主输入填充：散点 (ρ, y)
    X_padded = torch.zeros(B, N_max, 2)        # 全零填充
    mask = torch.zeros(B, N_max, dtype=torch.bool)  # False = 填充位
    Y = torch.stack([item['Y'] for item in batch])  # 标签维度一致，直接堆叠

    for i, item in enumerate(batch):
        n = item['X'].shape[0]
        X_padded[i, :n] = item['X']   # 将有效散点复制到前 n 个位置
        mask[i, :n] = True            # 标记前 n 个位置为有效

    if is_ti:
        # 台基输入同样需要填充
        N_ped_max = max(item['X_ped'].shape[0] for item in batch)
        X_ped_padded = torch.zeros(B, N_ped_max, 2)
        mask_ped = torch.zeros(B, N_ped_max, dtype=torch.bool)
        for i, item in enumerate(batch):
            n_ped = item['X_ped'].shape[0]
            X_ped_padded[i, :n_ped] = item['X_ped']
            mask_ped[i, :n_ped] = True
        return X_padded, mask, X_ped_padded, mask_ped, Y
    else:
        return X_padded, mask, Y
