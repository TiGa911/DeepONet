"""训练 ProfileNet 模型：使用物理约束损失对 Te/ne/Ti 三个诊断类型的神经网络进行训练。

训练特性：
  - 物理约束损失函数：同时优化 MSE + 单调性 + 磁轴边界 + 平滑性四项
  - 早停机制：验证损失连续 patience 轮无改善则停止，防止过拟合
  - 余弦退火学习率调度：训练全程平滑降低学习率
  - 梯度裁剪：限制梯度范数 ≤ 1.0，防止训练不稳定
  - 模型保存：仅保存在验证集上最优的检查点

使用方法：
    python -m profile_nn.train --data ./profile_nn_data --datatype Te --epochs 500
    python -m profile_nn.train --data ./profile_nn_data --datatype ne
    python -m profile_nn.train --data ./profile_nn_data --datatype Ti
    python -m profile_nn.train --data ./profile_nn_data --datatype all   # 依次训练全部三个
    python -m profile_nn.train --data ./profile_nn_data --datatype all --device cuda  # GPU 训练
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

# 将父目录加入搜索路径，以便导入 profile_nn 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profile_nn.model import (
    ProfileNet_Te, ProfileNet_ne, ProfileNet_Ti, PhysicsConstrainedLoss,
)
from profile_nn.dataset import ProfileDataset, StratifiedProfileDataset, collate_fn


def train_one_model(datatype, data_dir, output_dir, epochs=500, batch_size=64,
                    lr=1e-3, w_mono=0.05, w_bdy=0.05, w_smooth=0.01,
                    w_log=0.1, patience=100, device='cpu',
                    split_method='random', plasma_mode='all'):
    """训练单个诊断类型的 ProfileNet 模型。

    训练流程：
      1. 加载训练集和验证集（ProfileDataset，固定 42 号种子划分）
      2. 实例化对应架构（ProfileNet_Te/ne/Ti）
      3. 使用 PhysicsConstrainedLoss 计算物理约束损失
      4. AdamW 优化器 + CosineAnnealingLR 余弦退火调度
      5. 每 epoch 在验证集上评估，记录最优模型
      6. 验证损失连续 patience=50 轮不降则早停

    Args:
        datatype: 诊断类型 'Te', 'ne', 或 'Ti'
        data_dir: 训练数据 .npz 文件所在目录
        output_dir: 模型保存目录
        epochs: 最大训练轮数，默认 500
        batch_size: 批次大小，默认 64
        lr: 初始学习率，默认 1e-3
        w_mono: 单调性约束权重 λ₁，默认 0.05
        w_bdy: 边界导数约束权重 λ₂，默认 0.05
        w_smooth: 平滑性约束权重 λ₃，默认 0.01
        patience: 早停容忍轮数，默认 50
        device: 训练设备 'cpu', 'cuda', 或 'mps'

    Returns:
        model: 训练好的模型（已加载最优检查点权重）
        history: 字典 {'train': [...], 'val': [...]}，每轮损失记录
    """

    print(f"\n{'='*60}")
    print(f"训练 ProfileNet_{datatype}")
    print(f"{'='*60}")

    # ---- 数据集 ----
    if split_method == 'stratified':
        train_ds = StratifiedProfileDataset(data_dir, datatype, split='train',
                                            plasma_mode=plasma_mode)
        val_ds = StratifiedProfileDataset(data_dir, datatype, split='val',
                                          plasma_mode=plasma_mode)
    else:
        train_ds = ProfileDataset(data_dir, datatype, split='train')
        val_ds = ProfileDataset(data_dir, datatype, split='val')
    print(f"训练集: {len(train_ds)} 样本, 验证集: {len(val_ds)} 样本")

    # 数据加载器：训练集打乱并丢弃最后不完整批次，验证集保持顺序
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn)

    # ---- 模型 ----
    is_ti = (datatype == 'Ti')
    if datatype == 'Te':
        model = ProfileNet_Te(hidden=128, num_output=201).to(device)
    elif datatype == 'ne':
        model = ProfileNet_ne(hidden=128, num_output=201).to(device)
    else:
        # Ti 使用双编码器架构：TXCS 编码 + Te 台基编码 + 融合 MLP
        model = ProfileNet_Ti(hidden=128, num_output=201).to(device)

    print(f"参数总数: {sum(p.numel() for p in model.parameters()):,}")

    # ---- 损失函数 & 优化器 ----
    # PhysicsConstrainedLoss: L = MSE + λ₁·L_mono + λ₂·L_bdy + λ₃·L_smooth
    criterion = PhysicsConstrainedLoss(w_mono=w_mono, w_bdy=w_bdy, w_smooth=w_smooth, w_log=w_log)
    # AdamW: 带权重衰减的 Adam，decay=1e-5 提供轻微正则化
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    # 余弦退火：学习率从 lr 平滑降至接近 0，训练后期精细收敛
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ---- 训练循环 ----
    best_val_loss = float('inf')
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    patience_counter = 0
    history = {'train': [], 'val': []}

    for epoch in range(epochs):
        # ===== 训练阶段 =====
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()

            if is_ti:
                # Ti 批次包含主输入和台基输入两组数据
                X, mask, X_ped, mask_ped, Y = [b.to(device) for b in batch]
                y_pred = model(X, mask, X_ped, mask_ped)
            else:
                # Te/ne 批次仅包含主输入
                X, mask, Y = [b.to(device) for b in batch]
                y_pred = model(X, mask)

            # 前向传播 → 损失 → 反向传播 → 梯度裁剪 → 参数更新
            loss, loss_dict = criterion(y_pred, Y)
            loss.backward()
            # 梯度裁剪：限制全局梯度范数 ≤ 1.0，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss_dict)

        # 学习率调度：每个 epoch 后更新
        scheduler.step()

        # ===== 验证阶段 =====
        model.eval()
        val_losses = []
        with torch.no_grad():  # 验证时关闭梯度计算，节省内存
            for batch in val_loader:
                if is_ti:
                    X, mask, X_ped, mask_ped, Y = [b.to(device) for b in batch]
                    y_pred = model(X, mask, X_ped, mask_ped)
                else:
                    X, mask, Y = [b.to(device) for b in batch]
                    y_pred = model(X, mask)

                _, loss_dict = criterion(y_pred, Y)
                val_losses.append(loss_dict)

        # ===== 损失汇总 =====
        def avg_key(dicts, key):
            """计算字典列表中指定键的平均值。"""
            return np.mean([d[key] for d in dicts])

        train_avg = {k: avg_key(train_losses, k) for k in train_losses[0]}
        val_avg = {k: avg_key(val_losses, k) for k in val_losses[0]}
        history['train'].append(train_avg)
        history['val'].append(val_avg)

        # 每 50 轮打印一次训练状态
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"训练 MSE={train_avg['mse']:.4e} mono={train_avg['mono']:.4e} | "
                  f"验证 MSE={val_avg['mse']:.4e} mono={val_avg['mono']:.4e}")

        # ===== 早停检查 =====
        # 当验证总损失连续 patience 轮未改善时停止训练
        val_total = val_avg['total']
        if val_total < best_val_loss - 1e-6:
            best_val_loss = val_total
            patience_counter = 0
            # 保存最优模型权重快照
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停于 epoch {epoch+1}")
                break

    # ---- 保存最优模型 ----
    # 加载验证损失最小的检查点
    model.load_state_dict(best_state)
    save_path = os.path.join(output_dir, f'ProfileNet_{datatype}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'datatype': datatype,
        'history': history,  # 完整训练历史，便于后续分析
        'config': {
            'hidden': 128, 'num_output': 201,
            'w_mono': w_mono, 'w_bdy': w_bdy, 'w_smooth': w_smooth, 'w_log': w_log,
        },
    }, save_path)
    print(f"模型已保存至 {save_path}")
    print(f"最优验证损失: {best_val_loss:.4e}")

    return model, history


def main():
    """命令行入口：解析参数，依次训练指定的诊断类型模型。"""
    parser = argparse.ArgumentParser(description='训练 ProfileNet 模型')
    parser.add_argument('--data', type=str, required=True,
                        help='训练数据 .npz 文件所在目录')
    parser.add_argument('--datatype', type=str, default='all',
                        choices=['Te', 'ne', 'Ti', 'all'],
                        help='要训练的诊断类型（all = 依次训练全部三个）')
    parser.add_argument('--output', type=str, default='./profile_nn_models',
                        help='模型保存目录')
    parser.add_argument('--epochs', type=int, default=500,
                        help='最大训练轮数')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='初始学习率')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'mps'],
                        help='训练设备（cpu/cuda/mps）')
    parser.add_argument('--w-mono', type=float, default=0.05,
                        help='单调性约束损失权重 λ₁')
    parser.add_argument('--w-bdy', type=float, default=0.05,
                        help='磁轴边界导数约束损失权重 λ₂')
    parser.add_argument('--w-smooth', type=float, default=0.01,
                        help='二阶导数平滑性约束损失权重 λ₃')
    parser.add_argument('--w-log', type=float, default=0.1,
                        help='log-space 损失权重（消融实验设为 0）')
    parser.add_argument('--split-method', type=str, default='random',
                        choices=['random', 'stratified'],
                        help='数据集划分方式：random=按文件随机, stratified=按炮号分层')
    parser.add_argument('--plasma-mode', type=str, default='all',
                        choices=['H', 'L', 'all'],
                        help='仅使用指定等离子体模式的样本（需 h98_index.csv）')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设备检查：请求 CUDA 但不可用时自动回退到 CPU
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA 不可用，回退到 CPU")
        device = 'cpu'

    # 确定训练范围
    datatypes = ['Te', 'ne', 'Ti'] if args.datatype == 'all' else [args.datatype]

    # 依次训练每种诊断类型
    for dt in datatypes:
        try:
            train_one_model(
                datatype=dt,
                data_dir=args.data,
                output_dir=str(output_dir),
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                w_mono=args.w_mono,
                w_bdy=args.w_bdy,
                w_smooth=args.w_smooth,
                w_log=args.w_log,
                device=device,
                split_method=args.split_method,
                plasma_mode=args.plasma_mode,
            )
        except FileNotFoundError as e:
            print(f"跳过 {dt}: {e}")
        except Exception as e:
            print(f"训练 {dt} 时出错: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n模型已保存至 {output_dir.absolute()}")


if __name__ == '__main__':
    main()
