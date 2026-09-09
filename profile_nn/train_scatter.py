# -*- coding: utf-8 -*-
"""
train_scatter.py — Scatter-MSE 训练：NN 直接从原始诊断散点学习

与 train.py 的区别：
  - train.py: MSE(y_pred_201, mtanh_label_201)
  - train_scatter.py: MSE(interp(y_pred_201, rho_scatter), y_scatter) + 物理约束

这是概念验证脚本，仅支持 ProfileNet Te。
若验证可行，再扩展到全部 4 架构 × 3 诊断。

用法:
  pytorch_run -m profile_nn.train_scatter --data ./profile_nn_data --datatype Te
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import sys
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# ScatterPhysicsLoss: 在诊断散点上算 MSE + 物理约束
# ================================================================

class ScatterPhysicsLoss(nn.Module):
    """将 201 点网格输出插值到诊断散点位置，计算 scatter-MSE + 物理约束。

    physics_loss_fn 提供 L_mono + L_bdy + L_smooth（在 201 点头出上计算）。
    """
    def __init__(self, physics_loss_fn, w_scatter=1.0):
        super().__init__()
        self.physics = physics_loss_fn
        self.w_scatter = w_scatter
        self.rho_grid = torch.linspace(0, 1, 201)  # 用于构建插值权重

    def _interp_to_scatter(self, y_pred_201, rho_scatter_list):
        """将 y_pred (B, 201) 线性插值到各样本的散点 ρ 位置。

        rho_scatter_list: list of (N_i,) tensors，每个样本的散点 ρ 坐标
        返回: list of (N_i,) tensors，插值后的预测值
        """
        device = y_pred_201.device
        rho_grid = self.rho_grid.to(device)

        y_scatter_list = []
        for i, rho_i in enumerate(rho_scatter_list):
            if rho_i.numel() == 0:
                y_scatter_list.append(torch.zeros(0, device=device))
                continue
            # 线性插值: 对每个 ρ 找相邻网格点
            rho_i = rho_i.to(device)
            # 将 ρ 映射到 [0, 200] 的浮点索引
            idx_float = rho_i * 200.0
            idx_lo = torch.clamp(torch.floor(idx_float).long(), 0, 200)
            idx_hi = torch.clamp(idx_lo + 1, 0, 200)
            # 权重
            w_hi = idx_float - idx_lo.float()
            w_lo = 1.0 - w_hi
            # 插值
            y_i = w_lo * y_pred_201[i, idx_lo] + w_hi * y_pred_201[i, idx_hi]
            y_scatter_list.append(y_i)
        return y_scatter_list

    def forward(self, y_pred_201, y_scatter_list, rho_scatter_list):
        # 1. 插值到散点位置 → scatter-MSE
        y_interp_list = self._interp_to_scatter(y_pred_201, rho_scatter_list)
        scatter_losses = []
        for y_pred_i, y_true_i in zip(y_interp_list, y_scatter_list):
            if y_pred_i.numel() > 0:
                scatter_losses.append(F.mse_loss(y_pred_i, y_true_i.to(y_pred_i.device)))
        loss_scatter = torch.stack(scatter_losses).mean() if scatter_losses else torch.tensor(0.0, device=y_pred_201.device)

        # 2. 物理约束：physics(y_pred, y_pred.detach()) — MSE/log 项≈0，只保留物理项
        loss_phys, _ = self.physics(y_pred_201, y_pred_201.detach())

        total = self.w_scatter * loss_scatter + loss_phys

        return total, {
            'scatter': loss_scatter.item(),
            'phys': loss_phys.item(),
            'total': total.item(),
        }


# ================================================================
# 自定义 Dataset + Collate（返回原始散点 ρ + val）
# ================================================================

class ScatterProfileDataset(torch.utils.data.Dataset):
    """与 ProfileDataset 相同，但额外返回 raw_rho 和 raw_val 用于 scatter-MSE。"""

    def __init__(self, data_dir, datatype, split='train', split_ratio=(0.7, 0.15, 0.15)):
        import glob as _glob
        pattern = os.path.join(data_dir, f'*_{datatype}_*.npz')
        all_files = sorted(_glob.glob(pattern))
        if not all_files:
            raise FileNotFoundError(f'No .npz files matching {pattern} in {data_dir}')

        n = len(all_files)
        rng = np.random.RandomState(42)
        indices = rng.permutation(n)
        train_end = int(n * split_ratio[0])
        val_end = train_end + int(n * split_ratio[1])

        if split == 'train':
            idx = indices[:train_end]
        elif split == 'val':
            idx = indices[train_end:val_end]
        else:
            idx = indices[val_end:]

        self.files = [all_files[i] for i in idx]
        self.is_ti = (datatype == 'Ti')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data = np.load(self.files[index])
        X_rho = data['X_rho'].astype(np.float32)
        X_val = data['X_val'].astype(np.float32)
        Y = data['Y'].astype(np.float32)  # 保留 mtanh label 用于验证

        X = np.stack([X_rho, X_val], axis=-1)

        return {
            'X': torch.from_numpy(X),
            'Y': torch.from_numpy(Y),
            'raw_rho': torch.from_numpy(X_rho),  # 新增
            'raw_val': torch.from_numpy(X_val),  # 新增
        }


def scatter_collate_fn(batch):
    """与原始 collate_fn 相同，但额外传递 raw_rho 和 raw_val 列表。"""
    batch_size = len(batch)
    max_pts = max(b['X'].shape[0] for b in batch)
    dim = 2

    X_padded = torch.zeros(batch_size, max_pts, dim)
    mask = torch.zeros(batch_size, max_pts, dtype=torch.bool)
    Y = torch.stack([b['Y'] for b in batch])

    raw_rho_list = []
    raw_val_list = []

    for i, b in enumerate(batch):
        n = b['X'].shape[0]
        X_padded[i, :n, :] = b['X']
        mask[i, :n] = True
        raw_rho_list.append(b['raw_rho'])
        raw_val_list.append(b['raw_val'])

    return X_padded, mask, Y, raw_rho_list, raw_val_list


# ================================================================
# 训练循环
# ================================================================

def train_scatter(datatype, data_dir, output_dir, epochs=500, batch_size=64,
                  lr=1e-3, w_mono=0.05, w_bdy=0.05, w_smooth=0.01,
                  patience=100, device='cpu'):
    from profile_nn.model import PhysicsConstrainedLoss

    # 根据诊断类型选择对应的 ProfileNet 子类
    if datatype == 'Te':
        from profile_nn.model import ProfileNet_Te as ModelClass
    elif datatype == 'ne':
        from profile_nn.model import ProfileNet_ne as ModelClass
    else:
        from profile_nn.model import ProfileNet_Ti as ModelClass

    # 数据
    train_ds = ScatterProfileDataset(data_dir, datatype, 'train')
    val_ds = ScatterProfileDataset(data_dir, datatype, 'val')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=scatter_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=scatter_collate_fn)
    print(f'  Train: {len(train_ds)} samples, Val: {len(val_ds)} samples')

    model = ModelClass()
    model = model.to(device)

    # 物理约束（在 201 点头出上计算，不含 MSE/L_log）
    phys_loss = PhysicsConstrainedLoss(w_mono=w_mono, w_bdy=w_bdy, w_smooth=w_smooth, w_log=0)
    # 总损失 = scatter-MSE + 物理约束
    criterion = ScatterPhysicsLoss(phys_loss)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    patience_counter = 0
    history = {'train': [], 'val': []}

    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []
        for batch in train_loader:
            X, mask, Y_mtanh, raw_rho, raw_val = batch
            X, mask = X.to(device), mask.to(device)
            y_pred = model(X, mask)
            loss, loss_dict = criterion(y_pred, raw_val, raw_rho)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append({k: v for k, v in loss_dict.items()})

        scheduler.step()

        # Val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                X, mask, Y_mtanh, raw_rho, raw_val = batch
                X, mask = X.to(device), mask.to(device)
                y_pred = model(X, mask)
                _, loss_dict = criterion(y_pred, raw_val, raw_rho)
                val_losses.append({k: v for k, v in loss_dict.items()})

        def avg(ld, key):
            vals = [d[key] for d in ld if key in d and not np.isnan(d[key])]
            return np.mean(vals) if vals else 0

        train_total = avg(train_losses, 'total')
        val_total = avg(val_losses, 'total')
        t_scatter = avg(train_losses, 'scatter')
        v_scatter = avg(val_losses, 'scatter')
        history['train'].append({'total': train_total, 'scatter': t_scatter})
        history['val'].append({'total': val_total, 'scatter': v_scatter})

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f'  Epoch {epoch+1:3d}/{epochs}: '
                  f'train={train_total:.6f} val={val_total:.6f} '
                  f'(scatter: {t_scatter:.6f}/{v_scatter:.6f})')

        # 早停
        if val_total < best_val_loss:
            best_val_loss = val_total
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'  Early stopping at epoch {epoch+1}')
                break

    # 保存
    model.load_state_dict(best_state)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'ProfileNet_{datatype}.pt')
    torch.save({
        'model_state_dict': best_state,
        'history': history,
        'best_val_loss': best_val_loss,
        'config': {
            'hidden': 128, 'num_output': 201,
            'w_mono': w_mono, 'w_bdy': w_bdy, 'w_smooth': w_smooth,
            'scatter_label': True,
        },
        'val_loss': float(best_val_loss),
    }, save_path)
    print(f'  Model saved: {save_path}')


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(description='Scatter-MSE training (ProfileNet Te)')
    parser.add_argument('--data', type=str, default='./profile_nn_data')
    parser.add_argument('--datatype', type=str, default='Te')
    parser.add_argument('--output', type=str, default='./profile_nn_models_scatter')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--w-mono', type=float, default=0.05)
    parser.add_argument('--w-bdy', type=float, default=0.05)
    parser.add_argument('--w-smooth', type=float, default=0.01)
    args = parser.parse_args()

    print(f'{"="*60}')
    print(f'Scatter-MSE Training — ProfileNet_{args.datatype}')
    print(f'Loss: MSE(y_pred@scatter, y_scatter) + physics constraints')
    print(f'{"="*60}')

    train_scatter(
        datatype=args.datatype,
        data_dir=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        w_mono=args.w_mono,
        w_bdy=args.w_bdy,
        w_smooth=args.w_smooth,
        device=args.device,
    )

    print(f'\nModel saved to {args.output}/')


if __name__ == '__main__':
    main()
