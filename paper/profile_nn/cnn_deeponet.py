"""CNN-DeepONet model — CNN encoder + shared CoordDecoder (DeepONet trunk).

This is the DeepONet-aligned CNN variant, introduced to resolve the
experimental confound between the pure CNN-1D control group and the three
DeepONet architectures (ProfileNet/LSTM/Transformer).

Key distinction from cnn_baseline.py (pure CNN-1D, control group):
  - Pure CNN-1D:      CNN encoder → residual blocks → CNN decoder → direct 201-pt
                      output (fully convolutional, NO branch-trunk dot product)
  - CNN-DeepONet:     CNN encoder → latent z → CoordDecoder (trunk net) → dot
                      product z·b(ρ) → Softplus. Shares the same CoordDecoder +
                      dot-product decoder as ProfileNet/LSTM/Transformer.

This makes the four-way encoder comparison (SetEncoder vs BiLSTM vs Transformer
vs CNN) a clean "encoder-only" comparison under a unified DeepONet decoder, while
the pure CNN-1D remains an independent control group isolating the effect of the
branch-trunk structural prior.

Architecture (DeepONet-aligned):
  - Scattered (rho, val) points → linear interpolation to 201-pt grid
  - 2-channel input: [interpolated_values, validity_mask]
  - CNNEncoder: Conv1d stem → N× ResidualBlock1D → 1×1 Conv → global max-pool → z
  - CoordDecoder (shared trunk): ρ → basis b(ρ)
  - Output: Softplus(z · b(ρ))

Usage:
  python -m profile_nn.cnn_deeponet --data ./profile_nn_data --datatype all --epochs 500
"""

import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reuse the grid interpolation / dataset / collate utilities from the pure CNN
from profile_nn.cnn_baseline import (
    scatter_to_grid, GridDataset, collate_cnn, ResidualBlock1D, GRID_SIZE,
)
from profile_nn.model import PhysicsConstrainedLoss, CoordDecoder


# ===========================================================================
# CNN Encoder (grid → latent vector z)
# ===========================================================================

class CNNEncoder(nn.Module):
    """CNN encoder: (B, 2, 201) grid → latent vector z ∈ (B, hidden).

    Mirrors the pure CNN's encoder stem and residual blocks, but terminates
    with a 1×1 convolution + global max-pooling to collapse the spatial feature
    map into a fixed-length latent vector — the branch-net output consumed by
    the CoordDecoder (trunk net) via the dot product.

    Input:  (B, 2, 201) — [interpolated_values, validity_mask]
    Output: (B, hidden)
    """

    def __init__(self, hidden=64, base_channels=32, num_res_blocks=3,
                 kernel_size=7):
        super().__init__()
        padding = kernel_size // 2

        self.stem = nn.Sequential(
            nn.Conv1d(2, base_channels, kernel_size, padding=padding),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(*[
            ResidualBlock1D(base_channels, kernel_size)
            for _ in range(num_res_blocks)
        ])
        # 1×1 conv maps the residual feature channels to the latent dimension
        self.head = nn.Conv1d(base_channels, hidden, 1)
        # Global max-pool over the ρ axis (PointNet-style symmetric aggregation)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.hidden = hidden

    def forward(self, x):
        out = self.stem(x)               # (B, C, 201)
        out = self.res_blocks(out)       # (B, C, 201)
        out = self.head(out)             # (B, hidden, 201)
        z = self.pool(out).squeeze(-1)   # (B, hidden)
        return z


# ===========================================================================
# CNN-DeepONet models
# ===========================================================================

class BaseCNNDeepONet(nn.Module):
    """Shared skeleton: CNN encoder → CoordDecoder → dot product → Softplus.

    Identical decoder convention to LSTM/Transformer baselines: hidden=64,
    plain F.softplus, CoordDecoder(hidden, output_dim=hidden).
    """

    def __init__(self, hidden=64, num_output=GRID_SIZE, base_channels=32,
                 num_res_blocks=3, kernel_size=7):
        super().__init__()
        self.encoder = CNNEncoder(
            hidden=hidden, base_channels=base_channels,
            num_res_blocks=num_res_blocks, kernel_size=kernel_size,
        )
        self.decoder = CoordDecoder(hidden=hidden, output_dim=hidden)
        self.hidden = hidden
        self.num_output = num_output
        self.register_buffer("rho_grid", torch.linspace(0, 1, num_output))

    def forward(self, x_grid):
        B = x_grid.shape[0]
        z = self.encoder(x_grid)  # (B, hidden)
        rho = self.rho_grid.unsqueeze(0).expand(B, -1)  # (B, num_output)
        basis = self.decoder(rho)  # (B, num_output, hidden)
        y = (z.unsqueeze(1) * basis).sum(dim=-1)  # (B, num_output)
        y = F.softplus(y)
        return y


class CNNDeepONet_Te(BaseCNNDeepONet):
    pass


class CNNDeepONet_ne(BaseCNNDeepONet):
    pass


class CNNDeepONet_Ti(BaseCNNDeepONet):
    """Dual-encoder CNN-DeepONet for Ti (TXCS + Te pedestal).

    Two parallel CNN encoders process the TXCS Ti grid and the Te pedestal grid;
    their latent vectors are concatenated and fused via an MLP before decoding —
    mirroring ProfileNet_Ti's dual-encoder design.
    """

    def __init__(self, hidden=64, num_output=GRID_SIZE, base_channels=32,
                 num_res_blocks=3, kernel_size=7):
        super().__init__(hidden, num_output, base_channels,
                         num_res_blocks, kernel_size)
        self.pedestal_encoder = CNNEncoder(
            hidden=hidden, base_channels=base_channels,
            num_res_blocks=num_res_blocks, kernel_size=kernel_size,
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, x_main, x_ped):
        B = x_main.shape[0]
        z_main = self.encoder(x_main)             # (B, hidden)
        z_ped = self.pedestal_encoder(x_ped)      # (B, hidden)
        z = self.fusion(torch.cat([z_main, z_ped], dim=-1))  # (B, hidden)
        rho = self.rho_grid.unsqueeze(0).expand(B, -1)
        basis = self.decoder(rho)                 # (B, num_output, hidden)
        y = (z.unsqueeze(1) * basis).sum(dim=-1)
        y = F.softplus(y)
        return y


# ===========================================================================
# Training
# ===========================================================================

def train_cnn_deeponet_model(datatype, data_dir, output_dir, epochs=500,
                             batch_size=64, lr=1e-3, w_mono=0.05, w_bdy=0.05,
                             w_smooth=0.01, w_log=0.1, patience=100, device='cpu',
                             split_method='random', plasma_mode='all'):
    """Train a CNN-DeepONet model (CNN encoder + shared CoordDecoder)."""

    print(f"\n{'='*60}")
    print(f"Training CNN-DeepONet — {datatype}")
    print(f"{'='*60}")

    train_ds = GridDataset(data_dir, datatype, split='train',
                           split_method=split_method, plasma_mode=plasma_mode)
    val_ds = GridDataset(data_dir, datatype, split='val',
                         split_method=split_method, plasma_mode=plasma_mode)
    print(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")
    print(f"Architecture: CNN encoder + CoordDecoder (DeepONet dot product)")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_cnn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_cnn)

    is_ti = (datatype == 'Ti')
    if is_ti:
        model = CNNDeepONet_Ti(base_channels=32, num_output=GRID_SIZE).to(device)
    else:
        model = CNNDeepONet_Te(base_channels=32, num_output=GRID_SIZE).to(device) \
            if datatype == 'Te' else CNNDeepONet_ne(base_channels=32,
                                                    num_output=GRID_SIZE).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    criterion = PhysicsConstrainedLoss(w_mono=w_mono, w_bdy=w_bdy,
                                       w_smooth=w_smooth, w_log=w_log)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    patience_counter = 0
    history = {'train': [], 'val': []}

    for epoch in range(epochs):
        # ---- Train ----
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            if is_ti:
                x_main, x_ped, Y = [b.to(device) for b in batch]
                y_pred = model(x_main, x_ped)
            else:
                x_main, Y = [b.to(device) for b in batch]
                y_pred = model(x_main)

            loss, loss_dict = criterion(y_pred, Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss_dict)

        scheduler.step()

        # ---- Validate ----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                if is_ti:
                    x_main, x_ped, Y = [b.to(device) for b in batch]
                    y_pred = model(x_main, x_ped)
                else:
                    x_main, Y = [b.to(device) for b in batch]
                    y_pred = model(x_main)
                _, loss_dict = criterion(y_pred, Y)
                val_losses.append(loss_dict)

        def avg_key(dicts, key):
            return float(np.mean([d[key] for d in dicts]))

        train_avg = {k: avg_key(train_losses, k) for k in train_losses[0]}
        val_avg = {k: avg_key(val_losses, k) for k in val_losses[0]}
        history['train'].append(train_avg)
        history['val'].append(val_avg)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"train MSE={train_avg['mse']:.4e} mono={train_avg['mono']:.4e} | "
                  f"val MSE={val_avg['mse']:.4e} mono={val_avg['mono']:.4e}")

        val_total = val_avg['total']
        if val_total < best_val_loss - 1e-6:
            best_val_loss = val_total
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(best_state)
    save_path = os.path.join(output_dir, f'CNNDeepONet_{datatype}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'datatype': datatype,
        'history': history,
        'config': {
            'base_channels': 32, 'num_output': GRID_SIZE,
            'w_mono': w_mono, 'w_bdy': w_bdy, 'w_smooth': w_smooth, 'w_log': w_log,
            'n_params': n_params,
        },
    }, save_path)
    print(f"Model saved → {save_path}")
    print(f"Best val loss: {best_val_loss:.4e}")

    return model, history


# ===========================================================================
# Inference API
# ===========================================================================

_model_dir_default = os.path.join(os.path.dirname(__file__), '..', 'profile_nn_models')
_cache = {}


def _load_cnn_deeponet_model(datatype, model_dir=None):
    """Load trained CNN-DeepONet model from disk. Cached per process."""
    if datatype in _cache:
        return _cache[datatype]

    if model_dir is None:
        model_dir = _model_dir_default

    model_path = os.path.join(model_dir, f'CNNDeepONet_{datatype}.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"CNNDeepONet model not found: {model_path}\n"
            f"Train first: python -m profile_nn.cnn_deeponet --data <dir> --datatype {datatype}"
        )

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    cfg = checkpoint.get('config', {})
    base_channels = cfg.get('base_channels', 32)

    if datatype == 'Ti':
        model = CNNDeepONet_Ti(base_channels=base_channels)
    elif datatype == 'Te':
        model = CNNDeepONet_Te(base_channels=base_channels)
    else:
        model = CNNDeepONet_ne(base_channels=base_channels)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    _cache[datatype] = model
    return model


def _pchip_monotone(y, floor):
    """PCHIP monotonicity safety net (shared by inference helpers)."""
    rho_201 = np.linspace(0, 1, len(y), dtype=np.float64)
    try:
        from scipy.interpolate import PchipInterpolator
        pchip = PchipInterpolator(rho_201, -y)
        y = np.maximum(-pchip(rho_201), floor)
    except Exception:
        pass
    y = np.maximum(y, floor)
    y = np.minimum.accumulate(y)
    return y


def cnn_deeponet_fit_te(rho_scattered, te_scattered, model_dir=None):
    """CNN-DeepONet (CNN encoder + CoordDecoder) for Te profile fitting.

    Args:
        rho_scattered: rho coordinates from TS
        te_scattered: Te values [eV]
    Returns:
        rho_201: (201,) uniform rho grid
        te_201: (201,) Te profile [keV]
    """
    rho = np.asarray(rho_scattered, dtype=np.float32)
    te = np.asarray(te_scattered, dtype=np.float32)
    te_keV = te / 1000.0  # eV → keV

    grid_val, grid_mask = scatter_to_grid(rho, te_keV)
    x = torch.from_numpy(np.stack([grid_val, grid_mask])).unsqueeze(0)

    model = _load_cnn_deeponet_model('Te', model_dir)
    with torch.no_grad():
        y = model(x).squeeze(0).cpu().numpy().astype(np.float64)

    rho_201 = np.linspace(0, 1, 201, dtype=np.float64)
    y = _pchip_monotone(y, 0.005)
    return rho_201, y


def cnn_deeponet_fit_ne(rho_scattered, ne_scattered, model_dir=None):
    """CNN-DeepONet for ne profile fitting.

    Args:
        rho_scattered: rho coordinates from Refl
        ne_scattered: ne values [1e19 m^-3]
    Returns:
        rho_201: (201,)
        ne_201: (201,) ne profile [1e19 m^-3]
    """
    rho = np.asarray(rho_scattered, dtype=np.float32)
    ne = np.asarray(ne_scattered, dtype=np.float32)

    grid_val, grid_mask = scatter_to_grid(rho, ne)
    x = torch.from_numpy(np.stack([grid_val, grid_mask])).unsqueeze(0)

    model = _load_cnn_deeponet_model('ne', model_dir)
    with torch.no_grad():
        y = model(x).squeeze(0).cpu().numpy().astype(np.float64)

    rho_201 = np.linspace(0, 1, 201, dtype=np.float64)
    y = _pchip_monotone(y, 0.001)
    return rho_201, y


def cnn_deeponet_fit_ti(rho_scattered, ti_scattered, te_ped_x, te_ped_y, model_dir=None):
    """CNN-DeepONet (dual encoder) for Ti profile fitting.

    Args:
        rho_scattered: rho coordinates from TXCS
        ti_scattered: Ti values [keV]
        te_ped_x, te_ped_y: Te pedestal data (rho >= 0.88)
    Returns:
        rho_201: (201,)
        ti_201: (201,) Ti profile [keV]
    """
    rho = np.asarray(rho_scattered, dtype=np.float32)
    ti = np.asarray(ti_scattered, dtype=np.float32)
    ped_x = np.asarray(te_ped_x, dtype=np.float32)
    ped_y = np.asarray(te_ped_y, dtype=np.float32)

    grid_val, grid_mask = scatter_to_grid(rho, ti)
    ped_val, ped_mask = scatter_to_grid(ped_x, ped_y)

    x_main = torch.from_numpy(np.stack([grid_val, grid_mask])).unsqueeze(0)
    x_ped = torch.from_numpy(np.stack([ped_val, ped_mask])).unsqueeze(0)

    model = _load_cnn_deeponet_model('Ti', model_dir)
    with torch.no_grad():
        y = model(x_main, x_ped).squeeze(0).cpu().numpy().astype(np.float64)

    rho_201 = np.linspace(0, 1, 201, dtype=np.float64)
    y = _pchip_monotone(y, 0.005)
    return rho_201, y


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train CNN-DeepONet models (CNN encoder + shared CoordDecoder)')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--datatype', type=str, default='all',
                        choices=['Te', 'ne', 'Ti', 'all'])
    parser.add_argument('--output', type=str, default='./profile_nn_models')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--w-mono', type=float, default=0.05)
    parser.add_argument('--w-bdy', type=float, default=0.05)
    parser.add_argument('--w-smooth', type=float, default=0.01)
    parser.add_argument('--w-log', type=float, default=0.1)
    parser.add_argument('--split-method', type=str, default='random',
                        choices=['random', 'stratified'])
    parser.add_argument('--plasma-mode', type=str, default='all',
                        choices=['H', 'L', 'all'])
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    datatypes = ['Te', 'ne', 'Ti'] if args.datatype == 'all' else [args.datatype]

    for dt in datatypes:
        try:
            train_cnn_deeponet_model(
                datatype=dt, data_dir=args.data, output_dir=str(output_dir),
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                w_mono=args.w_mono, w_bdy=args.w_bdy, w_smooth=args.w_smooth, w_log=args.w_log,
                device=device,
                split_method=args.split_method, plasma_mode=args.plasma_mode,
            )
        except FileNotFoundError as e:
            print(f"Skipping {dt}: {e}")
        except Exception as e:
            print(f"Error training {dt}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nCNN-DeepONet models saved in {output_dir.absolute()}")


if __name__ == '__main__':
    main()
