"""CNN-1D baseline model for profile fitting — independent control group for ProfileNet.

Architecture (standalone CNN, NOT DeepONet):
  - Scattered (rho, val) points → linear interpolation to 201-pt fixed grid
  - 2-channel input: [interpolated_values, validity_mask]
  - 1D ResNet: encoder → residual blocks → decoder → Softplus → 201-pt output
  - End-to-end convolutional, no branch-trunk dot product, no CoordDecoder

Contrast with ProfileNet & LSTM:
  - ProfileNet:  SetEncoder + CoordDecoder + dot product — DeepONet
  - LSTM:       BiLSTM encoder + CoordDecoder + dot product — DeepONet
  - CNN-1D:     Pure ResNet encoder-decoder — classic ConvNet, fully distinct

This is a genuine architectural control group, not a variant of the DeepONet pattern.

Usage:
  python -m profile_nn.cnn_baseline --data ./profile_nn_data --datatype all --epochs 500
"""

import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profile_nn.dataset import ProfileDataset, StratifiedProfileDataset
from profile_nn.model import PhysicsConstrainedLoss

GRID_SIZE = 201  # output grid = input grid for pure CNN


# ===========================================================================
# Scattered → Grid interpolation
# ===========================================================================

def scatter_to_grid(rho_scatter, val_scatter, grid_size=GRID_SIZE):
    """Convert scattered points to fixed grid with validity mask.

    Returns:
        grid_vals: (grid_size,) float32 — linearly interpolated values
        grid_mask: (grid_size,) float32 — 1.0 near data, 0.0 in data-free regions
    """
    from scipy.interpolate import interp1d

    rho_grid = np.linspace(0, 1, grid_size, dtype=np.float32)
    grid_mask = np.zeros(grid_size, dtype=np.float32)

    rho = np.asarray(rho_scatter, dtype=np.float64)
    val = np.asarray(val_scatter, dtype=np.float64)

    if len(rho) == 0:
        return np.zeros(grid_size, dtype=np.float32), grid_mask

    # Sort
    order = np.argsort(rho)
    rho_s, val_s = rho[order], val[order]

    # Mask: bins within 1.5× half-bin of any data point
    bin_width = 1.0 / (grid_size - 1)
    for i, rg in enumerate(rho_grid):
        if np.any(np.abs(rho_s - rg) < bin_width * 1.5):
            grid_mask[i] = 1.0

    # Linear interpolation
    if len(rho_s) >= 2:
        f = interp1d(rho_s, val_s, kind='linear',
                     bounds_error=False, fill_value='extrapolate')
        grid_vals = f(rho_grid).astype(np.float32)
    else:
        grid_vals = np.full(grid_size, float(val_s[0]), dtype=np.float32)

    # Clip extrapolation beyond data range to nearest valid value
    rho_min, rho_max = rho_s.min(), rho_s.max()
    if rho_min > 0:
        grid_vals[rho_grid < rho_min] = float(val_s[0])
    if rho_max < 1:
        grid_vals[rho_grid > rho_max] = float(val_s[-1])

    return grid_vals, grid_mask


# ===========================================================================
# CNN-1D ResNet Architecture (pure ConvNet, no DeepONet components)
# ===========================================================================

class ResidualBlock1D(nn.Module):
    """1D residual block: Conv → ReLU → Conv → +residual → ReLU.

    No BatchNorm — 1D profile data varies systematically from core to edge,
    making channel-wise statistics across spatial positions unstable.
    """

    def __init__(self, channels, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + residual)


class ProfileCNN(nn.Module):
    """Pure 1D CNN for profile fitting — independent control group.

    Input:  (B, 2, 201) — [interpolated_values, validity_mask]
    Output: (B, 201)    — refined profile (Softplus positivity)

    Architecture (fully convolutional, no DeepONet branch/trunk):
      Encoder:  Conv1d(2→32, k7) → ReLU
      ResBlocks: 4× ResidualBlock1D(32, k7)
      Decoder:   Conv1d(32→32→1, k7) → Softplus

    ~27K parameters — compact, faster than ProfileNet's ~58K.
    """

    def __init__(self, in_channels=2, base_channels=32, num_res_blocks=4,
                 kernel_size=7, num_output=GRID_SIZE):
        super().__init__()
        padding = kernel_size // 2

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size, padding=padding),
            nn.ReLU(),
        )

        self.res_blocks = nn.Sequential(*[
            ResidualBlock1D(base_channels, kernel_size)
            for _ in range(num_res_blocks)
        ])

        self.decoder = nn.Sequential(
            nn.Conv1d(base_channels, base_channels, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(base_channels, 1, kernel_size, padding=padding),
        )

    def forward(self, x):
        """x: (B, 2, 201) → y: (B, 201)"""
        out = self.encoder(x)
        out = self.res_blocks(out)
        out = self.decoder(out).squeeze(1)  # (B, 201)
        return F.softplus(out)


class ProfileCNN_Ti(nn.Module):
    """Pure 1D CNN for Ti with dual input (TXCS + Te pedestal).

    Two parallel ResNet encoders, fused by concatenation then decoded.
    Fully convolutional — no DeepONet fusion MLP or CoordDecoder.

    Input:
      x_main: (B, 2, 201) — [interpolated_Ti, Ti_mask]
      x_ped:  (B, 2, 201) — [interpolated_Te_ped, Te_ped_mask]
    Output: (B, 201) — Ti profile [keV]

    ~42K parameters.
    """

    def __init__(self, base_channels=32, num_res_blocks=3,
                 kernel_size=7, num_output=GRID_SIZE):
        super().__init__()
        padding = kernel_size // 2

        # Main branch (TXCS Ti)
        self.main_enc = nn.Sequential(
            nn.Conv1d(2, base_channels, kernel_size, padding=padding),
            nn.ReLU(),
        )
        self.main_res = nn.Sequential(*[
            ResidualBlock1D(base_channels, kernel_size)
            for _ in range(num_res_blocks)
        ])

        # Pedestal branch (Te pedestal for Ti boundary constraint)
        self.ped_enc = nn.Sequential(
            nn.Conv1d(2, base_channels, kernel_size, padding=padding),
            nn.ReLU(),
        )
        self.ped_res = nn.Sequential(*[
            ResidualBlock1D(base_channels, kernel_size)
            for _ in range(num_res_blocks)
        ])

        # Fusion + decoder
        fused_channels = base_channels * 2
        self.fusion = nn.Sequential(
            nn.Conv1d(fused_channels, fused_channels, kernel_size, padding=padding),
            nn.ReLU(),
            ResidualBlock1D(fused_channels, kernel_size),
            nn.Conv1d(fused_channels, base_channels, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(base_channels, 1, kernel_size, padding=padding),
        )

    def forward(self, x_main, x_ped):
        f_main = self.main_res(self.main_enc(x_main))  # (B, C, L)
        f_ped = self.ped_res(self.ped_enc(x_ped))      # (B, C, L)
        fused = torch.cat([f_main, f_ped], dim=1)       # (B, 2C, L)
        out = self.fusion(fused).squeeze(1)              # (B, L)
        return F.softplus(out)


# ===========================================================================
# Grid Dataset — wraps ProfileDataset
# ===========================================================================

class GridDataset(Dataset):
    """Wraps ProfileDataset: scattered points → (2, 201) grid tensors."""

    def __init__(self, data_dir, datatype, split='train',
                 split_ratio=(0.7, 0.15, 0.15), grid_size=GRID_SIZE,
                 split_method='random', plasma_mode='all'):
        if split_method == 'stratified':
            self.base_ds = StratifiedProfileDataset(
                data_dir, datatype, split, split_ratio, plasma_mode)
        else:
            self.base_ds = ProfileDataset(data_dir, datatype, split, split_ratio)
        self.grid_size = grid_size
        self.is_ti = (datatype == 'Ti')

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        item = self.base_ds[idx]

        # Main input: scattered → grid
        X = item['X'].numpy()
        rho, val = X[:, 0], X[:, 1]
        grid_val, grid_mask = scatter_to_grid(rho, val, self.grid_size)
        x_main = np.stack([grid_val, grid_mask], axis=0)  # (2, L)

        if self.is_ti:
            X_ped = item['X_ped'].numpy()
            ped_rho, ped_val = X_ped[:, 0], X_ped[:, 1]
            ped_gval, ped_gmask = scatter_to_grid(ped_rho, ped_val, self.grid_size)
            x_ped = np.stack([ped_gval, ped_gmask], axis=0)
            return {
                'x_main': torch.from_numpy(x_main),
                'x_ped': torch.from_numpy(x_ped),
                'Y': item['Y'],
            }
        else:
            return {
                'x_main': torch.from_numpy(x_main),
                'Y': item['Y'],
            }


def collate_cnn(batch):
    """Collate grid tensors into batched tensors."""
    is_ti = 'x_ped' in batch[0]
    x_main = torch.stack([item['x_main'] for item in batch])  # (B, 2, L)
    Y = torch.stack([item['Y'] for item in batch])            # (B, L)

    if is_ti:
        x_ped = torch.stack([item['x_ped'] for item in batch])
        return x_main, x_ped, Y
    else:
        return x_main, Y


# ===========================================================================
# Training
# ===========================================================================

def train_cnn_model(datatype, data_dir, output_dir, epochs=500, batch_size=64,
                    lr=1e-3, w_mono=0.05, w_bdy=0.05, w_smooth=0.01, w_log=0.1,
                    patience=100, device='cpu',
                    split_method='random', plasma_mode='all'):
    """Train a CNN-1D ResNet baseline model."""

    print(f"\n{'='*60}")
    print(f"Training CNN-1D ResNet Baseline — {datatype}")
    print(f"{'='*60}")

    train_ds = GridDataset(data_dir, datatype, split='train',
                            split_method=split_method, plasma_mode=plasma_mode)
    val_ds = GridDataset(data_dir, datatype, split='val',
                          split_method=split_method, plasma_mode=plasma_mode)
    print(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")
    print(f"Architecture: Pure ResNet encoder-decoder (no DeepONet components)")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_cnn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_cnn)

    is_ti = (datatype == 'Ti')
    if is_ti:
        model = ProfileCNN_Ti(base_channels=32, num_output=GRID_SIZE).to(device)
    else:
        model = ProfileCNN(in_channels=2, base_channels=32, num_output=GRID_SIZE).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    criterion = PhysicsConstrainedLoss(w_mono=w_mono, w_bdy=w_bdy, w_smooth=w_smooth, w_log=w_log)
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
    save_path = os.path.join(output_dir, f'CNN_{datatype}.pt')
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


def _load_cnn_model(datatype, model_dir=None):
    """Load trained CNN-1D ResNet model from disk. Cached per process."""
    if datatype in _cache:
        return _cache[datatype]

    if model_dir is None:
        model_dir = _model_dir_default

    model_path = os.path.join(model_dir, f'CNN_{datatype}.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"CNN model not found: {model_path}\n"
            f"Train first: python -m profile_nn.cnn_baseline --data <dir> --datatype {datatype}"
        )

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    cfg = checkpoint.get('config', {})
    base_channels = cfg.get('base_channels', 32)

    if datatype == 'Ti':
        model = ProfileCNN_Ti(base_channels=base_channels)
    else:
        model = ProfileCNN(in_channels=2, base_channels=base_channels)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    _cache[datatype] = model
    return model


def cnn_fit_te(rho_scattered, te_scattered, model_dir=None):
    """CNN-1D ResNet baseline for Te profile fitting.

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
    x = torch.from_numpy(np.stack([grid_val, grid_mask])).unsqueeze(0)  # (1, 2, 201)

    model = _load_cnn_model('Te', model_dir)
    with torch.no_grad():
        y = model(x).squeeze(0).cpu().numpy().astype(np.float64)

    rho_201 = np.linspace(0, 1, 201, dtype=np.float64)

    # PCHIP monotonicity safety net
    try:
        from scipy.interpolate import PchipInterpolator
        pchip = PchipInterpolator(rho_201, -y)
        y = np.maximum(-pchip(rho_201), 0.005)
        y = np.minimum.accumulate(y)
    except Exception:
        pass

    # Edge floor protection
    y = np.maximum(y, 0.005)
    y = np.minimum.accumulate(y)
    return rho_201, y


def cnn_fit_ne(rho_scattered, ne_scattered, model_dir=None):
    """CNN-1D ResNet baseline for ne profile fitting.

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

    model = _load_cnn_model('ne', model_dir)
    with torch.no_grad():
        y = model(x).squeeze(0).cpu().numpy().astype(np.float64)

    rho_201 = np.linspace(0, 1, 201, dtype=np.float64)

    try:
        from scipy.interpolate import PchipInterpolator
        pchip = PchipInterpolator(rho_201, -y)
        y = np.maximum(-pchip(rho_201), 0.001)
        y = np.minimum.accumulate(y)
    except Exception:
        pass

    # Edge floor protection
    y = np.maximum(y, 0.001)
    y = np.minimum.accumulate(y)
    return rho_201, y


def cnn_fit_ti(rho_scattered, ti_scattered, te_ped_x, te_ped_y, model_dir=None):
    """CNN-1D ResNet baseline for Ti profile fitting.

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

    model = _load_cnn_model('Ti', model_dir)
    with torch.no_grad():
        y = model(x_main, x_ped).squeeze(0).cpu().numpy().astype(np.float64)

    rho_201 = np.linspace(0, 1, 201, dtype=np.float64)

    try:
        from scipy.interpolate import PchipInterpolator
        pchip = PchipInterpolator(rho_201, -y)
        y = np.maximum(-pchip(rho_201), 0.01)
        y = np.minimum.accumulate(y)
    except Exception:
        pass

    # Edge floor protection
    y = np.maximum(y, 0.005)
    y = np.minimum.accumulate(y)
    return rho_201, y


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train CNN-1D ResNet baseline models (standalone architecture)')
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
            train_cnn_model(
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

    print(f"\nCNN-1D models saved in {output_dir.absolute()}")


if __name__ == '__main__':
    main()
