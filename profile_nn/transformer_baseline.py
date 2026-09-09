"""Transformer baseline model for profile fitting — third control group for ProfileNet.

Architecture:
  - Scattered (rho, val) points sorted by rho → Transformer Encoder (self-attention)
  - Same CoordDecoder + dot product decoder as ProfileNet & LSTM
  - Same PhysicsConstrainedLoss for fair comparison

Key difference from other baselines:
  - ProfileNet:  SetEncoder (MLP per-point + max pool) — permutation invariant
  - LSTM:        BiLSTM over sorted sequence — order-aware, captures radial structure
  - Transformer: Self-attention over sorted sequence — global attention, no recurrence

Usage:
  python -m profile_nn.transformer_baseline --data ./profile_nn_data --datatype all --epochs 500
"""

import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profile_nn.dataset import ProfileDataset, StratifiedProfileDataset, collate_fn
from profile_nn.model import PhysicsConstrainedLoss, CoordDecoder


# ===========================================================================
# Transformer Encoder
# ===========================================================================

class TransformerEncoder(nn.Module):
    """Transformer encoder: sorted (rho, val) sequence + positional encoding
       → multi-head self-attention → mean pool → latent vector z.

    Points are sorted by rho to form a radial sequence from core (rho~0) to
    edge (rho~1). Learnable positional encodings capture radial structure.
    Self-attention allows each point to directly attend to any other point —
    capturing global correlations rather than local sequential dependencies.

    Input:  (B, N, 2) — padded (rho, val), will be sorted by rho internally
    Output: (B, hidden)
    """

    def __init__(self, input_dim=2, hidden=64, num_layers=2, num_heads=4,
                 ffn_dim=256, dropout=0.1, max_len=35):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, hidden) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.hidden = hidden
        self.max_len = max_len

    def _get_pos_encoding(self, N, device):
        """Get positional encoding for sequence length N, interpolating if needed."""
        if N <= self.max_len:
            return self.pos_embed[:, :N, :]
        # Interpolate via F.interpolate (treat as 1D spatial grid)
        pos = self.pos_embed.permute(0, 2, 1)  # (1, hidden, max_len)
        pos = F.interpolate(pos, size=N, mode='linear', align_corners=True)
        pos = pos.permute(0, 2, 1)  # (1, N, hidden)
        return pos

    def forward(self, x, mask):
        """
        Args:
            x: (B, N, 2) — padded point set, any order
            mask: (B, N) — True where point is valid
        Returns:
            z: (B, hidden)
        """
        B, N, _ = x.shape
        lengths = mask.sum(dim=1).long()  # (B,)

        # Sort each sample's points by rho (channel 0), subsample if needed
        x_sorted = torch.zeros_like(x)
        mask_updated = mask.clone() if mask is not None else None
        for b in range(B):
            n = lengths[b].item()
            if n == 0:
                continue
            pts = x[b, :n]
            sort_idx = torch.argsort(pts[:, 0])  # sort by rho
            pts = pts[sort_idx]

            # Subsample long sequences to max_len (random subset, keeps first/last)
            if n > self.max_len:
                mid = torch.randperm(n - 2, device=x.device)[:self.max_len - 2] + 1
                keep = torch.cat([torch.zeros(1, device=x.device, dtype=torch.long),
                                  mid,
                                  torch.full((1,), n - 1, device=x.device, dtype=torch.long)])
                keep = torch.unique(keep)
                keep = keep[keep.argsort()]
                keep = keep[:self.max_len]
                pts = pts[keep]
                n = len(pts)
                mask_updated[b, :] = False
                mask_updated[b, :n] = True

            x_sorted[b, :n] = pts
        mask = mask_updated

        # Project to hidden dim
        h = self.input_proj(x_sorted)  # (B, N, hidden)

        # Add positional encoding (supports sequences longer than max_len)
        pos = self._get_pos_encoding(N, h.device)  # (1, N, hidden)
        h = h + pos

        # Build attention mask: True = IGNORE (opposite of our mask convention)
        # src_key_padding_mask: (B, N), True = ignore
        key_padding_mask = ~mask if mask is not None else None

        # Transformer forward
        h = self.transformer(h, src_key_padding_mask=key_padding_mask)  # (B, N, hidden)

        # Mean pool over valid positions (use mask to exclude padding)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()  # (B, N, 1)
            h = h * mask_expanded
            z = h.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)  # (B, hidden)
        else:
            z = h.mean(dim=1)  # (B, hidden)

        return z


# ===========================================================================
# Transformer-based Profile Models
# ===========================================================================

class BaseTransformerProfileNet(nn.Module):
    """Shared skeleton: Transformer encoder → CoordDecoder → dot product → Softplus."""

    def __init__(self, hidden=64, num_layers=2, num_heads=4, ffn_dim=256,
                 dropout=0.1, num_output=201, max_len=64):
        super().__init__()
        self.encoder = TransformerEncoder(
            input_dim=2, hidden=hidden, num_layers=num_layers,
            num_heads=num_heads, ffn_dim=ffn_dim, dropout=dropout,
            max_len=max_len,
        )
        self.decoder = CoordDecoder(hidden=hidden, output_dim=hidden)
        self.hidden = hidden
        self.num_output = num_output
        self.register_buffer("rho_grid", torch.linspace(0, 1, num_output))

    def forward(self, x, mask):
        B = x.shape[0]
        z = self.encoder(x, mask)  # (B, hidden)
        rho = self.rho_grid.unsqueeze(0).expand(B, -1)
        basis = self.decoder(rho)  # (B, num_output, hidden)
        y = (z.unsqueeze(1) * basis).sum(dim=-1)  # (B, num_output)
        y = F.softplus(y)
        return y


class TransformerProfileNet_Te(BaseTransformerProfileNet):
    """Te from TS via Transformer."""
    pass


class TransformerProfileNet_ne(BaseTransformerProfileNet):
    """ne from Refl via Transformer."""
    pass


class TransformerProfileNet_Ti(nn.Module):
    """Ti via dual Transformer (TXCS + Te pedestal)."""

    def __init__(self, hidden=64, num_layers=2, num_heads=4, ffn_dim=256,
                 dropout=0.1, num_output=201, max_len=64):
        super().__init__()
        self.encoder_main = TransformerEncoder(
            input_dim=2, hidden=hidden, num_layers=num_layers,
            num_heads=num_heads, ffn_dim=ffn_dim, dropout=dropout,
            max_len=max_len,
        )
        self.encoder_ped = TransformerEncoder(
            input_dim=2, hidden=hidden, num_layers=num_layers,
            num_heads=num_heads, ffn_dim=ffn_dim, dropout=dropout,
            max_len=max_len,
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.decoder = CoordDecoder(hidden=hidden, output_dim=hidden)
        self.hidden = hidden
        self.num_output = num_output
        self.register_buffer("rho_grid", torch.linspace(0, 1, num_output))

    def forward(self, x_main, mask_main, x_ped, mask_ped):
        B = x_main.shape[0]
        z_main = self.encoder_main(x_main, mask_main)
        z_ped = self.encoder_ped(x_ped, mask_ped)
        z = self.fusion(torch.cat([z_main, z_ped], dim=-1))
        rho = self.rho_grid.unsqueeze(0).expand(B, -1)
        basis = self.decoder(rho)
        y = (z.unsqueeze(1) * basis).sum(dim=-1)
        y = F.softplus(y)
        return y


# ===========================================================================
# Training
# ===========================================================================

def train_transformer_model(datatype, data_dir, output_dir, epochs=500, batch_size=64,
                            lr=1e-3, w_mono=0.05, w_bdy=0.05, w_smooth=0.01, w_log=0.1,
                            patience=100, device='cpu',
                            hidden=64, num_layers=2, num_heads=4, ffn_dim=256,
                            dropout=0.1, max_len=None,
                            split_method='random', plasma_mode='all'):
    """Train a Transformer baseline model."""

    # Auto-select max_len per datatype (covers dataset max sequence lengths)
    if max_len is None:
        max_len_map = {'Te': 35, 'ne': 64, 'Ti': 64}
        max_len = max_len_map.get(datatype, 64)

    print(f"\n{'='*60}")
    print(f"Training Transformer Baseline — {datatype}")
    print(f"{'='*60}")

    # Datasets (same split as ProfileNet/LSTM/CNN)
    if split_method == 'stratified':
        train_ds = StratifiedProfileDataset(data_dir, datatype, split='train',
                                            plasma_mode=plasma_mode)
        val_ds = StratifiedProfileDataset(data_dir, datatype, split='val',
                                          plasma_mode=plasma_mode)
    else:
        train_ds = ProfileDataset(data_dir, datatype, split='train')
        val_ds = ProfileDataset(data_dir, datatype, split='val')
    print(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn)

    # Model
    is_ti = (datatype == 'Ti')
    kwargs = dict(hidden=hidden, num_layers=num_layers, num_heads=num_heads,
                  ffn_dim=ffn_dim, dropout=dropout, num_output=201,
                  max_len=max_len)
    if is_ti:
        model = TransformerProfileNet_Ti(**kwargs).to(device)
    else:
        model = BaseTransformerProfileNet(**kwargs).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"  hidden={hidden}, layers={num_layers}, heads={num_heads}, ffn={ffn_dim}")

    # Loss & optimizer
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
                X, mask, X_ped, mask_ped, Y = [b.to(device) for b in batch]
                y_pred = model(X, mask, X_ped, mask_ped)
            else:
                X, mask, Y = [b.to(device) for b in batch]
                y_pred = model(X, mask)

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
                    X, mask, X_ped, mask_ped, Y = [b.to(device) for b in batch]
                    y_pred = model(X, mask, X_ped, mask_ped)
                else:
                    X, mask, Y = [b.to(device) for b in batch]
                    y_pred = model(X, mask)
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
    save_path = os.path.join(output_dir, f'Transformer_{datatype}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'datatype': datatype,
        'history': history,
        'config': {
            'hidden': hidden, 'num_layers': num_layers, 'num_heads': num_heads,
            'ffn_dim': ffn_dim, 'dropout': dropout, 'num_output': 201,
            'max_len': max_len,
            'w_mono': w_mono, 'w_bdy': w_bdy, 'w_smooth': w_smooth, 'w_log': w_log,
            'n_params': n_params,
        },
    }, save_path)
    print(f"Model saved -> {save_path}")
    print(f"Best val loss: {best_val_loss:.4e}")

    return model, history


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Transformer baseline models')
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
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--ffn-dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max-len', type=int, default=None,
                        help='Max sequence length (auto-detected per datatype if unset)')
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
            train_transformer_model(
                datatype=dt, data_dir=args.data, output_dir=str(output_dir),
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                w_mono=args.w_mono, w_bdy=args.w_bdy, w_smooth=args.w_smooth, w_log=args.w_log,
                patience=50, device=device,
                hidden=args.hidden, num_layers=args.num_layers,
                num_heads=args.num_heads, ffn_dim=args.ffn_dim,
                dropout=args.dropout, max_len=args.max_len,
                split_method=args.split_method, plasma_mode=args.plasma_mode,
            )
        except FileNotFoundError as e:
            print(f"Skipping {dt}: {e}")
        except Exception as e:
            print(f"Error training {dt}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nTransformer models saved in {output_dir.absolute()}")


if __name__ == '__main__':
    main()
