"""LSTM baseline model for profile fitting — second control group for ProfileNet.

Architecture:
  - Scattered (rho, val) points sorted by rho → BiLSTM sequence encoder
  - Same CoordDecoder + dot product decoder as ProfileNet
  - Same PhysicsConstrainedLoss for fair comparison

Key difference from ProfileNet:
  - ProfileNet: SetEncoder (MLP per-point + max pool) — permutation invariant
  - LSTM:      BiLSTM over sorted sequence — order-aware, captures radial structure

Usage:
  python -m profile_nn.lstm_baseline --data ./profile_nn_data --datatype all --epochs 500
"""

import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profile_nn.dataset import ProfileDataset, collate_fn
from profile_nn.model import PhysicsConstrainedLoss, CoordDecoder


# ===========================================================================
# BiLSTM Encoder
# ===========================================================================

class LSTMEncoder(nn.Module):
    """BiLSTM encoder: sorted (rho, val) sequence → latent vector z.

    Points are sorted by rho within the forward pass to form a radial sequence
    from magnetic axis (ρ≈0) to edge (ρ≈1). A 2-layer BiLSTM processes the
    sequence, and the final forward+backward hidden states are concatenated
    and projected to the latent dimension.

    Input:  (B, N, 2) — padded (rho, val), unsorted
    Output: (B, hidden)
    """

    def __init__(self, input_dim=2, hidden=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden, num_layers,
            batch_first=True, bidirectional=True,
        )
        self.proj = nn.Linear(hidden * 2, hidden)  # bi-directional → hidden
        self.hidden = hidden
        self.num_layers = num_layers

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

        # Sort each sample's points by rho (channel 0)
        # Use mask to avoid sorting padding zeros
        x_sorted = torch.zeros_like(x)
        for b in range(B):
            n = lengths[b].item()
            if n == 0:
                continue
            pts = x[b, :n]  # valid points
            sort_idx = torch.argsort(pts[:, 0])  # sort by rho
            x_sorted[b, :n] = pts[sort_idx]

        # Pack sorted sequences
        # Need lengths sorted descending for pack_padded_sequence
        lengths_sorted, perm_idx = lengths.sort(descending=True)
        x_sorted = x_sorted[perm_idx]
        mask_sorted = mask[perm_idx]

        # Remove zero-length sequences (shouldn't happen but be safe)
        valid_mask_len = lengths_sorted > 0
        if not valid_mask_len.all():
            lengths_sorted = lengths_sorted[valid_mask_len]
            x_sorted = x_sorted[valid_mask_len]
            perm_idx = perm_idx[valid_mask_len]

        if lengths_sorted[0].item() == 0:
            return torch.zeros(B, self.hidden, device=x.device)

        try:
            packed = pack_padded_sequence(
                x_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True
            )
            _, (h_n, _) = self.lstm(packed)
            # h_n: (num_layers*2, B_eff, hidden)
            # Last layer: forward at index -2, backward at index -1
            h_forward = h_n[-2]   # (B_eff, hidden)
            h_backward = h_n[-1]  # (B_eff, hidden)
            z_eff = self.proj(torch.cat([h_forward, h_backward], dim=-1))

            # Restore original batch order
            z = torch.zeros(B, self.hidden, device=x.device)
            z[perm_idx] = z_eff
        except Exception:
            z = torch.zeros(B, self.hidden, device=x.device)

        return z


# ===========================================================================
# LSTM-based Profile Models
# ===========================================================================

class BaseLSTMProfileNet(nn.Module):
    """Shared skeleton: LSTM encoder → CoordDecoder → dot product → Softplus."""

    def __init__(self, hidden=64, lstm_layers=1, num_output=201):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim=2, hidden=hidden, num_layers=lstm_layers)
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


class LSTMProfileNet_Te(BaseLSTMProfileNet):
    """Te from TS via LSTM."""
    pass


class LSTMProfileNet_ne(BaseLSTMProfileNet):
    """ne from Refl via LSTM."""
    pass


class LSTMProfileNet_Ti(nn.Module):
    """Ti via dual BiLSTM (TXCS + Te pedestal)."""

    def __init__(self, hidden=64, lstm_layers=1, num_output=201):
        super().__init__()
        self.encoder_main = LSTMEncoder(input_dim=2, hidden=hidden, num_layers=lstm_layers)
        self.encoder_ped = LSTMEncoder(input_dim=2, hidden=hidden, num_layers=lstm_layers)
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
        z_main = self.encoder_main(x_main, mask_main)  # (B, hidden)
        z_ped = self.encoder_ped(x_ped, mask_ped)      # (B, hidden)
        z = self.fusion(torch.cat([z_main, z_ped], dim=-1))
        rho = self.rho_grid.unsqueeze(0).expand(B, -1)
        basis = self.decoder(rho)
        y = (z.unsqueeze(1) * basis).sum(dim=-1)
        y = F.softplus(y)
        return y


# ===========================================================================
# Training
# ===========================================================================

def train_lstm_model(datatype, data_dir, output_dir, epochs=500, batch_size=64,
                     lr=1e-3, w_mono=0.05, w_bdy=0.05, w_smooth=0.01,
                     patience=100, device='cpu'):
    """Train a BiLSTM baseline model."""

    print(f"\n{'='*60}")
    print(f"Training LSTM Baseline — {datatype}")
    print(f"{'='*60}")

    # Datasets (same split as ProfileNet)
    train_ds = ProfileDataset(data_dir, datatype, split='train')
    val_ds = ProfileDataset(data_dir, datatype, split='val')
    print(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn)

    # Model
    is_ti = (datatype == 'Ti')
    if is_ti:
        model = LSTMProfileNet_Ti(hidden=64, num_output=201).to(device)
    else:
        model = BaseLSTMProfileNet(hidden=64, num_output=201).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Loss & optimizer
    criterion = PhysicsConstrainedLoss(w_mono=w_mono, w_bdy=w_bdy, w_smooth=w_smooth)
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
    save_path = os.path.join(output_dir, f'LSTM_{datatype}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'datatype': datatype,
        'history': history,
        'config': {
            'hidden': 64, 'num_output': 201, 'lstm_layers': 1,
            'w_mono': w_mono, 'w_bdy': w_bdy, 'w_smooth': w_smooth,
            'n_params': n_params,
        },
    }, save_path)
    print(f"Model saved → {save_path}")
    print(f"Best val loss: {best_val_loss:.4e}")

    return model, history


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='Train LSTM baseline models')
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
            train_lstm_model(
                datatype=dt, data_dir=args.data, output_dir=str(output_dir),
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                w_mono=args.w_mono, w_bdy=args.w_bdy, w_smooth=args.w_smooth,
                device=device,
            )
        except FileNotFoundError as e:
            print(f"Skipping {dt}: {e}")
        except Exception as e:
            print(f"Error training {dt}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nLSTM models saved in {output_dir.absolute()}")


if __name__ == '__main__':
    main()
