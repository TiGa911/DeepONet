"""Train ProfileNet models with physics-constrained loss.

Usage:
    python -m profile_nn.train --data ./profile_nn_data --datatype Te --epochs 500
    python -m profile_nn.train --data ./profile_nn_data --datatype ne
    python -m profile_nn.train --data ./profile_nn_data --datatype Ti
    python -m profile_nn.train --data ./profile_nn_data --datatype all   # train all three
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profile_nn.model import (
    ProfileNet_Te, ProfileNet_ne, ProfileNet_Ti, PhysicsConstrainedLoss,
)
from profile_nn.dataset import ProfileDataset, collate_fn


def train_one_model(datatype, data_dir, output_dir, epochs=500, batch_size=64,
                    lr=1e-3, w_mono=0.1, w_bdy=0.5, w_smooth=0.05,
                    patience=50, device='cpu'):
    """Train a single ProfileNet model for one diagnostic type."""

    print(f"\n{'='*60}")
    print(f"Training ProfileNet_{datatype}")
    print(f"{'='*60}")

    # Datasets
    train_ds = ProfileDataset(data_dir, datatype, split='train')
    val_ds = ProfileDataset(data_dir, datatype, split='val')
    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn)

    # Model
    is_ti = (datatype == 'Ti')
    if datatype == 'Te':
        model = ProfileNet_Te(hidden=128, num_output=201).to(device)
    elif datatype == 'ne':
        model = ProfileNet_ne(hidden=128, num_output=201).to(device)
    else:
        model = ProfileNet_Ti(hidden=128, num_output=201).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss & optimizer
    criterion = PhysicsConstrainedLoss(w_mono=w_mono, w_bdy=w_bdy, w_smooth=w_smooth)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train': [], 'val': []}

    for epoch in range(epochs):
        # --- Train ---
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

        # --- Validate ---
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

        # Aggregate
        def avg_key(dicts, key):
            return np.mean([d[key] for d in dicts])

        train_avg = {k: avg_key(train_losses, k) for k in train_losses[0]}
        val_avg = {k: avg_key(val_losses, k) for k in val_losses[0]}
        history['train'].append(train_avg)
        history['val'].append(val_avg)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"train MSE={train_avg['mse']:.4e} mono={train_avg['mono']:.4e} | "
                  f"val MSE={val_avg['mse']:.4e} mono={val_avg['mono']:.4e}")

        # Early stopping
        val_total = val_avg['total']
        if val_total < best_val_loss - 1e-6:
            best_val_loss = val_total
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Save best model
    model.load_state_dict(best_state)
    save_path = os.path.join(output_dir, f'ProfileNet_{datatype}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'datatype': datatype,
        'history': history,
        'config': {'hidden': 128, 'num_output': 201, 'w_mono': w_mono, 'w_bdy': w_bdy, 'w_smooth': w_smooth},
    }, save_path)
    print(f"Model saved to {save_path}")
    print(f"Best val loss: {best_val_loss:.4e}")

    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train ProfileNet models')
    parser.add_argument('--data', type=str, required=True,
                        help='Directory containing training .npz files')
    parser.add_argument('--datatype', type=str, default='all',
                        choices=['Te', 'ne', 'Ti', 'all'],
                        help='Which diagnostic model to train')
    parser.add_argument('--output', type=str, default='./profile_nn_models',
                        help='Directory to save trained models')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--w-mono', type=float, default=0.1,
                        help='Weight for monotonicity loss')
    parser.add_argument('--w-bdy', type=float, default=0.5,
                        help='Weight for boundary derivative loss')
    parser.add_argument('--w-smooth', type=float, default=0.05,
                        help='Weight for 2nd derivative smoothness loss')
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
                device=device,
            )
        except FileNotFoundError as e:
            print(f"Skipping {dt}: {e}")
        except Exception as e:
            print(f"Error training {dt}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nModels saved in {output_dir.absolute()}")


if __name__ == '__main__':
    main()
