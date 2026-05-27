"""ProfileDataset: loads .npz training pairs, handles variable-length padding."""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class ProfileDataset(Dataset):
    """Dataset of (scattered diagnostic points, fitted profile) pairs.

    Each .npz file contains:
      X_rho, X_val  — scattered rho & y values (variable length)
      Y             — 201-point fitted profile (the mtanh label)
      For Ti samples only: X_te_ped_rho, X_te_ped_val

    File naming convention: {shot}_{datatype}_{time_dir}.npz
    """

    def __init__(self, data_dir, datatype, split='train', split_ratio=(0.7, 0.15, 0.15)):
        """
        Args:
            data_dir: path to directory containing .npz files
            datatype: 'Te', 'ne', or 'Ti'
            split: 'train', 'val', or 'test'
            split_ratio: (train, val, test) fractions
        """
        self.data_dir = Path(data_dir)
        self.datatype = datatype

        # Gather all .npz files for this datatype
        pattern = f'*_{datatype}_*.npz'
        all_files = sorted(self.data_dir.glob(pattern))

        if not all_files:
            raise FileNotFoundError(
                f"No .npz files found in {data_dir} matching pattern '{pattern}'"
            )

        # Deterministic split by filename hash (avoids temporal leakage)
        np.random.seed(42)
        indices = np.random.permutation(len(all_files))
        n = len(all_files)
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
        Y = data['Y'].astype(np.float32)

        # Build (N, 2) point set
        X = np.stack([X_rho, X_val], axis=-1)  # (N, 2)

        if self.is_ti:
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
    """Custom collate: pad variable-length point sets to batch max.

    Returns:
        X_padded:    (B, N_max, 2)  with zeros for padding
        mask:        (B, N_max)     True = valid point
        Y:           (B, 201)
        (X_ped_padded, mask_ped): only for Ti batches
    """
    is_ti = 'X_ped' in batch[0]

    # Find max point count in this batch
    N_max = max(item['X'].shape[0] for item in batch)
    B = len(batch)

    # Pad main inputs
    X_padded = torch.zeros(B, N_max, 2)
    mask = torch.zeros(B, N_max, dtype=torch.bool)
    Y = torch.stack([item['Y'] for item in batch])

    for i, item in enumerate(batch):
        n = item['X'].shape[0]
        X_padded[i, :n] = item['X']
        mask[i, :n] = True

    if is_ti:
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
