"""DeepONet-style set encoder + coordinate decoder for profile fitting.

Three model variants:
  - ProfileNet_Te: scattered TS points -> Te 201-pt profile
  - ProfileNet_ne: scattered Refl points -> ne 201-pt profile
  - ProfileNet_Ti: scattered TXCS points + Te pedestal -> Ti 201-pt profile

Architecture:
  SetEncoder: per-point MLP(2->64->128->128) + global max pool -> latent z
  CoordDecoder: MLP(1->128->128) on rho -> basis functions
  Output: dot product of latent z and basis b(rho) at 201 grid points

Total params: ~50K per model, trainable on CPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared modules
# ---------------------------------------------------------------------------

class SetEncoder(nn.Module):
    """PointNet-style encoder for variable-length set of (rho, y) pairs."""

    def __init__(self, input_dim=2, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, N, 2)  — padded (rho, y) points
            mask: (B, N)  — True where point is valid
        Returns:
            z: (B, hidden)
        """
        feat = self.mlp(x)  # (B, N, hidden)
        if mask is not None:
            feat = feat * mask.unsqueeze(-1).float()
            # Set masked positions to large negative so max pooling ignores them
            feat = feat.masked_fill(~mask.unsqueeze(-1), -1e9)
        z, _ = feat.max(dim=1)  # (B, hidden)
        return z


class CoordDecoder(nn.Module):
    """Trunk net: maps scalar rho -> basis functions b(rho)."""

    def __init__(self, hidden=128, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, rho_grid):
        """
        Args:
            rho_grid: (201,) or (B, 201) — target rho coordinates
        Returns:
            basis: (201, output_dim) or (B, 201, output_dim)
        """
        single = rho_grid.dim() == 1
        if single:
            rho_grid = rho_grid.unsqueeze(-1)  # (201,) -> (201, 1)
        else:
            rho_grid = rho_grid.unsqueeze(-1)  # (B, 201) -> (B, 201, 1)
        basis = self.net(rho_grid)  # (B, 201, hidden)
        return basis


# ---------------------------------------------------------------------------
# Base profile network
# ---------------------------------------------------------------------------

class BaseProfileNet(nn.Module):
    """Shared skeleton: set encoder -> coord decoder -> dot product -> softplus."""

    def __init__(self, hidden=128, num_output=201):
        super().__init__()
        self.encoder = SetEncoder(input_dim=2, hidden=hidden)
        self.decoder = CoordDecoder(hidden=hidden, output_dim=hidden)
        self.hidden = hidden
        self.num_output = num_output
        self.register_buffer("rho_grid", torch.linspace(0, 1, num_output))

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, N, 2)  — padded (rho, y) scattered points
            mask: (B, N)  — True for valid points
        Returns:
            y: (B, num_output) — predicted profile
        """
        B = x.shape[0]
        z = self.encoder(x, mask)  # (B, hidden)

        rho = self.rho_grid.unsqueeze(0).expand(B, -1)  # (B, num_output)
        basis = self.decoder(rho)  # (B, num_output, hidden)

        y = (z.unsqueeze(1) * basis).sum(dim=-1)  # (B, num_output)

        # Softplus ensures positivity (Te/ne/Ti all > 0 physically)
        y = F.softplus(y)
        return y


# ---------------------------------------------------------------------------
# Diagnostic-specific models
# ---------------------------------------------------------------------------

class ProfileNet_Te(BaseProfileNet):
    """Te profile from TS scattered points.

    Input:  (B, N_ts, 2)  — (rho, Te) pairs from Thomson scattering
    Output: (B, 201)       — Te profile on uniform rho grid [keV]
    """
    pass


class ProfileNet_ne(BaseProfileNet):
    """ne profile from Refl scattered points.

    Input:  (B, N_refl, 2) — (rho, ne) pairs from reflectometer
    Output: (B, 201)       — ne profile on uniform rho grid [1e19 m^-3]
    """
    pass


class ProfileNet_Ti(BaseProfileNet):
    """Ti profile from TXCS scattered points + Te pedestal.

    The Te pedestal provides boundary information for the Ti profile
    since TXCS data is typically sparse (3-15 points).

    Input:  (B, N_txcs, 2) — (rho, Ti) pairs from TXCS
            (B, N_ped, 2)  — (rho, Te) pairs from Te pedestal (rho >= 0.88)
    Output: (B, 201)       — Ti profile on uniform rho grid [keV]
    """

    def __init__(self, hidden=128, num_output=201):
        super().__init__(hidden, num_output)
        # Second encoder for Te pedestal branch
        self.pedestal_encoder = SetEncoder(input_dim=2, hidden=hidden)
        # Fusion MLP to combine both latent vectors
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, x_txcs, mask_txcs, x_ped, mask_ped):
        """Forward pass with dual encoder branches.

        Args:
            x_txcs:  (B, N_txcs, 2) — TXCS (rho, Ti) points
            mask_txcs: (B, N_txcs)
            x_ped:   (B, N_ped, 2)  — Te pedestal (rho, Te) points
            mask_ped: (B, N_ped)
        Returns:
            y: (B, num_output) — Ti profile [keV]
        """
        B = x_txcs.shape[0]
        z_txcs = self.encoder(x_txcs, mask_txcs)    # (B, hidden)
        z_ped = self.pedestal_encoder(x_ped, mask_ped)  # (B, hidden)
        z = self.fusion(torch.cat([z_txcs, z_ped], dim=-1))  # (B, hidden)

        rho = self.rho_grid.unsqueeze(0).expand(B, -1)
        basis = self.decoder(rho)  # (B, num_output, hidden)

        y = (z.unsqueeze(1) * basis).sum(dim=-1)
        y = F.softplus(y)
        return y


# ---------------------------------------------------------------------------
# Physics-constrained loss
# ---------------------------------------------------------------------------

class PhysicsConstrainedLoss(nn.Module):
    """Weighted sum of MSE + physics regularizers.

    L = L_MSE + w_mono * L_monotonicity + w_bdy * L_boundary + w_smooth * L_smoothness
    """

    def __init__(self, w_mono=0.1, w_bdy=0.5, w_smooth=0.05):
        super().__init__()
        self.w_mono = w_mono
        self.w_bdy = w_bdy
        self.w_smooth = w_smooth
        self.mse = nn.MSELoss()
        self.rho_grid = torch.linspace(0, 1, 201)

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: (B, 201) — predicted profiles
            y_true: (B, 201) — ground truth profiles
        Returns:
            total_loss, loss_dict
        """
        device = y_pred.device
        rho = self.rho_grid.to(device)
        drho = rho[1] - rho[0]

        # Data fidelity
        loss_mse = self.mse(y_pred, y_true)

        # Monotonicity: penalize positive derivatives (y should decrease with rho)
        dy = torch.diff(y_pred, dim=1) / drho  # (B, 200)
        loss_mono = F.relu(dy).mean()  # only penalize INCREASING segments

        # Boundary: derivative ~ 0 near rho=0 (magnetic axis symmetry)
        boundary_idx = (rho < 0.05).sum().item()
        dy_boundary = torch.diff(y_pred[:, :boundary_idx], dim=1) / drho
        loss_bdy = (dy_boundary ** 2).mean()

        # Smoothness: penalize large 2nd derivatives
        d2y = torch.diff(y_pred, n=2, dim=1) / (drho ** 2)
        loss_smooth = d2y.abs().mean()

        total = (loss_mse
                 + self.w_mono * loss_mono
                 + self.w_bdy * loss_bdy
                 + self.w_smooth * loss_smooth)

        loss_dict = {
            'mse': loss_mse.item(),
            'mono': loss_mono.item(),
            'bdy': loss_bdy.item(),
            'smooth': loss_smooth.item(),
            'total': total.item(),
        }
        return total, loss_dict
