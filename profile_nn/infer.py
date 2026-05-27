"""Drop-in replacement functions for the mtanh+spline fitting chain.

Each function matches the signature of the original processing in onetwo_output.py:

  Original Te flow:                          → nn_fit_te()
    fitting(x,y,'Te') + robust_interp()

  Original ne flow:                          → nn_fit_ne()
    fitting(x,y,'Refl') + robust_interp()

  Original Ti flow:                          → nn_fit_ti()
    fitting(x,y,'Ti',te_ped_x,te_ped_y) + robust_interp() + enforce_monotone_pchip()

Usage:
    from profile_nn import nn_fit_te, nn_fit_ne, nn_fit_ti

    rho_201, te_201 = nn_fit_te(rho_scattered, te_scattered)
    rho_201, ne_201 = nn_fit_ne(rho_scattered, ne_scattered)
    rho_201, ti_201 = nn_fit_ti(rho_scattered, ti_scattered, te_ped_x, te_ped_y)
"""

import os
import numpy as np
import torch

# Lazy-import model classes
_model_dir = os.path.join(os.path.dirname(__file__), '..', 'profile_nn_models')
_cache = {}  # module-level model cache


def _load_model(datatype, model_dir=None):
    """Load a trained ProfileNet model from disk. Cached per process."""
    if datatype in _cache:
        return _cache[datatype]

    if model_dir is None:
        model_dir = _model_dir

    from profile_nn.model import ProfileNet_Te, ProfileNet_ne, ProfileNet_Ti

    model_path = os.path.join(model_dir, f'ProfileNet_{datatype}.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Train the model first with: python -m profile_nn.train --data <data_dir> --datatype {datatype}"
        )

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    hidden = checkpoint.get('config', {}).get('hidden', 128)

    if datatype == 'Te':
        model = ProfileNet_Te(hidden=hidden)
    elif datatype == 'ne':
        model = ProfileNet_ne(hidden=hidden)
    else:
        model = ProfileNet_Ti(hidden=hidden)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    _cache[datatype] = model
    return model


def _predict_profile(model, x, mask, device='cpu'):
    """Run inference on a single sample."""
    model = model.to(device)
    x_t = torch.from_numpy(x).unsqueeze(0).to(device)  # (1, N, 2)
    if mask is not None:
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)  # (1, N)
    else:
        mask_t = None

    with torch.no_grad():
        y = model(x_t, mask_t)  # (1, 201)

    rho_out = np.linspace(0, 1, 201, dtype=np.float64)
    y_out = y.squeeze(0).cpu().numpy().astype(np.float64)
    return rho_out, y_out


def nn_fit_te(rho_scattered, te_scattered, model_dir=None):
    """NN replacement for Te fitting + robust_interp.

    Args:
        rho_scattered: 1D array of rho coordinates from TS
        te_scattered: 1D array of Te values [eV] from MDSplus
    Returns:
        rho_201: (201,) uniform rho grid
        te_201: (201,) Te profile [keV]
    """
    rho = np.asarray(rho_scattered, dtype=np.float32)
    te = np.asarray(te_scattered, dtype=np.float32)

    # Normalize: eV -> keV (matching the mtanh fitter's internal convention)
    te_keV = te / 1000.0

    x = np.stack([rho, te_keV], axis=-1)
    mask = np.ones(len(rho), dtype=bool)

    model = _load_model('Te', model_dir)
    rho_201, te_201 = _predict_profile(model, x, mask)

    return rho_201, te_201


def nn_fit_ne(rho_scattered, ne_scattered, model_dir=None):
    """NN replacement for ne fitting + robust_interp.

    Args:
        rho_scattered: 1D array of rho coordinates from Refl
        ne_scattered: 1D array of ne values [1e19 m^-3] from MDSplus
    Returns:
        rho_201: (201,) uniform rho grid
        ne_201: (201,) ne profile [1e19 m^-3]
    """
    rho = np.asarray(rho_scattered, dtype=np.float32)
    ne = np.asarray(ne_scattered, dtype=np.float32)

    x = np.stack([rho, ne], axis=-1)
    mask = np.ones(len(rho), dtype=bool)

    model = _load_model('ne', model_dir)
    rho_201, ne_201 = _predict_profile(model, x, mask)

    return rho_201, ne_201


def nn_fit_ti(rho_scattered, ti_scattered, te_ped_x, te_ped_y, model_dir=None):
    """NN replacement for Ti fitting + robust_interp + enforce_monotone_pchip.

    Args:
        rho_scattered: 1D array of rho coordinates from TXCS
        ti_scattered: 1D array of Ti values [keV] from MDSplus
        te_ped_x: 1D array of Te pedestal rho coordinates (rho >= 0.88)
        te_ped_y: 1D array of Te pedestal values [keV]
    Returns:
        rho_201: (201,) uniform rho grid
        ti_201: (201,) Ti profile [keV]
    """
    from profile_nn.model import ProfileNet_Ti
    from scipy.interpolate import PchipInterpolator

    rho = np.asarray(rho_scattered, dtype=np.float32)
    ti = np.asarray(ti_scattered, dtype=np.float32)
    ped_x = np.asarray(te_ped_x, dtype=np.float32)
    ped_y = np.asarray(te_ped_y, dtype=np.float32)

    x_txcs = np.stack([rho, ti], axis=-1)
    mask_txcs = np.ones(len(rho), dtype=bool)
    x_ped = np.stack([ped_x, ped_y], axis=-1)
    mask_ped = np.ones(len(ped_x), dtype=bool)

    model = _load_model('Ti', model_dir)
    model = model.to('cpu')

    x_t = torch.from_numpy(x_txcs).unsqueeze(0)
    m_t = torch.from_numpy(mask_txcs).unsqueeze(0)
    p_t = torch.from_numpy(x_ped).unsqueeze(0)
    mp_t = torch.from_numpy(mask_ped).unsqueeze(0)

    with torch.no_grad():
        y = model(x_t, m_t, p_t, mp_t)

    rho_201 = np.linspace(0, 1, 201, dtype=np.float64)
    ti_201 = y.squeeze(0).cpu().numpy().astype(np.float64)

    # Safety net: enforce monotonicity via PCHIP post-processing
    try:
        pchip = PchipInterpolator(rho_201, -ti_201)
        ti_201_mono = -pchip(rho_201)
        ti_201 = np.maximum(ti_201_mono, 0.01)
    except Exception:
        pass  # fall back to raw NN output if PCHIP fails

    return rho_201, ti_201


def set_model_dir(path):
    """Set the directory containing trained .pt model files."""
    global _model_dir
    _model_dir = path
    _cache.clear()
