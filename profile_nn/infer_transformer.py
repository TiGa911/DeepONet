"""Transformer Baseline inference module — drop-in replacement for mtanh+spline.

Same API as profile_nn/infer.py (ProfileNet) and profile_nn/infer_lstm.py (LSTM).
Only the underlying model architecture differs.

Each function matches the original fitting chain signatures:

  Original Te flow:                              -> nn_fit_te_transformer()
    fitting(x,y,'TS') + robust_interp()

  Original ne flow:                              -> nn_fit_ne_transformer()
    fitting(x,y,'Refl') + robust_interp()

  Original Ti flow:                              -> nn_fit_ti_transformer()
    fitting(x,y,'Ti',te_ped_x,te_ped_y) + robust_interp() + enforce_monotone_pchip()

Design principles:
  - Input/output units identical to original fitting chain — true drop-in
  - Models auto-loaded from disk and cached in process memory
  - All outputs pass PCHIP monotonicity safety net

Usage:
    from profile_nn.infer_transformer import (
        nn_fit_te_transformer, nn_fit_ne_transformer, nn_fit_ti_transformer,
        set_model_dir_transformer,
    )

    set_model_dir_transformer('./profile_nn_models')

    rho_201, te_201 = nn_fit_te_transformer(rho_ts, te_ev)       # Te: eV in, keV out
    rho_201, ne_201 = nn_fit_ne_transformer(rho_refl, ne_1e19)   # ne: 1e19 m^-3
    rho_201, ti_201 = nn_fit_ti_transformer(rho_txcs, ti_keV,    # Ti: keV in/out
                                              te_ped_x, te_ped_y)
"""

import os
import numpy as np
import torch

# Model file default path: ../profile_nn_models/ relative to this file
_model_dir = os.path.join(os.path.dirname(__file__), '..', 'profile_nn_models')
_cache = {}  # Process-level model cache: {datatype: model}


def _load_transformer_model(datatype, model_dir=None):
    """Load a trained Transformer Baseline model from disk, cached per datatype.

    Load sequence:
      1. Check cache — multiple calls within same process load only once
      2. Read .pt checkpoint, extract model config (hidden, layers, heads, etc.)
      3. Instantiate the appropriate architecture class
      4. Load weights and set to eval mode

    Args:
        datatype: 'Te', 'ne', or 'Ti'
        model_dir: Model directory, defaults to ../profile_nn_models/

    Returns:
        Trained PyTorch model instance (on CPU, eval mode)

    Raises:
        FileNotFoundError: If model file is missing, with training command hint
    """
    if datatype in _cache:
        return _cache[datatype]

    if model_dir is None:
        model_dir = _model_dir

    from profile_nn.transformer_baseline import (
        BaseTransformerProfileNet, TransformerProfileNet_Ti,
    )

    model_path = os.path.join(model_dir, f'Transformer_{datatype}.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Transformer model not found: {model_path}\n"
            f"Train first: python -m profile_nn.transformer_baseline --data <data_dir> --datatype {datatype}"
        )

    # weights_only=False because checkpoint contains non-tensor data (config, history)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', {})
    hidden = config.get('hidden', 64)
    num_layers = config.get('num_layers', 2)
    num_heads = config.get('num_heads', 4)
    ffn_dim = config.get('ffn_dim', 256)
    dropout = config.get('dropout', 0.1)

    # Infer max_len from checkpoint if not in config (backward compat)
    max_len = config.get('max_len')
    if max_len is None:
        # Read from state dict: pos_embed shape is (1, max_len, hidden)
        state = checkpoint.get('model_state_dict', {})
        for key in ['encoder.pos_embed', 'encoder_main.pos_embed']:
            if key in state:
                max_len = state[key].shape[1]
                break
        if max_len is None:
            max_len = 64  # fallback

    if datatype == 'Ti':
        model = TransformerProfileNet_Ti(
            hidden=hidden, num_layers=num_layers, num_heads=num_heads,
            ffn_dim=ffn_dim, dropout=dropout, num_output=201,
            max_len=max_len,
        )
    else:
        model = BaseTransformerProfileNet(
            hidden=hidden, num_layers=num_layers, num_heads=num_heads,
            ffn_dim=ffn_dim, dropout=dropout, num_output=201,
            max_len=max_len,
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    _cache[datatype] = model
    return model


def _predict_transformer_profile(model, x, mask, device='cpu'):
    """Run Transformer model inference on a single sample.

    Converts numpy arrays to PyTorch tensors, runs forward pass, converts back.
    Batch dimension is 1 (single-sample inference).

    Args:
        model: Transformer model instance
        x: (N, 2) numpy array — (rho, y) point pairs
        mask: (N,) bool numpy array — valid point indicators
        device: inference device, default 'cpu'

    Returns:
        rho_out: (201,) float64 — uniform rho grid [0, 1]
        y_out:   (201,) float64 — predicted profile values
    """
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


# ===========================================================================
# Public Inference API
# ===========================================================================

def nn_fit_te_transformer(rho_scattered, te_scattered, model_dir=None):
    """Transformer Te fit: TS scatter -> Te continuous profile.

    Replaces original call chain:
        x_fit, y_fit, _, _, _ = fitting(x, y_keV, 'TS')
        x_201, y_201 = robust_interp(x_fit, y_fit, 'Te')

    Args:
        rho_scattered: 1D array, TS rho coordinates (normalized 0~1)
        te_scattered:  1D array, Te values [eV] from MDSplus raw data
        model_dir:     Optional, model directory path

    Returns:
        rho_201: (201,) uniform rho grid [0, 1]
        te_201:  (201,) Te profile [keV] — NOTE: output is keV
    """
    rho = np.asarray(rho_scattered, dtype=np.float32)
    te = np.asarray(te_scattered, dtype=np.float32)

    # Unit normalization: eV -> keV (matches training data distribution)
    te_keV = te / 1000.0

    x = np.stack([rho, te_keV], axis=-1)
    mask = np.ones(len(rho), dtype=bool)

    model = _load_transformer_model('Te', model_dir)
    rho_201, te_201 = _predict_transformer_profile(model, x, mask)

    # Monotonicity safety net
    try:
        from scipy.interpolate import PchipInterpolator
        pchip = PchipInterpolator(rho_201, -te_201)
        te_201 = np.maximum(-pchip(rho_201), 0.005)
        te_201 = np.minimum.accumulate(te_201)
    except Exception:
        pass

    # Edge floor protection
    te_201 = np.maximum(te_201, 0.005)
    te_201 = np.minimum.accumulate(te_201)
    return rho_201, te_201


def nn_fit_ne_transformer(rho_scattered, ne_scattered, model_dir=None):
    """Transformer ne fit: Refl scatter -> ne continuous profile.

    Replaces original call chain:
        x_fit, y_fit, _, _, _ = fitting(x, y, 'Refl')
        x_201, y_201 = robust_interp(x_fit, y_fit, 'ne')

    Args:
        rho_scattered: 1D array, reflectometer rho coordinates
        ne_scattered:  1D array, ne values [1e19 m^-3] from MDSplus
        model_dir:     Optional, model directory path

    Returns:
        rho_201: (201,) uniform rho grid [0, 1]
        ne_201:  (201,) ne profile [1e19 m^-3]
    """
    rho = np.asarray(rho_scattered, dtype=np.float32)
    ne = np.asarray(ne_scattered, dtype=np.float32)

    x = np.stack([rho, ne], axis=-1)
    mask = np.ones(len(rho), dtype=bool)

    model = _load_transformer_model('ne', model_dir)
    rho_201, ne_201 = _predict_transformer_profile(model, x, mask)

    # Monotonicity safety net
    try:
        from scipy.interpolate import PchipInterpolator
        pchip = PchipInterpolator(rho_201, -ne_201)
        ne_201 = np.maximum(-pchip(rho_201), 0.001)
        ne_201 = np.minimum.accumulate(ne_201)
    except Exception:
        pass

    # Edge floor protection
    ne_201 = np.maximum(ne_201, 0.001)
    ne_201 = np.minimum.accumulate(ne_201)
    return rho_201, ne_201


def nn_fit_ti_transformer(rho_scattered, ti_scattered, te_ped_x, te_ped_y, model_dir=None):
    """Transformer Ti fit: TXCS scatter + Te pedestal -> Ti continuous profile.

    Replaces original call chain:
        x_fit, y_fit, _, _, _ = fitting(x, y, 'Ti', te_ped_x=..., te_ped_y=...)
        x_201, y_201 = robust_interp(x_fit, y_fit, 'Ti')
        x_201, y_201 = enforce_monotone_pchip(x_201, y_201)

    Key difference from Te/ne:
      - Requires additional Te pedestal data (rho >= 0.88 region)
      - Uses dual Transformer encoder architecture
      - Both TXCS and Te pedestal encoders use separate self-attention

    Args:
        rho_scattered: 1D array, TXCS rho coordinates
        ti_scattered:  1D array, Ti values [keV] from MDSplus TXCS diagnostic
        te_ped_x:      1D array, Te pedestal rho coordinates (rho >= 0.88)
        te_ped_y:      1D array, Te pedestal values [keV]
        model_dir:     Optional, model directory path

    Returns:
        rho_201: (201,) uniform rho grid [0, 1]
        ti_201:  (201,) Ti profile [keV]
    """
    from scipy.interpolate import PchipInterpolator

    rho = np.asarray(rho_scattered, dtype=np.float32)
    ti = np.asarray(ti_scattered, dtype=np.float32)
    ped_x = np.asarray(te_ped_x, dtype=np.float32)
    ped_y = np.asarray(te_ped_y, dtype=np.float32)

    x_txcs = np.stack([rho, ti], axis=-1)
    mask_txcs = np.ones(len(rho), dtype=bool)
    x_ped = np.stack([ped_x, ped_y], axis=-1)
    mask_ped = np.ones(len(ped_x), dtype=bool)

    model = _load_transformer_model('Ti', model_dir)
    model = model.to('cpu')

    x_t = torch.from_numpy(x_txcs).unsqueeze(0)    # (1, N_txcs, 2)
    m_t = torch.from_numpy(mask_txcs).unsqueeze(0)  # (1, N_txcs)
    p_t = torch.from_numpy(x_ped).unsqueeze(0)      # (1, N_ped, 2)
    mp_t = torch.from_numpy(mask_ped).unsqueeze(0)  # (1, N_ped)

    with torch.no_grad():
        y = model(x_t, m_t, p_t, mp_t)  # (1, 201)

    rho_201 = np.linspace(0, 1, 201, dtype=np.float64)
    ti_201 = y.squeeze(0).cpu().numpy().astype(np.float64)

    # Monotonicity safety net
    try:
        pchip = PchipInterpolator(rho_201, -ti_201)
        ti_201_mono = -pchip(rho_201)
        ti_201 = np.maximum(ti_201_mono, 0.01)
        ti_201 = np.minimum.accumulate(ti_201)
    except Exception:
        pass

    # Edge floor protection
    ti_201 = np.maximum(ti_201, 0.005)
    ti_201 = np.minimum.accumulate(ti_201)
    return rho_201, ti_201


def set_model_dir_transformer(path):
    """Set directory containing trained Transformer .pt model files.

    After calling:
      - Subsequent nn_fit_te/ne/ti_transformer calls load models from new path
      - Cache is cleared, next call will reload

    Args:
        path: absolute or relative path to model directory
    """
    global _model_dir
    _model_dir = path
    _cache.clear()
