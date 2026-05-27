"""Generate training data for ProfileNet by running existing mtanh fitter on EAST MDSplus data.

Saves (scattered diagnostic points, fitted 201-pt profile) pairs as .npz files.

Usage:
    python -m profile_nn.generate_data --shot 81481           # single shot
    python -m profile_nn.generate_data --shots 81481 81482    # multiple shots
    python -m profile_nn.generate_data --shots 81481 81482 --synthetic  # add synthetic augmentation
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from readMDS_onetwo import readmds
from fitting_mtanh import fitting
from scipy.interpolate import interp1d, PchipInterpolator

try:
    from MDSplus.connection import Connection
    HAS_MDS = True
except ImportError:
    HAS_MDS = False


# ---------------------------------------------------------------------------
# Utility functions (same as onetwo_output.py)
# ---------------------------------------------------------------------------

def robust_interp(x, y, datatype, thresholds=None, offsets=None, num_points=201):
    if thresholds is None:
        thresholds = {"ne": 0.1, "Te": 0.05, "Ti": 0.05}
    if offsets is None:
        offsets = {"ne": 0.05, "Te": 0.05, "Ti": 0.05}
    th = thresholds.get(datatype, 0.0)
    off = offsets.get(datatype, 0.0)
    x, y = np.array(x), np.array(y)
    mask = y >= th
    x_clean, y_clean = x[mask], y[mask]
    if len(y_clean) == 0:
        y_lifted = y + off
        f = interp1d(x, y_lifted, kind='linear', fill_value="extrapolate")
        x_fit = np.linspace(0, 1.0, num_points)
        return x_fit, f(x_fit)
    f = interp1d(x_clean, y_clean, kind='linear', fill_value="extrapolate")
    x_fit = np.linspace(0, 1.0, num_points)
    y_fit = f(x_fit)
    if np.min(y_fit) < th:
        y_lifted = y + off
        f = interp1d(x, y_lifted, kind='linear', fill_value="extrapolate")
        x_fit = np.linspace(0, 1.0, num_points)
        y_fit = f(x_fit)
    return x_fit, y_fit


def enforce_monotone_pchip(x, y, num_points=201, decreasing=True):
    x, y = np.asarray(x), np.asarray(y)
    order = np.argsort(x)
    x, y = x[order], y[order]
    y_proc = -y if decreasing else y
    pchip = PchipInterpolator(x, y_proc)
    x_new = np.linspace(0, 1, num_points)
    y_new = pchip(x_new)
    return x_new, -y_new if decreasing else y_new


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def extract_te_sample(shot, time_point, output_dir):
    """Extract one Te training sample."""
    data, status, real_time = readmds(shot, time_point)
    if status['TS_status'] != 1:
        return None

    try:
        x, y, datatype = data['Te']['TS']['Rho'], data['Te']['TS']['data'], data['Te']['TS']['type']
        # y is raw MDSplus Te in eV. Convert to keV to match nn_fit_te() normalization.
        y_keV = y / 1000.0
        x_fit, y_fit, _, _, _ = fitting(x, y, datatype)  # fitting() internally /1000, returns keV
        x_201, y_201 = robust_interp(x_fit, y_fit, datatype)  # y_201 in keV

        # Get Te pedestal for later Ti samples
        mask_ped = x_fit >= 0.88
        te_ped_x = x_fit[mask_ped]
        te_ped_y = y_fit[mask_ped]

        time_str = f"{real_time:.5f}".replace('.', 'd')
        np.savez(
            os.path.join(output_dir, f"{shot}_Te_{time_str}.npz"),
            X_rho=x.astype(np.float32),   # rho, dimensionless
            X_val=y_keV.astype(np.float32),  # Te [keV] — matches nn_fit_te input
            Y=y_201.astype(np.float32),      # Te [keV] — target profile
            shot=shot, time=real_time, datatype='Te',
        )
        return {'te_ped_x': te_ped_x, 'te_ped_y': te_ped_y}
    except Exception as e:
        print(f"  Te extraction failed @ {time_point:.3f}s: {e}")
        return None


def extract_ne_sample(shot, time_point, output_dir):
    """Extract one ne training sample."""
    data, status, real_time = readmds(shot, time_point)
    if status['Refl_status'] != 1:
        return None

    try:
        x, y, datatype = data['ne']['Refl']['Rho'], data['ne']['Refl']['data'], data['ne']['Refl']['type']
        x_fit, y_fit, _, _, _ = fitting(x, y, datatype)
        x_201, y_201 = robust_interp(x_fit, y_fit, datatype)

        time_str = f"{real_time:.5f}".replace('.', 'd')
        np.savez(
            os.path.join(output_dir, f"{shot}_ne_{time_str}.npz"),
            X_rho=x.astype(np.float32),
            X_val=y.astype(np.float32),
            Y=y_201.astype(np.float32),
            shot=shot, time=real_time, datatype='ne',
        )
        return True
    except Exception as e:
        print(f"  ne extraction failed @ {time_point:.3f}s: {e}")
        return None


def extract_ti_sample(shot, time_point, output_dir, te_ped_data):
    """Extract one Ti training sample. Requires Te pedestal data."""
    if te_ped_data is None:
        return None

    data, status, real_time = readmds(shot, time_point)
    if status['TXCS_status'] != 1:
        return None

    try:
        x, y, datatype = data['Ti']['TXCS']['Rho'], data['Ti']['TXCS']['data'], data['Ti']['type']
        x_fit, y_fit, _, _, _ = fitting(
            x, y, datatype,
            te_ped_x=te_ped_data['te_ped_x'],
            te_ped_y=te_ped_data['te_ped_y'],
        )
        x_201, y_201 = robust_interp(x_fit, y_fit, datatype)
        x_201, y_201 = enforce_monotone_pchip(x_201, y_201)

        time_str = f"{real_time:.5f}".replace('.', 'd')
        np.savez(
            os.path.join(output_dir, f"{shot}_Ti_{time_str}.npz"),
            X_rho=x.astype(np.float32),
            X_val=y.astype(np.float32),
            X_te_ped_rho=te_ped_data['te_ped_x'].astype(np.float32),
            X_te_ped_val=te_ped_data['te_ped_y'].astype(np.float32),
            Y=y_201.astype(np.float32),
            shot=shot, time=real_time, datatype='Ti',
        )
        return True
    except Exception as e:
        print(f"  Ti extraction failed @ {time_point:.3f}s: {e}")
        return None


# ---------------------------------------------------------------------------
# Synthetic data augmentation
# ---------------------------------------------------------------------------

def generate_synthetic_samples(output_dir, num_samples=500, seed=42):
    """Generate synthetic (rho, y) -> profile pairs for testing without MDSplus.

    Uses parameterized mtanh-like profiles with random noise to simulate
    realistic diagnostic scatter. Useful for development and offline testing.
    """
    rng = np.random.default_rng(seed)
    print(f"Generating {num_samples} synthetic samples per datatype...")

    for datatype, y_range, noise_frac, n_pts_range in [
        ('Te',  (0.5, 5.0),   0.03, (15, 30)),   # keV
        ('ne',  (1.0, 6.0),   0.05, (12, 22)),   # 1e19 m^-3
        ('Ti',  (0.3, 4.0),   0.04, (3, 12)),    # keV
    ]:
        for i in range(num_samples):
            # Random pedestal parameters
            ped_height = rng.uniform(*y_range)
            ped_pos = rng.uniform(0.88, 0.96)
            ped_width = rng.uniform(0.02, 0.08)
            core_val = ped_height * rng.uniform(1.5, 3.0)

            # Generate smooth profile: core flat + mtanh pedestal
            rho_fine = np.linspace(0, 1, 201)
            profile = core_val - (core_val - ped_height) * (
                0.5 * (1 + np.tanh((rho_fine - ped_pos) / ped_width))
            )
            profile = profile - profile[-1] + rng.uniform(0.01, 0.1)  # edge offset
            profile = np.maximum(profile, 0.01)

            # Scatter diagnostic-like points with noise
            n_pts = rng.integers(*n_pts_range)
            rho_scattered = np.sort(rng.uniform(0.0, 1.02, n_pts))
            rho_scattered = np.clip(rho_scattered, 0.0, 1.0)
            y_true_at_pts = np.interp(rho_scattered, rho_fine, profile)
            y_noisy = y_true_at_pts * (1 + rng.normal(0, noise_frac, n_pts))
            y_noisy = np.maximum(y_noisy, 0.005)

            if datatype == 'Ti':
                # Ti needs Te pedestal synthetic data too
                mask_ped = rho_fine >= 0.88
                ped_rho = rho_fine[mask_ped]
                ped_val = profile[mask_ped] * rng.uniform(0.9, 1.1, mask_ped.sum())
                ped_val = np.maximum(ped_val, 0.01)
                np.savez(
                    os.path.join(output_dir, f"synthetic_Ti_{i:05d}.npz"),
                    X_rho=rho_scattered.astype(np.float32),
                    X_val=y_noisy.astype(np.float32),
                    X_te_ped_rho=ped_rho.astype(np.float32),
                    X_te_ped_val=ped_val.astype(np.float32),
                    Y=profile.astype(np.float32),
                    shot=-1, time=float(i), datatype=datatype,
                )
            else:
                np.savez(
                    os.path.join(output_dir, f"synthetic_{datatype}_{i:05d}.npz"),
                    X_rho=rho_scattered.astype(np.float32),
                    X_val=y_noisy.astype(np.float32),
                    Y=profile.astype(np.float32),
                    shot=-1, time=float(i), datatype=datatype,
                )

        print(f"  {datatype}: {num_samples} synthetic samples saved")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Generate ProfileNet training data')
    parser.add_argument('--shots', type=int, nargs='+', help='EAST shot numbers')
    parser.add_argument('--output', type=str, default='./profile_nn_data',
                        help='Output directory for .npz files')
    parser.add_argument('--synthetic', type=int, default=0, metavar='N',
                        help='Also generate N synthetic samples per datatype')
    parser.add_argument('--time-limit', type=int, default=0, metavar='N',
                        help='Limit to first N time slices per shot (0 = all)')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Synthetic data always available
    if args.synthetic > 0:
        generate_synthetic_samples(str(output_dir), num_samples=args.synthetic)

    # Real MDSplus data
    if args.shots:
        if not HAS_MDS:
            print("ERROR: MDSplus not available. Install MDSplus to generate real data.")
            print("Use --synthetic N for synthetic test data instead.")
            sys.exit(1)

        MDS_IP = '202.127.204.42'
        conn = Connection(MDS_IP)

        for shot in args.shots:
            print(f"\n{'='*50}")
            print(f"Processing shot {shot}")
            print(f"{'='*50}")

            try:
                conn.openTree('TS_EAST', shot)
                TS_times = conn.get(r'dim_of(\Te_coreTS)').data()
                conn.closeTree('TS_EAST', shot)
            except Exception as e:
                print(f"Cannot read TS times for shot {shot}: {e}")
                continue

            if args.time_limit > 0:
                TS_times = TS_times[:args.time_limit]

            print(f"Found {len(TS_times)} time points")

            te_count = ne_count = ti_count = 0

            for t in TS_times:
                # Te must succeed for ne and Ti to proceed
                te_ped = extract_te_sample(shot, t, str(output_dir))
                if te_ped is not None:
                    te_count += 1
                else:
                    continue  # skip ne and Ti if Te failed

                ne_ok = extract_ne_sample(shot, t, str(output_dir))
                if ne_ok:
                    ne_count += 1

                ti_ok = extract_ti_sample(shot, t, str(output_dir), te_ped)
                if ti_ok:
                    ti_count += 1

            print(f"Shot {shot} done: Te={te_count}, ne={ne_count}, Ti={ti_count}")

    # Summary
    all_files = sorted(output_dir.glob('*.npz'))
    te_files = [f for f in all_files if '_Te_' in f.name]
    ne_files = [f for f in all_files if '_ne_' in f.name]
    ti_files = [f for f in all_files if '_Ti_' in f.name]
    print(f"\n{'='*50}")
    print(f"Total dataset: Te={len(te_files)}, ne={len(ne_files)}, Ti={len(ti_files)}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
