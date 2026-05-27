# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a plasma physics research codebase for the EAST tokamak, focused on Lower Hybrid Current Drive (LHCD) modeling. It contains a Python translation of the METIS LH module (originally MATLAB), diagnostic data readers, equilibrium file handlers, and profile fitting utilities.

## Key Dependencies

- numpy, scipy, matplotlib
- MDSplus (Python connector for EAST MDSplus data server)
- tkinter (for the GUI in `LHW_gui.py`)
- PyTorch >= 1.13 (for `profile_nn/` neural network profile fitting; CPU-only is fine)

There is no `requirements.txt` or `setup.py`; dependencies must be installed manually in the Python environment.

## Running the Code

There is no formal build system, test suite, or linter. The project is a flat collection of Python scripts.

### Main Entry Points

- **CLI:** `python LHW_main.py` — prompts for shot/time, reads from EAST MDSplus, runs METIS-LH, and plots 4-panel profiles (current density, power density, cumulative current, cumulative power).
- **GUI:** `python LHW_gui.py` — launches a Tkinter interface. Supports two data sources:
  - `mds`: reads Te/ne from EAST MDSplus (TS + Refl diagnostics)
  - `txt`: reads Te/ne from a local text file (e.g., `数据wu1.txt`)

### Other Standalone Scripts

- `python lower_view.py` — alternative standalone LH power deposition calculator (not using METIS model)
- `python nml.py` — NML smoothing demo using `readmds` + `fitting`
- `python mesh2.py` — flux surface geometry calculation from a gfile

### NN Profile Fitting Entry Points

- **Data generation:** `python -m profile_nn.generate_data --shots 81481 81482 --output ./profile_nn_data` — reads EAST MDSplus, runs existing mtanh fitter as labeler, saves (scattered points, fitted profile) pairs as `.npz`. Use `--synthetic 500` for offline test data without MDSplus access.
- **Training:** `python -m profile_nn.train --data ./profile_nn_data --datatype all --epochs 500 --output ./profile_nn_models` — trains ProfileNet models (Te, ne, Ti) with physics-constrained loss.
- **NN-based ONETWO pipeline:** `python onetwo_output_nn.py` — drop-in replacement for `onetwo_output.py` using trained NN instead of mtanh+spline fitting. Same interactive prompt, same output structure (results go to `onetwo_nn/` subdirectory).

## Architecture

### Data Flow (METIS-LH Pipeline)

```
readMDS.py          ->  geqdsk.py          ->  mesh2.py
(EAST MDSplus         (gfile load/save,      (flux surface geometry
 diagnostic data)     RZmap to rho)          g_to_23)
      |                                           |
      v                                           v
fitting_mtanh.py  <-  data_clean_new.py  <-   (Vrho, dvdpsi, phi)
(profile fitting      (IQR + iterative
 w/ mtanh/spline)     outlier cleaning)
      |
      v
LHW_main.py  ->  LHW_metis.py  ->  plots / txt output
(build cons/profil/  (METIS LH model:
 option dicts)        external_call_metis_lh_model)
```

### Core Modules

- **`LHW_metis.py`** — The METIS LH model translation. Exports `external_call_metis_lh_model(cons, profil, option)`. This is the physics engine; it handles Landau damping, upshift modes (`newmodel`, `1/q`, `Bpol`, etc.), two-lobe power spectra, and returns power deposition `plh_prof` and current density `jlh_prof`.
- **`LHW_main.py`** — Orchestrates the full pipeline. `build_east_metis_inputs(shot, time, ...)` fetches MDSplus data, fits profiles, reads the local gfile, computes `Vrho` via `mesh2.g_to_23`, and assembles the `cons`/`profil`/`option` dictionaries expected by `LHW_metis.py`.
- **`LHW_gui.py`** — Wraps the pipeline in a GUI. Also provides `build_east_metis_inputs_from_txt()` for offline runs using a text file instead of MDSplus.
- **`readMDS.py`** — Connects to EAST MDSplus servers (`202.127.204.42` by default) and reads:
  - `efit_east` tree for gfile data
  - `TS_EAST` for Thomson scattering (Te, ne)
  - `TXCS_EAST` for X-ray crystal spectrometer (Ti, Te)
  - `Brem_EAST` for Zeff
  - `energy_east` for H98
  - `ReflJ_EAST` for reflectometer (ne)
  - `EAST` tree for LH power signals (`PLHI2`, `PLHR2`)
- **`geqdsk.py`** — Reads/writes EFIT `gfile` equilibrium files (fixed-width Fortran format). Also provides `RZmap()` to map (R,Z) diagnostic points to flux surface coordinate `rho`, and `read_from_MDS()` to pull equilibrium data from an open MDSplus connection.
- **`mesh2.py`** — Computes flux surface meshes and geometric quantities from a gfile via `g_to_23(gfile, psi_norm, theta)`. Returns `dvdpsi`, `phi`, `vol`, `area`, Jacobian, magnetic field components, etc. Used to compute `Vrho = dV/drho`.
- **`fitting_mtanh.py`** — Profile fitting engine. `fitting(a, b, datatype, ...)` supports:
  - `datatype='Te'` — mtanh pedestal + core spline with smooth transition
  - `datatype='neTS'` / `'Refl'` — front-pedestal fitting + mtanh
  - `datatype='Ti'` — Ti-specific handling with optional Te pedestal fallback
  - L-mode vs H-mode branching based on H98 threshold
  - Debug mode: saves stage-by-stage NPZ/PNG/JSON outputs to `debug_dir`
- **`data_clean_new.py`** — Data cleaning utilities (`clean()`, `IQR()`, `iter_clean()`, `del_lowdata()`). Used by `fitting_mtanh.py` before fitting.
- **`left_smooth.py`** — Left boundary smoothing (`new_sm_extremity`) to force derivative ≈ 0 near rho=0.
- **`re_fitting.py`** — Pedestal fitting helper using `mtanh` + least squares (`leastsq`).
- **`nml.py`** — NML (inverse scale length) smoothing with `nml_smooth(x, y, x_core_end, x_ped_start)`.

### Important Conventions

- **gfile naming:** Local equilibrium files must be named `{shot}_{time}_gfile` (e.g., `81481_5.3_gfile`). `LHW_main.py` expects this exact format.
- **Units (METIS-LH pipeline):**
  - ne profile from fitting: typically in units of `1e19 m^-3` (code multiplies by `1e19` before passing to METIS)
  - Te profile: keV from fitting, converted to eV for METIS (`* 1e3`)
  - Power: METIS uses Watts; EAST MDSplus gives kW (`PLHI2`, `PLHR2`)
  - Current density `jlh_prof`: returned as A/m², often converted to A/cm² for plots (`/ 1e4`)
- **Units (ONETWO pipeline — `onetwo_output.py`, `onetwo_output_nn.py`):**
  - TEIN (Te for ONETWO): expected in eV. NOTE: `onetwo_output.py` writes keV (likely a bug); `onetwo_output_nn.py` correctly writes eV via `* 1e3`.
  - ENEIN (ne for ONETWO): cm⁻³. Input from Refl is `1e19 m⁻³`, converted via `* 1e13`.
  - TIIN (Ti for ONETWO): keV (no conversion needed; TXCS data already in keV).
  - extcurrf_curr (LHW current): A/cm². Input from `lower_onetwo` is MA/m², converted via `* 1e6 / 1e4`.
- **Rho grid:** Usually `np.linspace(0, 1, gdate['nw'])` where `nw` comes from the gfile.
- **Vrho calculation:** `Vrho = dvdpsi * dpsi_drho`, computed on the psi grid from `mesh2.g_to_23`, then interpolated to the rho grid.
- **`eta_scale`:** An empirical machine calibration factor in `option` that uniformly scales the driven current and efficiency without changing profile shapes.

### Network / External Dependencies

- MDSplus connection to EAST servers (`202.127.204.12` for power, `202.127.204.42` for diagnostics in `readMDS.py`) is required for live data runs. Offline runs (`txt` mode) only need the local gfile and a text file with rho/ne/Te columns.

### ONETWO Integration Files

- `onetwo_output.py`, `onetwo_time_analysis.py`, `readMDS_onetwo.py`, `ecrh_onetwo.py`, `outone_*.py` — Related to ONETWO transport code workflow (output parsing, time analysis, ECRH module). These are auxiliary scripts for post-processing ONETWO runs and are not part of the main METIS-LH pipeline.
- `onetwo_output_nn.py` — NN variant of `onetwo_output.py`. Replaces `fitting()` + `robust_interp()` + `enforce_monotone_pchip()` calls with `profile_nn.infer` functions. All other logic (MDSplus read, LHW via `lower_onetwo`, gfile save, ONETWO execution) is identical.

### NN Profile Fitting (`profile_nn/`)

A PyTorch package that replaces the mtanh+spline fitting chain with a small neural network (~50K params).

**Architecture:** DeepONet-style set encoder + coordinate decoder.
- **SetEncoder** (`model.py:SetEncoder`): PointNet-style MLP(2→64→128→128) processes each (ρ, y) diagnostic point independently; global max pooling produces a fixed-length latent vector `z`. This naturally handles variable diagnostic channel counts (TS ~20-25 pts, Refl ~15-20, TXCS ~3-15).
- **CoordDecoder** (`model.py:CoordDecoder`): MLP(1→128→128) maps target rho coordinate to basis functions `b(ρ)`. Output = `z · b(ρ)` (branch-trunk dot product), followed by Softplus for positivity.
- **ProfileNet_Ti** has a dual encoder: one for TXCS points, one for Te pedestal data (ρ≥0.88), fused via MLP — matching the original `fitting()` Ti logic that requires Te pedestal boundary info.

**Physics-constrained loss** (`model.py:PhysicsConstrainedLoss`):
`L = L_MSE + λ₁·L_mono + λ₂·L_bdy + λ₃·L_smooth`
- `L_mono`: penalizes positive derivatives (non-monotonic segments) via `ReLU(diff(y))`
- `L_bdy`: penalizes non-zero derivative near ρ=0 (magnetic axis symmetry)
- `L_smooth`: penalizes large second derivatives

**Inference** (`infer.py`): Three drop-in functions matching original signatures:
- `nn_fit_te(rho, te_eV)` → `(rho_201, te_keV)` — input eV, output keV
- `nn_fit_ne(rho, ne_1e19)` → `(rho_201, ne_1e19)` — same units in/out
- `nn_fit_ti(rho, ti_keV, te_ped_x, te_ped_y)` → `(rho_201, ti_keV)` — Ti applies PCHIP monotonicity as safety-net post-processing

**Data format** (`dataset.py`): `.npz` files with `X_rho`, `X_val` (scattered points), `Y` (201-pt label). Ti samples additionally have `X_te_ped_rho`, `X_te_ped_val`. Variable-length inputs are padded to batch-max with boolean masks.

## Notes for Editing

- The code contains mixed Chinese and English comments.
- There are no unit tests. Validation is typically done by visual inspection of the 4-panel output plots and checking cumulative current/power integrals against input power.
- Many scripts have `if __name__ == "__main__":` blocks with hardcoded shot numbers or interactive `input()` prompts.
- When modifying `LHW_metis.py`, preserve the exact dictionary keys expected in `cons`, `profil`, and `option`; `LHW_main.py` and `LHW_gui.py` depend on them.
