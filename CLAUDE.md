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

- **Data inspection:** `python npz_browser.py` — browse `.npz` dataset (summary stats, per-shot listing, bad-data scan, sample plots). Use `--datatype Te`, `--shot 81481`, `--bad`, `--plot Te` flags.
- **Data generation:** `python -m profile_nn.generate_data --shots 81481 81482 --output ./profile_nn_data` — reads EAST MDSplus, runs existing mtanh fitter as labeler, saves (scattered points, fitted profile) pairs as `.npz`. Use `--synthetic 500` for offline test data without MDSplus access. Current dataset: 161 shots, ~3231 samples.
- **Training (ProfileNet):** `python -m profile_nn.train --data ./profile_nn_data --datatype all --epochs 500 --output ./profile_nn_models` — trains DeepONet-style models.
- **Training (LSTM):** `python -m profile_nn.lstm_baseline --datatype all --epochs 500` — trains BiLSTM+CoordDecoder baseline.
- **Training (CNN):** `python -m profile_nn.cnn_baseline --datatype all --epochs 500` — trains pure ResNet CNN-1D baseline (no DeepONet components — control group).
- **Training (Transformer):** `python -m profile_nn.transformer_baseline --datatype all --epochs 500` — trains Transformer Encoder+CoordDecoder baseline.
- **ONETWO pipelines (5 variants):**
  - `python onetwo_output.py` — original mtanh+spline (output → `results/{shot}/{time_dir}/onetwo/`)
  - `python onetwo_output_nn.py` — ProfileNet (output → `results/{shot}/{time_dir}/onetwo_nn/`)
  - `python onetwo_output_lstm.py` — LSTM (output → `results/{shot}/{time_dir}/onetwo_lstm/`)
  - `python onetwo_output_cnn.py` — CNN-1D (output → `results/{shot}/{time_dir}/onetwo_cnn/`)
  - `python onetwo_output_transformer.py` — Transformer (output → `results/{shot}/{time_dir}/onetwo_transformer/`)
  All five share the same interactive prompt, output structure, error-logging pattern (`_log_error()` → `results/{shot}/pipeline_errors.log`), and ONETWO execution logic; they only differ in the profile fitting step.
  - `python outonetwo.py` — parallel performance-analysis version of the mtanh pipeline (with timing stats).
  - **Pre-flight check:** `python check_gfile.py` — checks which shots/time points have gfile on MDSplus before running pipelines.
- **Cross-method comparison:** `python compare_profiles.py <shot> <time>` — extracts TEIN/ENEIN/TIIN from all five pipelines' `inone` files, plots overlay + deviation, computes MAE/RMSE/MaxAE/MeanRel% vs mtanh baseline. Also supports `--batch` for all-time-point scanning and `--csv-only` for summary CSV. No PyTorch dependency. Auto-normalizes TEIN units (detects mtanh keV vs NN eV, converts to common unit).
- **V4 vs V2 comparison (archived):** Results confirmed real H-mode data (V4) significantly improves Core/Pedestal MAE vs synthetic H-mode data (V2). See `profile_nn/优化记录.md` for details.

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
- **`readMDS.py`** — Original MDSplus reader for METIS-LH pipeline. Connects to `202.127.204.42` and reads TS_EAST (Te, ne), TXCS_EAST (Ti), efit_east (gfile), Brem_EAST (Zeff), energy_east (H98), EAST tree (PLHI2/PLHR2). Used by `LHW_main.py`, `LHW_gui.py`, `fitting_mtanh.py`, `lower_view.py`. (Note: `lower_view_final.py` was migrated to `readMDS_onetwo`.)
- **`readMDS_onetwo.py`** — ONETWO-specific fork of readMDS. Key differences: (1) auto-saves gfile as `{shot}_{real_time}_gfile` using the actual gfile timestamp from MDSplus — note this saves under `real_time` not `time`, (2) gfile time-gap warning when `|gfile_time - requested_time| > 0.5s`, (3) empty g_times array check with clear error, (4) adds ReflJ_EAST reflectometer ne support, (5) includes Brem_EAST Zeff reading, (6) Ti robustness: `z_TXCS` fallback (linear array if MDSplus read fails), channel alignment between z_TXCS/Ti_TXCS/Te_TXCS, empty-Ti handling (status flag). Used by all 6 ONETWO pipelines, `lower_view_final_nn.py`, `profile_nn/generate_data.py`, `scan_compare_shots.py`, and now also `lower_view_final.py`. When fixing a bug in one, check the other.
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

- **gfile naming:** Local equilibrium files must be named `{shot}_{time}_gfile` (e.g., `81481_5.3_gfile`). Note: `readMDS_onetwo.py` saves under the MDSplus gfile timestamp (`real_time`), while the ONETWO pipeline's time directories use the TS diagnostic timestamp (`time`). These differ slightly (different diagnostic systems, different time bases). `lower_view_final_nn.py::_find_gfile()` handles this by searching sequentially: exact match → same-shot fuzzy → cross-shot fallback.
- **LHW NaN protection:** Both `lower_view_final.py` and `lower_view_final_nn.py` apply defensive `np.clip()` on Te/ne profiles before the LHW physics calculation (`te ≥ 0.001`, `ne ≥ 0`) to prevent `sqrt(negative)→NaN` in the power deposition integral. Do not remove these guards.
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

### Server Deployment: HPC SDK Conflict

ONETWO pipelines run on a Linux server that may have NVIDIA HPC SDK installed. Its `libgomp.so.1` conflicts with PyTorch:

```
OSError: libgomp.so.1: undefined symbol: __pgi_nvomp_async_val_to_queue_num
```

**Fix (per-command, safe):**
```bash
env LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v hpc_sdk | tr '\n' ':') python3 ...
```

**Convenience alias (add to ~/.bashrc):**
```bash
alias pytorch_run='env LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ":" "\n" | grep -v hpc_sdk | tr "\n" ":") python3'
```

Then use `pytorch_run` in place of `python3` for any script that imports torch. `onetwo_129_201` does not link HPC SDK, so removing its path is safe for ONETWO execution. Scripts that don't import torch (`onetwo_output.py`, `compare_profiles.py`) use plain `python3`.

### Server Deployment: MDSplus RPATH Workaround

The MDSplus shared library (`libMdsLib.so`) has a hardcoded RPATH to `/pkg/mdsplus/5.0/gcc-4.4/lib` which doesn't exist on the deployment server. LD_LIBRARY_PATH alone cannot override RPATH. **One-time fix** (already applied on current server):

```bash
sudo mkdir -p /pkg/mdsplus/5.0/gcc-4.4/lib
sudo ln -s /usr/local/mdsplus/lib/* /pkg/mdsplus/5.0/gcc-4.4/lib/
```

Without this symlink, ONETWO exits with code 127 (dynamic linker cannot find `libMdsLib.so` even with correct LD_LIBRARY_PATH). This is a server-environment fix, not a code fix — it survives reboots and doesn't need to be in the Python scripts.

### ONETWO Integration Files

- **`onetwo_output.py`** — Original mtanh+spline ONETWO pipeline. Reads MDSplus → `fitting()` + `robust_interp()` + `enforce_monotone_pchip()` → Namelist assembly → ONETWO execution. Output to `results/{shot}/{time_dir}/onetwo/`. Uses `lower_onetwo()` from `lower_view_final.py` for LHW calculation.
- **`onetwo_output_nn.py`** / **`onetwo_output_lstm.py`** / **`onetwo_output_cnn.py`** / **`onetwo_output_transformer.py`** — NN variants replacing the fitting chain. All other logic (MDSplus read, LHW calculation via `lower_onetwo_nn()`, gfile save, Namelist assembly, ONETWO execution) is identical across all four. Output to `results/{shot}/{time_dir}/onetwo_nn/` / `onetwo_lstm/` / `onetwo_cnn/` / `onetwo_transformer/` respectively — isolated from each other and from the original.
- **ONETWO execution** — All 6 pipeline files resolve the binary via `_ONETWO_EXE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'onetwo_129_201')` and set `LD_LIBRARY_PATH` in `subprocess.run(..., env=_env)` before execution. The required library paths are: `/usr/local/mdsplus/lib`, `/home/fusion/imd/onetwo5/lib`, `/home/fusion/imd/auto12/netcdf4.1.3/pgi-1410/lib`, `/home/fusion/imd/auto12/cfetr_bin/lib`, `/home/fusion/imd/auto12/hdf5/pgi-1410/lib`. Additionally, MDSplus has a hardcoded RPATH to `/pkg/mdsplus/5.0/gcc-4.4/lib` which requires a symlink workaround on the deployment server.
- **Error logging** — All pipelines share a `_log_error(shot, time, stage, error_msg)` pattern that appends to `results/{shot}/pipeline_errors.log`. This replaces the old `print()`-based error reporting.
- **Known issue: ZFIT exit 91** — ONETWO may exit with code 91 (`FATAL ERROR in ZFIT: exceeded temperature range for curve fits`) when input temperature profiles (TEIN/TIIN) have edge values below ~0.1 keV. This is an internal ONETWO Fortran atomic-physics limitation; the template was designed for CFETR (high-temperature) and may reject EAST edge profiles. Not a Python pipeline bug.
- **`lower_view_final_nn.py`** — LHW power/current calculator that accepts externally provided NN profiles via `lower_onetwo_nn(shot, time, result_dir, te_profile_nn, ne_profile_nn, te_rho_nn, ne_rho_nn)`. This is the bridge between NN-fitted Te/ne profiles and the LHW physics code — bypasses internal mtanh fitting to avoid negative-value→NaN issues. Also includes `_find_gfile()` with 3-strategy gfile search (exact → same-shot fuzzy → cross-shot fallback). Used by all four NN ONETWO pipelines.
- **`onetwo_129_201`** — Fortran ONETWO executable (must exist in CWD when running any pipeline). Actual binary at `/home/fusion/imd/onetwo5/source/onetwo_v5.9.4/onetwo_129_201`, symlinked into `~/fit/`.
- **`inone_template`** — ONETWO Namelist input template. Exists only on the **deployment server**, not in the local repo. Must be in CWD when running any ONETWO pipeline.
- **`Namelist3.py`** — Fortran namelist I/O library. Provides `Namelist` class (dict subclass) with `read(path)` / `write(path)` for ONETWO input files (`inone`). Used by all ONETWO pipelines and `compare_profiles.py`.
- **`compare_profiles.py`** — Cross-method comparison tool. Reads TEIN/ENEIN/TIIN from all five pipelines' `inone` files, plots overlay + deviation from mtanh baseline, writes quantitative metrics. Zero PyTorch dependency.
- **`check_gfile.py`** — Pre-flight utility to check which shots/time points have gfile on MDSplus before running pipelines. Use to avoid wasting time on shots without equilibrium data.
- `onetwo_time_analysis.py`, `outonetwo.py`, `ecrh_onetwo.py`, `outone_*.py` — Auxiliary scripts for ONETWO post-processing (time analysis, parallel performance profiling, ECRH module, output parsing).

### NN Profile Fitting (`profile_nn/`)

A PyTorch package that replaces the mtanh+spline fitting chain with small neural networks. Four architectures are implemented — ProfileNet, LSTM, and Transformer share the same DeepONet-style CoordDecoder + PhysicsConstrainedLoss; CNN-1D is an independent control group:

**ProfileNet** (`model.py`, `infer.py`): DeepONet with SetEncoder (PointNet-style). Permutation-invariant — order of diagnostic points doesn't matter.
**LSTM** (`lstm_baseline.py`, `infer_lstm.py`): BiLSTM encoder + same CoordDecoder. Requires ρ-sorted input but captures radial sequence structure.
**CNN-1D** (`cnn_baseline.py`, `infer_cnn.py`): Pure ResNet Conv1d encoder-decoder (no CoordDecoder, no branch-trunk dot product). Independent architecture — a genuine control group, not a DeepONet variant.

**Transformer** (`transformer_baseline.py`, `infer_transformer.py`): TransformerEncoder (2-layer, 4-head self-attention, learnable positional encoding) + same CoordDecoder. Captures global correlations via self-attention rather than recurrence (LSTM) or pooling (SetEncoder).

| Model | Te val | ne val | Ti val | Params (Te) | Architecture |
|-------|--------|--------|--------|-------------|-------------|
| **LSTM** | **0.325** ✓ | **0.358** ✓ | 0.152 | 175K | BiLSTM + CoordDecoder |
| **ProfileNet** | 0.370 | 0.376 | **0.149** ✓ | 58K | SetEncoder + CoordDecoder |
| **Transformer** | **0.367** | **0.373** | **0.162** | 127K | TransformerEncoder + CoordDecoder |
| **CNN-1D** | 0.806 | 1.056 | 0.286 | 80K | Pure ResNet (no DeepONet) |

**Deployment recommendation (V4 — real H-mode data):** validation best: Te → CNN, ne → Transformer, Ti → LSTM. The FED paper (`paper/manuscript_fed.tex`, 5-shot 24-time-point test set) uses the test-set-based ensemble: **Te → CNN, ne → ProfileNet, Ti → LSTM**. Models are in `profile_nn_models/` (renamed from `profile_nn_models_v4/` after cleanup). All trained with real H-mode data (342 Te + 302 ne + 148 Ti H-mode samples from 52 shots).

**Shared architecture across ProfileNet & LSTM (DeepONet-style):**
- **Encoder** (branch net): Variable-length scattered points → fixed latent vector `z ∈ R¹²⁸`. ProfileNet uses PointNet SetEncoder (permutation-invariant); LSTM uses 2-layer BiLSTM(hidden=64) with ρ-sorted input.
- **CoordDecoder** (trunk net, `model.py:CoordDecoder`): MLP(1→128→128, SiLU) maps target ρ coordinate to basis functions `b(ρ)`. **Shared identically** by ProfileNet and LSTM — this is the key DeepONet component.
- **Output:** `Softplus(z · b(ρ))` — branch-trunk dot product ensures smoothness and positivity.
- **ProfileNet_Ti** / **LSTMProfileNet_Ti**: Dual encoder (TXCS points + Te pedestal ρ≥0.88), fused via MLP(256→128→128).

**CNN-1D** (`cnn_baseline.py`): Intentionally **does not use DeepONet**. Scatter→linear interp to 201-pt grid → 2-channel input [values, mask] → Conv1d encoder → 4× ResidualBlock1D (no BatchNorm) → Conv1d decoder → Softplus → direct 201-pt output. This architectural contrast confirms that the DeepONet branch-trunk pattern is the key performance enabler.

**Physics-constrained loss** (shared across all four, `model.py:PhysicsConstrainedLoss`):
`L = L_MSE + λ₁·L_mono + λ₂·L_bdy + λ₃·L_smooth`
- `L_mono` (0.1): penalizes positive derivatives via ReLU(diff(y)) — only punishes non-monotonic segments
- `L_bdy` (0.5): penalizes non-zero derivative near ρ=0 (magnetic axis symmetry)
- `L_smooth` (0.05): penalizes large second derivatives

**Inference** — all four APIs match the original fitting function signatures:
- ProfileNet: `nn_fit_te/ne/ti()` from `profile_nn.infer`
- LSTM: `nn_fit_te/ne/ti_lstm()` from `profile_nn.infer_lstm`
- CNN: `nn_fit_te/ne/ti_cnn()` from `profile_nn.infer_cnn`
- Transformer: `nn_fit_te/ne/ti_transformer()` from `profile_nn.infer_transformer`

All share the same signature pattern:

**Data format** (`dataset.py`): `.npz` files with `X_rho`, `X_val` (scattered points), `Y` (201-pt label). Ti samples additionally have `X_te_ped_rho`, `X_te_ped_val`. Variable-length inputs are padded to batch-max with boolean masks.

## Notes for Editing

- **Preserve original code when modifying files** — keep the original version as a commented-out block above the modification, marked `=== 原始版本（保留参考）===` / `=== 修改版：... ===`. This applies to any functional change to existing code. Example:
  ```python
  # === 原始版本（保留参考）===
  # from readMDS import readmds
  
  # === 修改版：统一使用 ONETWO 专用读取器 ===
  from readMDS_onetwo import readmds
  ```
- The code contains mixed Chinese and English comments.
- There are no unit tests. Validation is typically done by visual inspection of the 4-panel output plots and checking cumulative current/power integrals against input power.
- Many scripts have `if __name__ == "__main__":` blocks with hardcoded shot numbers or interactive `input()` prompts.
- When modifying `LHW_metis.py`, preserve the exact dictionary keys expected in `cons`, `profil`, and `option`; `LHW_main.py` and `LHW_gui.py` depend on them.
- **All NN ONETWO pipelines share the same structure** — when fixing a bug in one (e.g., unit conversion, Namelist assembly), check and fix the other four (`onetwo_output_nn.py`, `onetwo_output_lstm.py`, `onetwo_output_cnn.py`, `onetwo_output_transformer.py`) as well.
- **ProfileNet, LSTM, and Transformer share `CoordDecoder`** — changes to `model.py:CoordDecoder` affect all three. CNN-1D is intentionally independent (pure ResNet, no DeepONet components).
- **`lower_view_final_nn.py`** — the LHW bridge shared by all four NN pipelines. All use `lower_onetwo_nn()` (NOT individual per-model functions). A bug here propagates everywhere.
- **All four NN models use `PhysicsConstrainedLoss`** — λ weights are consistent across ProfileNet/LSTM/CNN/Transformer for fair comparison. Changing them for one model breaks comparability.
- **Unit bug reference:** Original `onetwo_output.py` writes TEIN in keV; all NN variants correctly write eV (×1000). `compare_profiles.py` auto-detects and normalizes this (if TEIN max < 50, treats as keV and multiplies by 1000).
- **MDSplus signal compatibility:** `readMDS_onetwo.py` uses diagnostic signal names from 2023+ campaigns. Shots from 2019-2021 campaigns may have different signal names in `ReflJ_EAST` / `TXCS_EAST` trees (e.g., `\ne_ReflJ` vs `\ne_Refl`). Te (TS_EAST) is universally compatible.
- **Training set:** 161 shots, ~3200 original samples + 342 Te / 302 ne / 148 Ti real H-mode samples from 52 H-mode shots. H-mode time points selected from H98>1 segments. Covers campaigns from 70k to 149k. Best validation shots: 110000, 125000, 140100, 141000. 2024 has dense coverage in clusters (140087-140096, 140277-140305, 140660-140739, 143600-143634).

## Documentation Maintenance Rules

- **After every model retraining:** update `profile_nn/优化记录.md` — add a new section documenting what changed (data, architecture, hyperparameters), training metrics, deployment recommendation changes, and any bugs fixed.
- **After every server file change (add/delete/rename files that get deployed):** update `profile_nn/使用说明.md` — keep §1.2 (file structure), §13.1 (deployment checklist), §13.5 (deployment scenarios), and any affected Q&A entries in sync with the actual file layout.
- These two docs are the authoritative record of model evolution and deployment state — treat them as part of the codebase, not afterthoughts.
