# -*- coding: utf-8 -*-
"""
trunk_spectrum.py — CoordDecoder 主干基函数的空间频率谱分析（后续计划：主干可解释性）

分析共享 CoordDecoder（trunk net）输出的 128 个基函数 b_k(ρ) 的空间频率分布，
解释"主干在密集诊断上轻微过度平滑"的机理（审稿意见 m4 的定量支撑）。

用法:
  python trunk_spectrum.py

输出:
  paper_results/figures/fig_trunk_spectrum.png
  stdout 汇总统计（主导波数分布、高频基占比）
"""
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from profile_nn.model import CoordDecoder

MODEL_PATH = os.path.join(SCRIPT_DIR, 'profile_nn_models', 'ProfileNet_Te.pt')
FIG_DIR = os.path.join(SCRIPT_DIR, 'paper_results', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

RHO = np.linspace(0, 1, 1001)


def load_trunk():
    decoder = CoordDecoder(hidden=128, output_dim=128, n_freq=32)
    ckpt = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    sd = ckpt['model_state_dict']
    trunk_sd = {k.replace('decoder.', ''): v
                for k, v in sd.items() if k.startswith('decoder.')}
    decoder.load_state_dict(trunk_sd)
    decoder.eval()
    return decoder


def dominant_wavenumber(y):
    """FFT 峰值频率（周期数 / 单位 ρ）。"""
    y = np.asarray(y)
    y = y - y.mean()
    fft = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), d=RHO[1] - RHO[0])
    return float(freqs[np.argmax(fft)])


def main():
    decoder = load_trunk()
    with torch.no_grad():
        basis = decoder(torch.from_numpy(RHO.astype(np.float32))).numpy()  # (1001, 128)

    ks = np.array([dominant_wavenumber(basis[:, i]) for i in range(128)])

    print(f'Trunk basis spectrum ({len(ks)} basis functions):')
    print(f'  dominant wavenumber (cycles/rho): median={np.median(ks):.1f}, '
          f'90th pct={np.percentile(ks, 90):.1f}, max={ks.max():.1f}')
    print(f'  low-frequency bases (k<10): {(ks < 10).mean()*100:.0f}%')
    print(f'  high-frequency bases (k>40): {(ks > 40).mean()*100:.0f}%')

    # ---- 图 ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300)

    ax = axes[0]
    for idx in [0, 5, 20, 50, 90, 127]:
        y = basis[:, idx]
        y = y / (np.abs(y).max() + 1e-12)
        ax.plot(RHO, y, lw=1.0, label=f'$b_{{{idx+1}}}$ (k={ks[idx]:.0f})')
    ax.set_xlabel(r'$\rho$')
    ax.set_ylabel('normalized basis')
    ax.set_title('(a) Example trunk basis functions')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(ks, bins=30, color='#457B9D', alpha=0.8)
    ax.axvline(np.median(ks), color='red', ls='--', lw=1,
               label=f'median k = {np.median(ks):.1f}')
    ax.set_xlabel('dominant wavenumber (cycles / $\\rho$)')
    ax.set_ylabel('count')
    ax.set_title('(b) Spatial-frequency spectrum of the 128 basis functions')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, 'fig_trunk_spectrum.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


if __name__ == '__main__':
    main()
