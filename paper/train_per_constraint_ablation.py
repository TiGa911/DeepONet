# -*- coding: utf-8 -*-
"""
train_per_constraint_ablation.py — 逐约束消融训练驱动器（审稿意见 M4 提前执行）

对 Te 诊断、4 个架构 × 4 个单约束关闭配置（mono_off / bdy_off / smooth_off / log_off）
共 16 次训练，每次训练保留其余三个物理约束为默认权重。

输出：
  profile_nn_models_ablation_pc/{config}/{Arch}_Te.pt

用法:
  python paper/train_per_constraint_ablation.py --smoke --epochs 5   # 冒烟测试（测速）
  python paper/train_per_constraint_ablation.py                      # 全量 16 次训练
"""
import argparse
import os
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

parser = argparse.ArgumentParser()
parser.add_argument('--smoke', action='store_true', help='冒烟模式：每配置仅 5 轮')
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--configs', nargs='+',
                    default=['mono_off', 'bdy_off', 'smooth_off', 'log_off'],
                    help='要训练的配置子集')
parser.add_argument('--archs', nargs='+',
                    default=['ProfileNet', 'LSTM', 'CNN-1D', 'Transformer'],
                    help='要训练的架构子集')
args = parser.parse_args()

DATA_DIR = os.path.join(ROOT, 'profile_nn_data')
OUTPUT_BASE = os.path.join(ROOT, 'profile_nn_models_ablation_pc')

# 配置 → 权重覆盖（其余权重用脚本默认值：mono 0.05, bdy 0.05, smooth 0.01, log 0.1）
CONFIG_WEIGHTS = {
    'mono_off':   {'--w-mono': '0'},
    'bdy_off':    {'--w-bdy': '0'},
    'smooth_off': {'--w-smooth': '0'},
    'log_off':    {'--w-log': '0'},
}

ARCH_MODULE = {
    'ProfileNet': 'profile_nn.train',
    'LSTM': 'profile_nn.lstm_baseline',
    'CNN-1D': 'profile_nn.cnn_baseline',
    'Transformer': 'profile_nn.transformer_baseline',
}


def run_one(arch, config, epochs, log_fh):
    out_dir = os.path.join(OUTPUT_BASE, config)
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, '-m', ARCH_MODULE[arch],
           '--data', DATA_DIR, '--datatype', 'Te',
           '--epochs', str(epochs), '--output', out_dir]
    for flag, val in CONFIG_WEIGHTS[config].items():
        cmd += [flag, val]
    tag = f'[{arch:>12s} / {config:>11s}]'
    print(f'{tag} start: ' + ' '.join(cmd[-6:]), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, encoding='utf-8',
                          errors='replace')
    dt = time.time() - t0
    tail = '\n'.join((proc.stdout or '').strip().splitlines()[-6:])
    status = 'OK' if proc.returncode == 0 else f'FAIL rc={proc.returncode}'
    line = f'{tag} {status}  {dt/60:.1f} min\n{tail}\n' + '=' * 60 + '\n'
    print(line, flush=True)
    log_fh.write(line)
    log_fh.flush()
    return proc.returncode == 0


def main():
    epochs = 5 if args.smoke else args.epochs
    log_path = os.path.join(OUTPUT_BASE, 'training_log.txt')
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    log_fh = open(log_path, 'a', encoding='utf-8')
    log_fh.write(f'\n===== session {time.strftime("%Y-%m-%d %H:%M:%S")} '
                 f'smoke={args.smoke} epochs={epochs} =====\n')
    print(f'Smoke={args.smoke}, epochs={epochs}, '
          f'{len(args.archs)} archs x {len(args.configs)} configs = '
          f'{len(args.archs)*len(args.configs)} runs')
    ok = 0
    for arch in args.archs:
        for config in args.configs:
            if run_one(arch, config, epochs, log_fh):
                ok += 1
    print(f'Done: {ok}/{len(args.archs)*len(args.configs)} succeeded')
    log_fh.close()


if __name__ == '__main__':
    main()
