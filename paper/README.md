# Paper Scripts — 小论文独立脚本集

本目录包含小论文所需的所有脚本、依赖和模型，可独立于主项目运行。

## 文件结构

```
paper/                          # ★ 一键部署：整个文件夹传到服务器即可
├── verify_fit.py               # MDSplus 在线拟合验证
├── offline_fit.py              # 离线拟合 + LHW + inone（不需要 MDSplus）
├── quick_scan.py               # MDSplus 候选炮号扫描
├── scan_compare_shots.py       # 炮号诊断完整性扫描
├── compare_profiles.py         # 五方法剖面交叉对比
│
├── onetwo_output.py            # mtanh ONETWO 管线
├── onetwo_output_nn.py         # ProfileNet ONETWO 管线
├── onetwo_output_lstm.py       # LSTM ONETWO 管线
├── onetwo_output_cnn.py        # CNN-1D ONETWO 管线
├── onetwo_output_transformer.py # Transformer ONETWO 管线
│
├── readMDS_onetwo.py           # EAST MDSplus 数据读取
├── lower_view.py               # LHW 物理模型
├── lower_view_final.py         # LHW 包装（mtanh）
├── lower_view_final_nn.py      # LHW 包装（NN，支持离线模式）
├── mesh2.py                    # 磁面几何计算
├── fitting_mtanh.py            # mtanh 基线拟合器
├── data_clean_new.py           # 数据清洗
├── re_fitting.py               # 台基重拟合
├── left_smooth.py              # 左边界平滑
├── polynomial.py               # 多项式工具
├── geqdsk.py                   # gfile 平衡文件读写
├── Namelist3.py                # Fortran Namelist I/O
│
├── profile_nn/                 # NN 推理包 (10 .py)
└── profile_nn_models/          # 训练好的模型权重 (12 .pt)
```

## 使用方法

### 1. 环境要求

```bash
pip install numpy scipy matplotlib MDSplus torch
```

服务器端注意 HPC SDK 冲突：
```bash
alias pytorch_run='env LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ":" "\n" | grep -v hpc_sdk | tr "\n" ":") python3'
```

### 2. 扫描候选炮号

```bash
# 初扫（2 分钟）
python3 quick_scan.py

# 精细扫描某个区间
python3 quick_scan.py --fine 156000 157000
```

### 3. 拟合验证

```bash
# 单个时间点快速测试
pytorch_run verify_fit.py 156010 --time 5.0

# 全部时间点
pytorch_run verify_fit.py 156010

# 仅 NN 模型（跳过 mtanh）
pytorch_run verify_fit.py 156010 --methods nn
```

输出在 `paper_results/{shot}/`，每个时间点生成一张 `comparison/` 多方法对比图。

### 4. 离线拟合（不需要 MDSplus）

从本地数据文件读取 Te/ne/Ti 散点，无需连接 EAST。

#### 输入格式：JSON 文件

```json
{
  "shot": 156010, "time": 5.0,
  "Te": {
    "rho": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "value": [1500, 1300, 1100, 900, 700, 500, 350, 200, 120],
    "unit": "eV"
  },
  "ne": {
    "rho": [0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95],
    "value": [2.2, 2.1, 1.9, 1.7, 1.4, 1.1, 0.8],
    "unit": "1e19"
  },
  "Ti": {
    "rho": [0.3, 0.5, 0.7, 0.85],
    "value": [0.8, 0.65, 0.45, 0.25],
    "unit": "keV"
  }
}
```

- **Te** 值单位：`"eV"` 或 `"keV"`（脚本自动识别并转换）
- **ne** 值单位：`"1e19"`（即 10¹⁹ m⁻³）
- **Ti** 值单位：`"keV"`
- **Te/ne/Ti 全部可选**——有哪个就拟合哪个，少了的自动跳过
- ρ 点数不要求对齐，各诊断独立

#### 运行

```bash
# 仅拟合（不需要 MDSplus）
pytorch_run offline_fit.py data.json

# 拟合 + LHW 电流驱动 + inone 生成（需要 gfile）
pytorch_run offline_fit.py data.json --gfile g0_input

# 指定 LHW 使用的拟合方法和 Zeff
pytorch_run offline_fit.py data.json --gfile g0_input --lhw-method nn --z-eff 2.2
```

#### 输出

```
offline_results/{shot}_{time}/
├── comparison/
│   ├── Te_comparison.png   # 5 方法叠加对比
│   ├── ne_comparison.png
│   └── Ti_comparison.png
├── fit_result.json         # 所有方法 201 点数值数组
└── lhw/                    # 仅当指定 --gfile 时生成
    ├── LHCD_data_*.txt     # LHW 功率/电流剖面数据
    ├── LHCD_plot_*.png     # LHW 4 面板图
    └── inone               # ONETWO Namelist 输入文件
```

#### 从 MDSplus 导出离线数据

如果已有 MDSplus 连接，可将任意炮号导出为离线 JSON，之后反复使用无需联网：

```python
import json, numpy as np, sys; sys.path.insert(0, '.')
from readMDS_onetwo import readmds

shot, t = 156005, 4.0
data, status, rt = readmds(shot, t)

out = {'shot': shot, 'time': t}
if status['TS_status']:
    out['Te'] = {'rho': data['Te']['TS']['Rho'].tolist(),
                 'value': data['Te']['TS']['data'].tolist(), 'unit': 'eV'}
if status['Refl_status']:
    out['ne'] = {'rho': data['ne']['Refl']['Rho'].tolist(),
                 'value': data['ne']['Refl']['data'].tolist(), 'unit': '1e19'}
if status['TXCS_status']:
    out['Ti'] = {'rho': data['Ti']['TXCS']['Rho'].tolist(),
                 'value': data['Ti']['TXCS']['data'].tolist(), 'unit': 'keV'}

with open(f'data_{shot}_{t:.3f}s.json', 'w') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print('saved')
```

### 5. 批量处理测试集

```bash
for shot in 156005 156010 156100 156400; do
    echo "=== Shot $shot ==="
    pytorch_run verify_fit.py $shot
done
```

## 注意事项

- 本目录所有脚本自包含，不依赖上级目录的 `readMDS.py`、`lower_view*.py` 等
- `verify_fit.py` 需要 MDSplus 连接到 EAST 服务器 (`202.127.204.42`)
- `offline_fit.py` 不需要 MDSplus，本地 JSON 文件即可运行
- NN 模型脚本需用 `pytorch_run` 启动以避开 HPC SDK 冲突；纯 mtanh 可用 `python3`
- 模型权重位于 `profile_nn_models/`，由脚本自动加载
