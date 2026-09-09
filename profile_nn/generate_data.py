"""生成 ProfileNet 训练数据：通过运行现有 mtanh 拟合法对 EAST MDSplus 数据进行标注。

核心流程：
  1. 连接 EAST MDSplus 服务器，读取指定炮号的诊断数据
  2. 按 TS 时间点逐个迭代，调用 readmds() 获取 Te/ne/Ti 散点
  3. 使用现有的 fitting_mtanh.fitting() + robust_interp() 生成 201 点连续剖面作为标签
  4. 将 (散点, 标注剖面) 对保存为 .npz 文件
  5. 可选：生成参数化合成数据用于离线开发测试

使用方法：
    python -m profile_nn.generate_data --shot 81481               # 单个炮号
    python -m profile_nn.generate_data --shots 81481 81482        # 多个炮号
    python -m profile_nn.generate_data --shots 81481 81482 --synthetic 500  # 附加合成数据
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# 将父目录加入搜索路径，以便导入同级的 readMDS_onetwo、fitting_mtanh 等模块
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
# 工具函数（与 onetwo_output.py 中的原始实现保持一致）
# ---------------------------------------------------------------------------

def robust_interp(x, y, datatype, thresholds=None, offsets=None, num_points=201):
    """鲁棒插值：将 mtanh 拟合后的散点插值到均匀 201 点网格。

    处理边界情况：
      - 如果所有 y 值都低于阈值，则整体抬高后再插值
      - 如果插值后最小值仍低于阈值，退回使用抬高策略

    Args:
        x: ρ 坐标数组（任意长度）
        y: 物理量数组（与 x 对应）
        datatype: 'Te', 'ne', 或 'Ti'（用于选择默认阈值和偏移量）
        thresholds: 各诊断类型的最小有效值阈值字典
        offsets: 各诊断类型的整体抬升量字典
        num_points: 输出网格点数，默认 201

    Returns:
        x_fit: (num_points,) 均匀 ρ 网格
        y_fit: (num_points,) 插值后的剖面值
    """
    if thresholds is None:
        thresholds = {"ne": 0.1, "Te": 0.05, "Ti": 0.05}
    if offsets is None:
        offsets = {"ne": 0.05, "Te": 0.05, "Ti": 0.05}
    th = thresholds.get(datatype, 0.0)
    off = offsets.get(datatype, 0.0)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # 过滤掉低于阈值的无效点
    mask = y >= th
    x_clean, y_clean = x[mask], y[mask]

    if len(y_clean) == 0:
        # 所有点都低于阈值：整体抬高，保证正性
        y_lifted = y + off
        f = interp1d(x, y_lifted, kind='linear', fill_value="extrapolate")
        x_fit = np.linspace(0, 1.0, num_points)
        return x_fit, f(x_fit)

    # 正常插值
    f = interp1d(x_clean, y_clean, kind='linear', fill_value="extrapolate")
    x_fit = np.linspace(0, 1.0, num_points)
    y_fit = f(x_fit)

    # 检查插值结果：如果还存在低于阈值的值，退回抬高策略
    if np.min(y_fit) < th:
        y_lifted = y + off
        f = interp1d(x, y_lifted, kind='linear', fill_value="extrapolate")
        x_fit = np.linspace(0, 1.0, num_points)
        y_fit = f(x_fit)

    return x_fit, y_fit


def enforce_monotone_pchip(x, y, num_points=201, decreasing=True):
    """使用保单调 PCHIP 插值强制剖面单调递减。

    等离子体剖面的物理约束：Te、ne、Ti 从芯部到边缘应单调递减。
    PCHIP（分段三次 Hermite 插值多项式）保证插值结果保持原始数据的单调性，
    避免三次样条可能出现的过冲和振荡。

    Args:
        x: ρ 坐标数组
        y: 物理量数组
        num_points: 输出网格点数
        decreasing: True 表示强制单调递减（Te/ne/Ti 的物理特性）

    Returns:
        x_new: (num_points,) 均匀网格
        y_new: (num_points,) 单调递减的剖面
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(x)
    x, y = x[order], y[order]
    # PCHIP 本身保证单调性，通过对 y 取反来实现单调递减约束
    y_proc = -y if decreasing else y
    pchip = PchipInterpolator(x, y_proc)
    x_new = np.linspace(0, 1, num_points)
    y_new = pchip(x_new)
    return x_new, -y_new if decreasing else y_new


# ---------------------------------------------------------------------------
# 保存后验证
# ---------------------------------------------------------------------------

def _validate_saved(filepath, max_x=30.0, max_y=30.0, min_y=-0.5):
    """检查刚保存的 .npz 文件，若数值超出物理合理范围则删除。

    这是一种防御性检查，防止因 MDSplus 数据异常或拟合失败而产生的
    无效训练样本污染数据集。

    Args:
        filepath: .npz 文件路径
        max_x: ρ 坐标的最大允许值（通常不超过 1.0，放宽以容忍边界外点）
        max_y: 物理值的最大允许值
        min_y: 物理值的最小允许值

    Returns:
        True 表示文件有效，False 表示文件已被删除
    """
    try:
        data = np.load(filepath)
        xv, yv = data['X_val'], data['Y']
        data.close()
        x_max, y_max, y_min = float(np.nanmax(xv)), float(np.nanmax(yv)), float(np.nanmin(yv))
        if not np.isfinite(x_max) or x_max > max_x or x_max < -1.0:
            os.remove(filepath)
            return False
        if not np.isfinite(y_max) or y_max > max_y or y_min < min_y:
            os.remove(filepath)
            return False
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 数据提取：从 MDSplus 诊断数据中提取训练样本
# ---------------------------------------------------------------------------

def extract_te_from_data(data, status, real_time, shot, output_dir, h98_value=np.nan):
    """从预读取的 MDSplus 数据中提取一个 Te 训练样本。"""
    if status.get('TS_status', 0) != 1 or 'TS' not in data.get('Te', {}):
        return None

    try:
        x = np.asarray(data['Te']['TS']['Rho'], dtype=np.float64)
        y = np.asarray(data['Te']['TS']['data'], dtype=np.float64)
        datatype = data['Te']['TS']['type']
        y_keV = y / 1000.0                # eV → keV（匹配 mtanh 拟合器内部约定）

        # 预过滤：拒绝极端异常诊断值
        y_valid = y_keV[np.isfinite(y_keV)]
        if len(y_valid) == 0 or y_valid.max() > 15.0 or y_valid.min() < -0.5:
            return None

        # mtanh 拟合法生成标注剖面
        x_fit, y_fit, _, _, _ = fitting(x, y, datatype)
        # 确保输出为 float64 数组（防御 MDSplus 类型问题）
        x_fit = np.asarray(x_fit, dtype=np.float64)
        y_fit = np.asarray(y_fit, dtype=np.float64)
        # 插值到均匀 201 点网格
        x_201, y_201 = robust_interp(x_fit, y_fit, datatype)

        # 后过滤：拒绝拟合失败的剖面（非有限值或超出物理范围）
        if not np.all(np.isfinite(y_201)) or y_201.min() < -0.5 or y_201.max() > 15.0:
            return None

        # 提取台基区域（ρ ≥ 0.88）—— 供 Ti 模型作为辅助输入
        mask_ped = x_fit >= 0.88
        te_ped_x = x_fit[mask_ped]
        te_ped_y = y_fit[mask_ped]

        # 保存 .npz：文件名包含炮号和时间点
        # 时间格式：将小数点替换为 'd'，如 5.30000 → 5d30000
        time_str = f"{real_time:.5f}".replace('.', 'd')
        filepath = os.path.join(output_dir, f"{shot}_Te_{time_str}.npz")
        np.savez(
            filepath,
            X_rho=x.astype(np.float32),
            X_val=y_keV.astype(np.float32),
            Y=y_201.astype(np.float32),
            shot=shot, time=real_time, datatype='Te',
            h98=float(h98_value) if np.isfinite(h98_value) else -1.0,
        )
        # 保存后验证：数值异常时自动删除
        if not _validate_saved(filepath, max_x=15.0, max_y=15.0, min_y=-0.5):
            return None
        return {'te_ped_x': te_ped_x, 'te_ped_y': te_ped_y}
    except Exception as e:
        print(f"  Te 提取失败 @ {real_time:.3f}s: {e}")
        return None


def extract_ne_from_data(data, status, real_time, shot, output_dir, h98_value=np.nan):
    """从诊断数据中提取一个 ne 训练样本。先尝试 TS 通道，失败则回退到 Refl。

    单位处理说明：
      - TS ne 原始单位为 m^-3，需除以 1e19 转换为 10^19 m^-3
      - Refl ne（来自 \\ne_ReflJ）已为 10^19 m^-3 量级，直接使用
      - 两种来源的标签 Y 统一为 10^19 m^-3 单位

    Args:
        data, status, real_time, shot, output_dir: 同 extract_te_from_data

    Returns:
        成功返回 True，失败返回 None
    """
    # 优先尝试 TS 通道（空间分辨率更高）
    ne_result = _try_extract_ne_ts(data, status, real_time, shot, output_dir, h98_value)
    if ne_result is not None:
        return True

    # TS 不可用或数据异常时回退到 Refl（反射计）
    ne_result = _try_extract_ne_refl(data, status, real_time, shot, output_dir, h98_value)
    if ne_result is not None:
        return True

    return None


def _try_extract_ne_ts(data, status, real_time, shot, output_dir, h98_value=np.nan):
    """尝试从 TS 诊断提取 ne。成功返回 True，不可用返回 None。

    TS ne 数据在某些炮号时段可能使用不同的 MDSplus 单位约定，
    当数值超出物理合理范围（>30 × 10^19 m^-3 或 < -1.0）时自动回退。
    """
    if status.get('TS_status', 0) != 1 or 'TS' not in data.get('ne', {}):
        return None

    try:
        x = np.asarray(data['ne']['TS']['Rho'], dtype=np.float64)
        y_raw = np.asarray(data['ne']['TS']['data'], dtype=np.float64)  # 原始单位 m^-3

        if not np.any(np.isfinite(x)):
            return None
        if not np.all(np.isfinite(y_raw)):
            return None

        # 归一化到 10^19 m^-3 单位
        y = y_raw / 1e19

        # 检查 TS ne 是否在物理合理范围内
        y_valid = y[np.isfinite(y)]
        if len(y_valid) == 0 or y_valid.max() > 30.0 or y_valid.min() < -1.0:
            # TS ne 超出合理范围 —— 可能是不同时段 MDSplus 单位约定不同
            # 自动回退到 Refl（由调用方处理）
            return None

        datatype = data['ne']['TS']['type']
        x_fit, y_fit, _, _, _ = fitting(x, y_raw, datatype)
        x_fit = np.asarray(x_fit, dtype=np.float64)
        y_fit = np.asarray(y_fit, dtype=np.float64)
        x_201, y_201 = robust_interp(x_fit, y_fit, datatype)

        if not np.all(np.isfinite(y_201)) or y_201.min() < -1.0 or y_201.max() > 50.0:
            return None

        time_str = f"{real_time:.5f}".replace('.', 'd')
        filepath = os.path.join(output_dir, f"{shot}_ne_{time_str}.npz")
        np.savez(
            filepath,
            X_rho=x.astype(np.float32),
            X_val=y.astype(np.float32),
            Y=y_201.astype(np.float32),
            shot=shot, time=real_time, datatype='ne',
            h98=float(h98_value) if np.isfinite(h98_value) else -1.0,
        )
        if not _validate_saved(filepath, max_x=25.0, max_y=25.0, min_y=-0.5):
            return None
        return True
    except Exception as e:
        print(f"  ne TS 提取失败 @ {real_time:.3f}s: {e}")
        return None


def _try_extract_ne_refl(data, status, real_time, shot, output_dir, h98_value=np.nan):
    """从反射计 (Reflectometer) 提取 ne 作为回退方案。

    Refl 的 \\ne_ReflJ 数据已在 ~10^19 m^-3 量级（典型范围 0.1-5），
    因此不需要除以 1e19，直接使用即可。
    """
    if status.get('Refl_status', 0) != 1 or 'Refl' not in data.get('ne', {}):
        return None

    try:
        x = np.asarray(data['ne']['Refl']['Rho'], dtype=np.float64)
        y_raw = np.asarray(data['ne']['Refl']['data'], dtype=np.float64)

        if not np.any(np.isfinite(x)):
            return None
        if not np.all(np.isfinite(y_raw)):
            return None

        # Refl 数据已在 10^19 m^-3 单位 —— 直接使用
        y = y_raw

        y_valid = y[np.isfinite(y)]
        if len(y_valid) == 0 or y_valid.max() > 20.0 or y_valid.min() < 0.0:
            return None

        datatype = data['ne']['Refl']['type']
        # y 已经使用正确单位，直接传入拟合函数
        x_fit, y_fit, _, _, _ = fitting(x, y, datatype)
        x_fit = np.asarray(x_fit, dtype=np.float64)
        y_fit = np.asarray(y_fit, dtype=np.float64)
        x_201, y_201 = robust_interp(x_fit, y_fit, datatype)

        if not np.all(np.isfinite(y_201)) or y_201.min() < -1.0 or y_201.max() > 50.0:
            return None

        time_str = f"{real_time:.5f}".replace('.', 'd')
        filepath = os.path.join(output_dir, f"{shot}_ne_{time_str}.npz")
        np.savez(
            filepath,
            X_rho=x.astype(np.float32),
            X_val=y.astype(np.float32),
            Y=y_201.astype(np.float32),
            shot=shot, time=real_time, datatype='ne',
            h98=float(h98_value) if np.isfinite(h98_value) else -1.0,
        )
        if not _validate_saved(filepath, max_x=25.0, max_y=25.0, min_y=-0.5):
            return None
        return True
    except Exception as e:
        print(f"  ne Refl 提取失败 @ {real_time:.3f}s: {e}")
        return None


def extract_ti_from_data(data, status, real_time, shot, te_ped_data, output_dir, h98_value=np.nan):
    """从诊断数据中提取一个 Ti 训练样本。需要 Te 台基数据作为先决条件。

    Ti 提取的特殊性：
      - TXCS（X 射线晶体谱仪）通道数少（3-15 个），信息稀疏
      - 原始 mtanh 拟合中 Ti 依赖 Te 台基（ρ ≥ 0.88）来约束边界行为
      - 因此 Ti 提取必须在 Te 提取成功之后进行（由 te_ped_data 提供台基信息）
      - 最后通过 enforce_monotone_pchip 强制单调递减

    Args:
        data, status, real_time, shot, output_dir: 同其他提取函数
        te_ped_data: extract_te_from_data 返回的台基字典，包含 'te_ped_x' 和 'te_ped_y'

    Returns:
        成功返回 True，失败返回 None
    """
    if te_ped_data is None:
        return None
    if status.get('TXCS_status', 0) != 1 or 'TXCS' not in data.get('Ti', {}):
        return None

    try:
        x = np.asarray(data['Ti']['TXCS']['Rho'], dtype=np.float64)
        y = np.asarray(data['Ti']['TXCS']['data'], dtype=np.float64)
        datatype = data['Ti']['type']        # 注意：type 在 Ti 层级而非 TXCS 子级

        # 调用 mtanh 拟合，传入 Te 台基数据作为辅助约束
        x_fit, y_fit, _, _, _ = fitting(
            x, y, datatype,
            te_ped_x=te_ped_data['te_ped_x'],
            te_ped_y=te_ped_data['te_ped_y'],
        )
        x_fit = np.asarray(x_fit, dtype=np.float64)
        y_fit = np.asarray(y_fit, dtype=np.float64)
        x_201, y_201 = robust_interp(x_fit, y_fit, datatype)
        # 强制单调递减 —— Ti 剖面的物理约束
        x_201, y_201 = enforce_monotone_pchip(x_201, y_201)

        if not np.all(np.isfinite(y_201)) or y_201.min() < -0.5 or y_201.max() > 20.0:
            return None

        time_str = f"{real_time:.5f}".replace('.', 'd')
        filepath = os.path.join(output_dir, f"{shot}_Ti_{time_str}.npz")
        np.savez(
            filepath,
            X_rho=x.astype(np.float32),
            X_val=y.astype(np.float32),
            # Ti 额外存储台基数据 —— 训练时 ProfileNet_Ti 的双编码器需要
            X_te_ped_rho=te_ped_data['te_ped_x'].astype(np.float32),
            X_te_ped_val=te_ped_data['te_ped_y'].astype(np.float32),
            Y=y_201.astype(np.float32),
            shot=shot, time=real_time, datatype='Ti',
            h98=float(h98_value) if np.isfinite(h98_value) else -1.0,
        )
        if not _validate_saved(filepath, max_x=20.0, max_y=20.0, min_y=-0.5):
            return None
        return True
    except Exception as e:
        print(f"  Ti 提取失败 @ {real_time:.3f}s: {e}")
        return None


# ---------------------------------------------------------------------------
# 合成数据生成（用于离线开发测试，无需 MDSplus 连接）
# ---------------------------------------------------------------------------

def generate_synthetic_samples(output_dir, num_samples=500, seed=42):
    """生成参数化合成训练样本，用于无 MDSplus 环境下的开发和测试。

    合成策略：
      使用随机参数化的 mtanh 类剖面（芯部平坦 + 台基下降），
      在剖面曲线上撒点并添加高斯噪声，模拟真实诊断散点。

    参数范围：
      - Te: 台基高度 0.5~5.0 keV, 散点数 15~30, 噪声 3%
      - ne: 台基高度 1.0~6.0 ×10^19, 散点数 12~22, 噪声 5%
      - Ti: 台基高度 0.3~4.0 keV, 散点数 3~12, 噪声 4%

    Args:
        output_dir: 输出目录
        num_samples: 每种诊断类型生成的样本数
        seed: 随机种子（保证可复现）
    """
    rng = np.random.default_rng(seed)
    print(f"为每种诊断类型生成 {num_samples} 个合成样本...")

    for datatype, y_range, noise_frac, n_pts_range in [
        ('Te',  (0.5, 5.0),   0.03, (15, 30)),   # keV
        ('ne',  (1.0, 6.0),   0.05, (12, 22)),   # 1e19 m^-3
        ('Ti',  (0.3, 4.0),   0.04, (3, 12)),    # keV
    ]:
        for i in range(num_samples):
            # ---- 随机台基参数 ----
            ped_height = rng.uniform(*y_range)              # 台基高度
            ped_pos = rng.uniform(0.88, 0.96)              # 台基位置（ρ 坐标）
            ped_width = rng.uniform(0.02, 0.08)            # 台基宽度
            core_val = ped_height * rng.uniform(1.5, 3.0)  # 芯部值（台基的 1.5-3 倍）

            # ---- 生成光滑剖面：芯部平坦 + mtanh 台基 ----
            rho_fine = np.linspace(0, 1, 201)
            profile = core_val - (core_val - ped_height) * (
                0.5 * (1 + np.tanh((rho_fine - ped_pos) / ped_width))
            )
            # 边界偏移：确保边缘处剖面 > 0
            profile = profile - profile[-1] + rng.uniform(0.01, 0.1)
            profile = np.maximum(profile, 0.01)

            # ---- 生成模拟诊断散点 ----
            n_pts = rng.integers(*n_pts_range)
            rho_scattered = np.sort(rng.uniform(0.0, 1.02, n_pts))
            rho_scattered = np.clip(rho_scattered, 0.0, 1.0)
            y_true_at_pts = np.interp(rho_scattered, rho_fine, profile)
            # 添加高斯噪声模拟测量误差
            y_noisy = y_true_at_pts * (1 + rng.normal(0, noise_frac, n_pts))
            y_noisy = np.maximum(y_noisy, 0.005)  # 确保正性

            if datatype == 'Ti':
                # Ti 需要合成 Te 台基数据
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

        print(f"  {datatype}: {num_samples} 个合成样本已保存")


def generate_h_mode_samples(output_dir, num_samples=300, seed=123):
    """生成 H-mode 参数化合成训练样本：窄台基 + 高峰值度。

    与 generate_synthetic_samples（L-mode）的关键区别：
      - ped_width: 0.005-0.025（窄 3-6 倍）
      - core/ped ratio: 2.0-4.5（高峰值度）
      - ped_pos: 0.90-0.98（台基更靠外）

    Args:
        output_dir: 输出目录
        num_samples: 每种诊断类型生成的 H-mode 样本数
        seed: 随机种子
    """
    rng = np.random.default_rng(seed)
    print(f"Generating {num_samples} H-mode synthetic samples per diagnostic...")

    for datatype, y_range, noise_frac, n_pts_range in [
        ('Te',  (1.0, 7.0),   0.02, (18, 35)),
        ('ne',  (2.0, 8.0),   0.03, (14, 25)),
    ]:
        for i in range(num_samples):
            ped_height = rng.uniform(*y_range)
            ped_pos = rng.uniform(0.90, 0.98)
            ped_width = rng.uniform(0.005, 0.025)       # narrow: 3-6x vs L-mode
            core_val = ped_height * rng.uniform(2.0, 4.5)  # high peaking

            rho_fine = np.linspace(0, 1, 201)
            profile = core_val - (core_val - ped_height) * (
                0.5 * (1 + np.tanh((rho_fine - ped_pos) / ped_width))
            )
            profile = profile - profile[-1] + rng.uniform(0.005, 0.05)
            profile = np.maximum(profile, 0.005)

            n_pts = rng.integers(*n_pts_range)
            rho_scattered = np.sort(rng.uniform(0.0, 1.02, n_pts))
            rho_scattered = np.clip(rho_scattered, 0.0, 1.0)
            y_true_at_pts = np.interp(rho_scattered, rho_fine, profile)
            y_noisy = y_true_at_pts * (1 + rng.normal(0, noise_frac, n_pts))
            y_noisy = np.maximum(y_noisy, 0.005)

            np.savez(
                os.path.join(output_dir, f"synthetic_H_{datatype}_{i:05d}.npz"),
                X_rho=rho_scattered.astype(np.float32),
                X_val=y_noisy.astype(np.float32),
                Y=profile.astype(np.float32),
                shot=-2, time=float(i), datatype=datatype,
            )

        print(f"  H-mode {datatype}: {num_samples} samples saved")


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

def main():
    """主入口：解析命令行参数，从 MDSplus 或合成数据生成训练样本。"""
    parser = argparse.ArgumentParser(description='生成 ProfileNet 训练数据')
    parser.add_argument('--shots', type=int, nargs='+',
                        help='EAST 炮号列表（空格分隔）')
    parser.add_argument('--output', type=str, default='./profile_nn_data',
                        help='.npz 文件输出目录')
    parser.add_argument('--synthetic', type=int, default=0, metavar='N',
                        help='额外生成 N 个 L-mode 合成样本/每种诊断类型')
    parser.add_argument('--h-mode', type=int, default=0, metavar='N',
                        help='额外生成 N 个 H-mode 合成样本（窄台基+高峰值度）')
    parser.add_argument('--time-limit', type=int, default=0, metavar='N',
                        help='每炮号最多处理前 N 个时间片（0=全部）')
    parser.add_argument('--h98-min', type=float, default=None,
                        help='H98 下界筛选（只生成 H98 >= 此值的时间点）')
    parser.add_argument('--h98-max', type=float, default=None,
                        help='H98 上界筛选（只生成 H98 <= 此值的时间点）')
    parser.add_argument('--diagnostics', type=str, nargs='+',
                        default=['Te', 'ne', 'Ti'],
                        choices=['Te', 'ne', 'Ti'],
                        help='只生成指定的诊断类型（默认全部）')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 合成数据始终可用（无需 MDSplus 连接）
    if args.synthetic > 0:
        generate_synthetic_samples(str(output_dir), num_samples=args.synthetic)
    if args.h_mode > 0:
        generate_h_mode_samples(str(output_dir), num_samples=args.h_mode)

    # 真实 MDSplus 数据
    if args.shots:
        if not HAS_MDS:
            print("错误: MDSplus 不可用。请安装 MDSplus 以生成真实数据。")
            print("使用 --synthetic N 可生成合成测试数据代替。")
            sys.exit(1)

        MDS_IP = '202.127.204.42'
        conn = Connection(MDS_IP)

        for shot in args.shots:
            print(f"\n{'='*50}")
            print(f"处理炮号 {shot}")
            print(f"{'='*50}")

            # 获取 TS 时间点列表作为迭代基准
            try:
                conn.openTree('TS_EAST', shot)
                TS_times = conn.get(r'dim_of(\Te_coreTS)').data()
                conn.closeTree('TS_EAST', shot)
            except Exception as e:
                print(f"无法读取炮号 {shot} 的 TS 时间点: {e}")
                continue

            if args.time_limit > 0:
                TS_times = TS_times[:args.time_limit]

            print(f"找到 {len(TS_times)} 个时间点")

            te_count = ne_count = ti_count = 0

            for t in TS_times:
                # 每个时间点调用一次 readmds，结果在 Te/ne/Ti 提取间共享
                try:
                    data, status, real_time = readmds(shot, t)
                except Exception as e:
                    print(f"  readmds 失败 @ {t:.3f}s: {e}")
                    continue

                # ---- H98 提取与筛选 ----
                h98_value = float(data.get('H98', {}).get('value', np.nan))
                if args.h98_min is not None and np.isfinite(h98_value) and h98_value < args.h98_min:
                    continue  # H98 低于下限，跳过该时间点
                if args.h98_max is not None and np.isfinite(h98_value) and h98_value > args.h98_max:
                    continue  # H98 高于上限，跳过该时间点

                # ---- Te 提取（必须在 ne 和 Ti 之前） ----
                te_ped = None
                if 'Te' in args.diagnostics:
                    if status.get('TS_status', 0) == 1:
                        te_ped = extract_te_from_data(data, status, real_time, shot, str(output_dir), h98_value)
                        if te_ped is not None:
                            te_count += 1
                        else:
                            continue  # Te 失败则跳过 ne 和 Ti（需要 Te 台基数据）
                    else:
                        continue  # TS 不可用则跳过后面的提取
                else:
                    # 不生成 Te 但需要台基数据给 ne/Ti：仍做 Te 提取但不保存
                    if status.get('TS_status', 0) == 1:
                        te_ped = extract_te_from_data(data, status, real_time, shot, str(output_dir), h98_value)
                        if te_ped is None:
                            continue
                    else:
                        continue

                # ---- ne 提取 ----
                if 'ne' in args.diagnostics:
                    ne_ok = extract_ne_from_data(data, status, real_time, shot, str(output_dir), h98_value)
                    if ne_ok:
                        ne_count += 1

                # ---- Ti 提取（依赖 te_ped） ----
                if 'Ti' in args.diagnostics and te_ped is not None:
                    ti_ok = extract_ti_from_data(data, status, real_time, shot, te_ped, str(output_dir), h98_value)
                    if ti_ok:
                        ti_count += 1

            print(f"炮号 {shot} 完成: Te={te_count}, ne={ne_count}, Ti={ti_count}")

    # 输出汇总统计
    all_files = sorted(output_dir.glob('*.npz'))
    te_files = [f for f in all_files if '_Te_' in f.name]
    ne_files = [f for f in all_files if '_ne_' in f.name]
    ti_files = [f for f in all_files if '_Ti_' in f.name]
    print(f"\n{'='*50}")
    print(f"数据集总计: Te={len(te_files)}, ne={len(ne_files)}, Ti={len(ti_files)}")
    print(f"输出目录: {output_dir.absolute()}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
