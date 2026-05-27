# -*- coding: utf-8 -*-
# Debug version of fitting_mtanh.py
# Added by ChatGPT for step-by-step Te fitting diagnosis.
#
# Main additions:
# 1) debug flag and debug_dir in fitting()
# 2) save raw / cleaned / stage outputs to npz + png
# 3) verbose summaries at each critical step
# 4) optional ability to bypass left smoothing / smooth transition for isolation tests

import os
import json
import numpy as np
from matplotlib import pyplot as plt
from numpy import exp
from scipy import interpolate
from scipy.optimize import leastsq, least_squares
from scipy.interpolate import CubicSpline
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import UnivariateSpline

from data_clean_new import clean
from left_smooth import new_sm_extremity
from readMDS import readmds


def _ensure_dir(path):
    if path is None:
        path = './debug_fit'
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def _to_numpy(x):
    try:
        return np.asarray(x, dtype=float)
    except Exception:
        return np.asarray(x)


def _arr_summary(name, arr):
    arr = _to_numpy(arr)
    if arr.size == 0:
        return {
            'name': name,
            'size': 0,
            'min': None,
            'max': None,
            'mean': None,
        }
    return {
        'name': name,
        'size': int(arr.size),
        'min': float(np.nanmin(arr)),
        'max': float(np.nanmax(arr)),
        'mean': float(np.nanmean(arr)),
    }


def _print_summary(name, x, y):
    sx = _arr_summary(name + '.x', x)
    sy = _arr_summary(name + '.y', y)
    print('--- {} ---'.format(name))
    print('x: size={size}, min={min}, max={max}, mean={mean}'.format(**sx))
    print('y: size={size}, min={min}, max={max}, mean={mean}'.format(**sy))


def _save_npz(debug_dir, filename, **kwargs):
    path = os.path.join(debug_dir, filename)
    np.savez(path, **kwargs)
    print('[DEBUG] saved npz -> {}'.format(path))


def _save_json(debug_dir, filename, data):
    path = os.path.join(debug_dir, filename)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print('[DEBUG] saved json -> {}'.format(path))


def _save_plot(debug_dir, filename, curves, title=''):
    """
    curves: list of dicts, each dict can have keys:
        x, y, style='o', label='', color=None, ms=6, lw=2, alpha=1.0
    """
    plt.figure(figsize=(8, 6), dpi=150)
    for c in curves:
        x = c.get('x', [])
        y = c.get('y', [])
        style = c.get('style', '-')
        label = c.get('label', '')
        color = c.get('color', None)
        ms = c.get('ms', 6)
        lw = c.get('lw', 2)
        alpha = c.get('alpha', 1.0)
        if 'o' in style or '.' in style or 'x' in style or '^' in style:
            plt.plot(x, y, style, label=label, color=color, ms=ms, alpha=alpha)
        else:
            plt.plot(x, y, style, label=label, color=color, lw=lw, alpha=alpha)
    plt.xlabel(r'$\rho$')
    plt.ylabel('value')
    if title:
        plt.title(title)
    plt.xlim(0.0, 1.05)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(debug_dir, filename)
    plt.savefig(path)
    plt.close()
    print('[DEBUG] saved plot -> {}'.format(path))


def _sort_xy(x, y):
    x = _to_numpy(x)
    y = _to_numpy(y)
    idx = np.argsort(x)
    return x[idx], y[idx]


def judge_plasma_mode(h98, threshold=0.7):
    """
    根据 H98 判定等离子体模式
    H98 >= threshold -> H mode
    H98 < threshold  -> L mode
    """
    if h98 is None:
        return 'H'
    try:
        if np.isnan(h98):
            return 'H'
    except Exception:
        pass

    if h98 >= threshold:
        return 'H'
    else:
        return 'L'


def lmode_spline_fit(x, y, num_points=201, x_end=1.0):
    """
    L模直接全剖面 spline 拟合，不做台基分段
    """
    x = np.asarray(x)
    y = np.asarray(y)

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    x_unique, unique_idx = np.unique(x, return_index=True)
    y_unique = y[unique_idx]

    if len(x_unique) < 4:
        x_fit = np.linspace(np.min(x_unique), min(np.max(x_unique), x_end), num_points)
        y_fit = np.interp(x_fit, x_unique, y_unique)
        return x_fit, y_fit

    spline = UnivariateSpline(x_unique, y_unique)
    x_fit = np.linspace(np.min(x_unique), min(np.max(x_unique), x_end), num_points)
    y_fit = spline(x_fit)

    return x_fit, y_fit


def front_ped_fitting(R, n):
    R1, n1 = [], []
    for i in range(len(R)):
        if 0 <= R[i] < 0.9:
            R1.append(R[i])
            n1.append(n[i])
    x = np.linspace(0, 0.8, 100)
    func = interpolate.UnivariateSpline(R1, n1)
    return x, func(x)


def front_ped_fitting_Te(R, n):
    R = np.asarray(R)
    n = np.asarray(n)
    R1, n1 = [], []
    for i in range(len(R)):
        if 0 <= R[i] < 0.95:
            R1.append(R[i])
            n1.append(n[i])
    x = np.linspace(0, 0.85, 100)

    index = np.where((0.8 < R) & (R <= 1.0))
    R2 = R[index]
    N2 = n[index]

    if len(R2) > 0:
        max_value = np.max(N2)
        min_value = np.min(N2)

        if len(N2) > 1:
            N2_without_max = np.delete(N2, np.argmax(N2))
            mean_value = np.mean(N2_without_max)
        else:
            mean_value = N2[0]

        # top = mean_value

        # R1.append(0.9)
        # n1.append(top)
        func = interpolate.UnivariateSpline(R1, n1)
        peak_value1 = func(0.85)
        peak_value2 = (1 - 0.4) * mean_value + 0.4 * max_value
        # peak_value2 = func(0.9)

        if peak_value2 > peak_value1:
            peak_value2 = peak_value1
    else:
        print('数据不足')
        func = interpolate.UnivariateSpline(R1, n1)
        peak_value1 = func(0.85)
        peak_value2 = func(0.9)
        min_value = 0

    return x, func(x), peak_value1, peak_value2, R2, N2, min_value


def mtanh(x, factor):
    A = factor[0]
    B = factor[1]
    alpha = factor[2]
    x_sys = factor[3]
    w = factor[4]
    beta = factor[5]
    z = (x_sys - x) / w
    mtanh_fun = ((1 + alpha * z) * exp(z) - (1 + beta * z) * exp(-z)) / (exp(z) + exp(-z))
    y = A * mtanh_fun + B
    return y


def residuals(p, y, x):
    return y - mtanh(x, p)


def ped_fitting(R, n, index):
    ped_pos = 0.91
    h = 3.59
    w = 0.05
    slope1 = 2
    slope2 = -3
    fac_init = np.array((h / 2, h / 2, slope1, ped_pos, w, slope2))

    R1 = R[index]
    n1 = n[index]
    factor = leastsq(residuals, fac_init, args=(n1, R1), maxfev=50000)
    ne_ped = factor[0][0] + factor[0][1]
    ne_width = 2 * factor[0][4]
    x1 = np.arange(np.min(R1), 1.2, 0.01)
    y1 = mtanh(x1, factor[0])
    return np.array((x1, y1, ne_ped, ne_width, factor), dtype=object)


def ped_fitting_TE(mtanhR, peak_value, min_value, ped_pos, w, slope1, slope2):
    fac_init = np.array(((peak_value + min_value) / 2,
                         (peak_value - min_value) / 2,
                         slope1, ped_pos, w, slope2))
    x1 = np.arange(mtanhR, 1.2, 0.01)
    y1 = mtanh(x1, fac_init)
    return np.array((x1, y1), dtype=object)



def _core_value_and_slope_at(x_core, y_core, x0):
    x_core = np.asarray(x_core, dtype=float)
    y_core = np.asarray(y_core, dtype=float)

    idx = np.argsort(x_core)
    x_core = x_core[idx]
    y_core = y_core[idx]

    x_unique, unique_idx = np.unique(x_core, return_index=True)
    y_unique = y_core[unique_idx]

    if len(x_unique) < 2:
        return float(y_unique[0]), 0.0

    if len(x_unique) >= 4:
        spl = UnivariateSpline(x_unique, y_unique, s=0)
        y0 = float(spl(x0))
        dy0 = float(spl.derivative()(x0))
        return y0, dy0

    y0 = float(np.interp(x0, x_unique, y_unique))
    j = np.searchsorted(x_unique, x0)
    if j <= 0:
        j0, j1 = 0, 1
    elif j >= len(x_unique):
        j0, j1 = len(x_unique) - 2, len(x_unique) - 1
    else:
        j0, j1 = j - 1, j
    dx = x_unique[j1] - x_unique[j0]
    dy0 = 0.0 if abs(dx) < 1e-12 else float((y_unique[j1] - y_unique[j0]) / dx)
    return y0, dy0


def _mtanh_value_and_slope_at(x0, peak_value, min_value, ped_pos, w, slope1, slope2):
    fac = np.array(((peak_value + min_value) / 2.0,
                    (peak_value - min_value) / 2.0,
                    slope1, ped_pos, w, slope2), dtype=float)
    y0 = float(mtanh(np.array([x0], dtype=float), fac)[0])
    dx = 1e-4
    y_minus = float(mtanh(np.array([x0 - dx], dtype=float), fac)[0])
    y_plus = float(mtanh(np.array([x0 + dx], dtype=float), fac)[0])
    dy0 = (y_plus - y_minus) / (2.0 * dx)
    return y0, dy0


def fit_te_peak_and_start_ls(x_part, y_part, R2, N2, min_value,
                             peak_init, mtanhR_init,
                             ped_pos, w, slope1, slope2,
                             peak_cap=None):
    x_part = np.asarray(x_part, dtype=float)
    y_part = np.asarray(y_part, dtype=float)
    R2 = np.asarray(R2, dtype=float)
    N2 = np.asarray(N2, dtype=float)

    if len(R2) < 2:
        return float(peak_init), float(mtanhR_init)

    peak_floor = max(float(min_value) + 1e-4, 0.2 * float(peak_init))
    peak_upper = float(peak_cap) if peak_cap is not None else float(max(np.max(N2), peak_init))
    peak_upper = max(peak_upper, peak_floor + 1e-4)

    lower = np.array([peak_floor, 0.80], dtype=float)
    upper = np.array([peak_upper, 0.88], dtype=float)
    p0 = np.array([float(np.clip(peak_init, lower[0], upper[0])), float(np.clip(mtanhR_init, lower[1], upper[1]))], dtype=float)

    def _resid(p):
        peak_value2, mtanhR = p
        fac = np.array(((peak_value2 + min_value) / 2.0,
                        (peak_value2 - min_value) / 2.0,
                        slope1, ped_pos, w, slope2), dtype=float)

        # pedestal candidate region fit
        y_ped_data = mtanh(R2, fac)
        r_data = (y_ped_data - N2)

        # continuity at pedestal start
        y_core0, slope_core0 = _core_value_and_slope_at(x_part, y_part, mtanhR)
        y_mt0, slope_mt0 = _mtanh_value_and_slope_at(mtanhR, peak_value2, min_value, ped_pos, w, slope1, slope2)

        scale_y = max(np.nanmax(np.abs(N2)), 1e-3)
        scale_s = max(abs(slope_core0), abs(slope_mt0), 1e-3)

        r_join_y = np.array([(y_mt0 - y_core0) / scale_y * 3.0])
        r_join_s = np.array([(slope_mt0 - slope_core0) / scale_s * 1.5])

        # mild regularization around original start point
        r_reg = np.array([(mtanhR - mtanhR_init) / 0.02 * 0.2])

        return np.concatenate([r_data, r_join_y, r_join_s, r_reg])

    try:
        res = least_squares(_resid, p0, bounds=(lower, upper), loss='soft_l1', f_scale=0.5, max_nfev=2000)
        peak_opt, start_opt = res.x
        return float(peak_opt), float(start_opt)
    except Exception:
        return float(peak_init), float(mtanhR_init)

def find_delete(xorigin, xclean, yorigin):
    yorigin = yorigin
    new_xorigin = np.full_like(xorigin, -1)
    new_yorigin = np.full_like(yorigin, -1)

    mask = ~np.in1d(xorigin, xclean)
    new_xorigin[mask] = xorigin[mask]
    new_yorigin[mask] = yorigin[mask]

    index = ~np.isnan(new_yorigin) & (new_yorigin != -1)
    new_xorigin = new_xorigin[index]
    new_yorigin = new_yorigin[index]

    return new_xorigin, new_yorigin


def new_sm_extremity_final(rawx, rawy, rp, neworder, mode='left'):
    if mode not in ['left', 'right']:
        raise ValueError("mode must be 'left' or 'right'.")
    if neworder <= 0:
        raise ValueError('neworder must be a positive number.')

    sorted_indices = np.argsort(rawx, axis=0)
    rawx, rawy = rawx[sorted_indices], rawy[sorted_indices]

    if mode == 'right':
        rawx = -1.0 * np.array(rawx[::-1])
        rawy = np.array(rawy[::-1])
        rmin, rmax = np.min(rawx), -1.0 * rp
    else:
        rawx = np.array(rawx)
        rawy = np.array(rawy)
        rmin, rmax = np.min(rawx), rp

    mask = (rawx >= rmin) & (rawx <= rmax)
    newx = rawx[mask]
    newy = rawy[mask]
    newx1 = rawx[~mask]
    newy1 = rawy[~mask]

    if len(newx) < 2:
        raise ValueError('At least 2 points are required in the target region.')

    newdx = newx[-1] - newx[-2]
    newdy = newy[-1] - newy[-2]
    new_derivative = newdy / newdx

    print('new_derivative = {}'.format(new_derivative))

    if new_derivative == 0.0:
        a = 0.0
        c = newy[-1]
        newy2 = np.zeros_like(newx) + newy[-1]
    else:
        a = new_derivative / neworder * ((newx[-1] - newx[0]) ** (neworder - 1))
        c = newy[-1] - a * (newx[-1] - newx[0]) ** neworder
        newy2 = a * (newx - newx[0]) ** neworder + c

    print('newy2 params: a={}, c={}'.format(a, c))

    newx3 = np.concatenate([newx, newx1])
    newy3 = np.concatenate([newy2, newy1])

    if mode == 'right':
        newx3 = -1.0 * newx3[::-1]
        newy3 = newy3[::-1]

    return newx3, newy3


from scipy.interpolate import CubicHermiteSpline
import numpy as np

from scipy.interpolate import CubicHermiteSpline
import numpy as np

def smooth_transition(x_core, y_core, x_ped, y_ped, transition_start=0.8, transition_end=0.88):
    core_end_idx = np.argmin(np.abs(x_core - transition_start))
    ped_start_idx = np.argmin(np.abs(x_ped - transition_end))

    x0 = x_core[core_end_idx]
    y0 = y_core[core_end_idx]
    x1 = x_ped[ped_start_idx]
    y1 = y_ped[ped_start_idx]

    if core_end_idx > 0:
        m0 = (y_core[core_end_idx] - y_core[core_end_idx - 1]) / (x_core[core_end_idx] - x_core[core_end_idx - 1])
    else:
        m0 = 0.0

    if ped_start_idx < len(x_ped) - 1:
        m1 = (y_ped[ped_start_idx + 1] - y_ped[ped_start_idx]) / (x_ped[ped_start_idx + 1] - x_ped[ped_start_idx])
    else:
        m1 = 0.0

    x_transition = np.linspace(x0, x1, 50)
    hermite = CubicHermiteSpline([x0, x1], [y0, y1], [m0, m1])
    y_transition = hermite(x_transition)

    mask_core = x_core < x0
    mask_ped = x_ped > x1

    x_combined = np.concatenate([x_core[mask_core], x_transition, x_ped[mask_ped]])
    y_combined = np.concatenate([y_core[mask_core], y_transition, y_ped[mask_ped]])

    return x_combined, y_combined


# def smooth_transition(x_core, y_core, x_ped, y_ped, transition_start=0.8, transition_end=0.9):
#     core_end_idx = np.argmin(np.abs(x_core - transition_start))
#     ped_start_idx = np.argmin(np.abs(x_ped - transition_end))

#     y_core_end = y_core[core_end_idx]
#     y_ped_start = y_ped[ped_start_idx]

#     if core_end_idx > 0:
#         slope_core = (y_core[core_end_idx] - y_core[core_end_idx - 1]) / \
#                      (x_core[core_end_idx] - x_core[core_end_idx - 1])
#     else:
#         slope_core = 0.0

#     if ped_start_idx < len(x_ped) - 1:
#         slope_ped = (y_ped[ped_start_idx + 1] - y_ped[ped_start_idx]) / \
#                     (x_ped[ped_start_idx + 1] - x_ped[ped_start_idx])
#     else:
#         slope_ped = 0.0

#     print('芯部端点斜率: {:.6f}, 台基端点斜率: {:.6f}'.format(slope_core, slope_ped))

#     # 如果台基起点高于芯部终点，则整体抬升芯部，避免连接处出现反折
#     if y_ped_start > y_core_end:
#         lift_amount = y_ped_start - y_core_end + 0.02
#         y_core_adjusted = y_core + lift_amount
#         print('抬升芯部数据，抬升量: {:.6f}'.format(lift_amount))
#     else:
#         y_core_adjusted = y_core.copy()
#         print('芯部数据正常，无需抬升')

#     y_core_end_adjusted = y_core_adjusted[core_end_idx]

#     # 过渡区网格
#     n_points = 50
#     x_transition = np.linspace(transition_start, transition_end, n_points)

#     # 构造线性变化的斜率函数 k(x)
#     t = (x_transition - transition_start) / (transition_end - transition_start)
#     slopes_transition = slope_core + (slope_ped - slope_core) * t

#     # 通过积分重建过渡曲线
#     y_transition = np.zeros(n_points)
#     y_transition[0] = y_core_end_adjusted

#     for i in range(1, n_points):
#         dx = x_transition[i] - x_transition[i - 1]
#         avg_slope = 0.5 * (slopes_transition[i - 1] + slopes_transition[i])
#         y_transition[i] = y_transition[i - 1] + avg_slope * dx

#     # 端点修正：保证过渡曲线在外侧边界与台基部分精确衔接
#     error = y_transition[-1] - y_ped_start
#     if abs(error) > 1e-12:
#         correction = (y_ped_start - y_transition[-1]) * t
#         y_transition += correction
#         print('应用端点修正: {:.6e}'.format(error))

#     mask_core = x_core <= transition_start
#     mask_ped = x_ped >= transition_end

#     x_combined = np.concatenate([x_core[mask_core], x_transition, x_ped[mask_ped]])
#     y_combined = np.concatenate([y_core_adjusted[mask_core], y_transition, y_ped[mask_ped]])

#     # 调试输出：实际过渡区斜率
#     dx_transition = np.diff(x_transition)
#     dy_transition = np.diff(y_transition)
#     actual_slopes = dy_transition / dx_transition
#     print('过渡区间实际斜率变化: {:.6f} -> {:.6f}'.format(actual_slopes[0], actual_slopes[-1]))
#     print('目标斜率变化: {:.6f} -> {:.6f}'.format(slope_core, slope_ped))

#     return x_combined, y_combined


def fitting(a, b, datatype,
            te_ped_x=[], te_ped_y=[],
            h98=None, h98_threshold=0.7,
            debug=False,
            debug_dir='./debug_fit',
            debug_case_name='case',
            skip_smooth_transition=False,
            skip_left_smooth=False):
    """
    Debug version of fitting().

    Parameters added:
    - debug: whether to save debug files
    - debug_dir: output folder for npz/png/json
    - debug_case_name: prefix for debug files
    - skip_smooth_transition: isolate whether smooth_transition breaks the curve
    - skip_left_smooth: isolate whether left boundary smoothing breaks the curve
    """
    xorigin = _to_numpy(a)
    yorigin = _to_numpy(b)

    if debug:
        debug_dir = _ensure_dir(debug_dir)
        _print_summary('raw_input', xorigin, yorigin)
        _save_npz(debug_dir, '{}_00_raw_input.npz'.format(debug_case_name),
                  raw_x=xorigin, raw_y=yorigin)
        _save_plot(debug_dir, '{}_00_raw_input.png'.format(debug_case_name), [
            {'x': xorigin, 'y': yorigin, 'style': 'o', 'label': 'raw data', 'ms': 5}
        ], title='{} | raw input'.format(debug_case_name))

    x, y, q = clean(a, b, datatype)
    x = _to_numpy(x)
    y = _to_numpy(y)
    del_x, del_y = find_delete(xorigin, x, yorigin)

    x, y = x[x <= 1.0], y[x <= 1.0]

    if debug:
        _print_summary('after_clean', x, y)
        _print_summary('deleted_points', del_x, del_y)
        _save_npz(debug_dir, '{}_01_after_clean.npz'.format(debug_case_name),
                  clean_x=x, clean_y=y, del_x=del_x, del_y=del_y, q=np.array([q]))
        _save_plot(debug_dir, '{}_01_after_clean.png'.format(debug_case_name), [
            {'x': xorigin, 'y': yorigin, 'style': 'o', 'label': 'raw data', 'ms': 5, 'alpha': 0.45},
            {'x': x, 'y': y, 'style': 'o', 'label': 'clean data', 'ms': 5},
            {'x': del_x, 'y': del_y, 'style': 'x', 'label': 'removed', 'ms': 8}
        ], title='{} | after clean'.format(debug_case_name))

    plasma_mode = judge_plasma_mode(h98, threshold=h98_threshold)
    print('Plasma mode = {}, H98 = {}'.format(plasma_mode, h98))

    if debug:
        meta = {
            'datatype': datatype,
            'h98': None if h98 is None else float(h98),
            'h98_threshold': float(h98_threshold),
            'plasma_mode': plasma_mode,
            'skip_smooth_transition': bool(skip_smooth_transition),
            'skip_left_smooth': bool(skip_left_smooth),
            'q': float(q),
        }
        _save_json(debug_dir, '{}_02_meta.json'.format(debug_case_name), meta)

    # ------------------------------------------------------
    # L mode
    # ------------------------------------------------------
    if plasma_mode == 'L':
        x1, y1 = lmode_spline_fit(x, y, num_points=201, x_end=1.0)

        if debug:
            _print_summary('lmode_profile_before_left_smooth', x1, y1)
            _save_npz(debug_dir, '{}_03_lmode_profile_before_left_smooth.npz'.format(debug_case_name),
                      x1=x1, y1=y1)
            _save_plot(debug_dir, '{}_03_lmode_profile_before_left_smooth.png'.format(debug_case_name), [
                {'x': x, 'y': y, 'style': 'o', 'label': 'clean data', 'ms': 5},
                {'x': x1, 'y': y1, 'style': '-', 'label': 'lmode spline', 'lw': 2}
            ], title='{} | lmode before left smooth'.format(debug_case_name))

        if datatype == 'neTS':
            cross_point = 0.30
        else:
            cross_point = x[0] + 0.05

        if not skip_left_smooth:
            x_left, y_left = new_sm_extremity(
                x1, y1,
                rmin=0.0,
                rmax=cross_point,
                mode='left',
                sf=1.0,
                neworder=2
            )

            right_mask = x1 >= cross_point
            x1 = np.concatenate((x_left, x1[right_mask]), axis=0)
            y1 = np.concatenate((y_left, y1[right_mask]), axis=0)

        if debug:
            _print_summary('lmode_final', x1, y1)
            _save_npz(debug_dir, '{}_04_lmode_final.npz'.format(debug_case_name),
                      x1=x1, y1=y1, cross_point=np.array([cross_point]))
            _save_plot(debug_dir, '{}_04_lmode_final.png'.format(debug_case_name), [
                {'x': x, 'y': y, 'style': 'o', 'label': 'clean data', 'ms': 5},
                {'x': x1, 'y': y1, 'style': '-', 'label': 'final fit', 'lw': 2}
            ], title='{} | lmode final'.format(debug_case_name))

        return x1, y1, del_x, del_y, q

    # ------------------------------------------------------
    # Ti branch
    # ------------------------------------------------------
    if datatype == 'Ti':
        ti_pedestal_data = x[x >= 0.9]
        if len(ti_pedestal_data) == 0 and len(te_ped_x) > 0 and len(te_ped_y) > 0:
            print('Ti台基部分无数据，使用TE台基数据')
            x_ped = np.linspace(x[-1], te_ped_x[0], 100)
            y_ped = np.linspace(y[-1], te_ped_y[0], 100)
            x = np.concatenate((x, x_ped, te_ped_x), axis=0)
            y = np.concatenate((y, y_ped, te_ped_y), axis=0)

            x_part, y_part = front_ped_fitting(x, y)
            x_ped = te_ped_x
            y_ped = te_ped_y

            if skip_smooth_transition:
                x1 = np.concatenate((x_part, x_ped[x_ped >= 0.9]), axis=0)
                y1 = np.concatenate((y_part, y_ped[x_ped >= 0.9]), axis=0)
            else:
                x1, y1 = smooth_transition(x_part, y_part, x_ped, y_ped,
                                           transition_start=0.8, transition_end=0.92)
        else:
            if x[-1] < 0.6:
                x_09 = np.linspace(x[-1], te_ped_x[0], 100)
                y_09 = np.linspace(y[-1], te_ped_y[0], 100)
                x = np.concatenate((x, x_09, te_ped_x), axis=0)
                y = np.concatenate((y, y_09, te_ped_y), axis=0)

            x_part, y_part, peak_value1, peak_value2, R2, N2, min_value = front_ped_fitting_Te(x, y)
            Rfit, nfit = ped_fitting_TE(0.9, peak_value2, min_value,
                                        ped_pos=0.93, w=0.02, slope1=0.02, slope2=-0.02)
            x1 = np.concatenate((x_part, Rfit[(Rfit <= 1.005) & (Rfit >= 0.86)]), axis=0)
            y1 = np.concatenate((y_part, nfit[(Rfit <= 1.005) & (Rfit >= 0.86)]), axis=0)

    # ------------------------------------------------------
    # Te branch
    # ------------------------------------------------------
    elif datatype == 'Te':
        x_part, y_part, peak_value1, peak_value2, R2, N2, min_value = front_ped_fitting_Te(x, y)

        if debug:
            _print_summary('te_stage_core_spline', x_part, y_part)
            _save_npz(debug_dir, '{}_03_te_core_stage.npz'.format(debug_case_name),
                      x=x, y=y,
                      x_part=x_part, y_part=y_part,
                      R2=R2, N2=N2,
                      peak_value1=np.array([peak_value1]),
                      peak_value2=np.array([peak_value2]),
                      min_value=np.array([min_value]))
            _save_plot(debug_dir, '{}_03_te_core_stage.png'.format(debug_case_name), [
                {'x': x, 'y': y, 'style': 'o', 'label': 'clean data', 'ms': 5},
                {'x': R2, 'y': N2, 'style': 'o', 'label': '0.8<rho<=1.0 data', 'ms': 5, 'alpha': 0.7},
                {'x': x_part, 'y': y_part, 'style': '-', 'label': 'core spline', 'lw': 2}
            ], title='{} | Te core spline stage'.format(debug_case_name))
            print('[DEBUG] peak_value1(0.85) = {}'.format(peak_value1))
            print('[DEBUG] peak_value2(0.90 target) = {}'.format(peak_value2))
            print('[DEBUG] min_value(edge) = {}'.format(min_value))

        peak_value2_opt, mtanhR_opt = fit_te_peak_and_start_ls(
            x_part, y_part, R2, N2, min_value,
            peak_init=peak_value2,
            mtanhR_init=0.84,
            ped_pos=0.93, w=0.05, slope1=-0.01, slope2=0,
            peak_cap=peak_value1
        )

        if debug:
            print('[DEBUG] peak_value2_opt = {}'.format(peak_value2_opt))
            print('[DEBUG] mtanhR_opt = {}'.format(mtanhR_opt))

        Rfit, nfit = ped_fitting_TE(mtanhR_opt, peak_value2_opt, min_value,
                                    ped_pos=0.93, w=0.05, slope1=-0.01, slope2=0)

        if debug:
            _print_summary('te_stage_ped_mtanh_seed', Rfit, nfit)
            _save_npz(debug_dir, '{}_04_te_ped_stage.npz'.format(debug_case_name),
                      Rfit=Rfit, nfit=nfit)
            _save_plot(debug_dir, '{}_04_te_ped_stage.png'.format(debug_case_name), [
                {'x': x, 'y': y, 'style': 'o', 'label': 'clean data', 'ms': 5},
                {'x': x_part, 'y': y_part, 'style': '-', 'label': 'core spline', 'lw': 2},
                {'x': Rfit, 'y': nfit, 'style': '--', 'label': 'ped mtanh', 'lw': 2}
            ], title='{} | Te pedestal seed'.format(debug_case_name))

        if skip_smooth_transition:
            x1 = np.concatenate((x_part[x_part <= 0.8], Rfit[Rfit >= 0.9]), axis=0)
            y1 = np.concatenate((y_part[x_part <= 0.8], nfit[Rfit >= 0.9]), axis=0)
        else:
            x1, y1 = smooth_transition(x_part, y_part, Rfit, nfit,
                                       transition_start=0.85, transition_end=0.9)

        if debug:
            _print_summary('te_stage_after_transition_before_shift', x1, y1)
            _save_npz(debug_dir, '{}_05_te_after_transition_before_shift.npz'.format(debug_case_name),
                      x1=x1, y1=y1)
            _save_plot(debug_dir, '{}_05_te_after_transition_before_shift.png'.format(debug_case_name), [
                {'x': x, 'y': y, 'style': 'o', 'label': 'clean data', 'ms': 5},
                {'x': x1, 'y': y1, 'style': '-', 'label': 'after transition', 'lw': 2}
            ], title='{} | Te after transition before shift'.format(debug_case_name))

        if np.any(x1 == 1.0):
            index_mini = (x1 == 1.0)
            mini = y1[index_mini]
            temp = np.abs(mini)
            y1 = y1 + temp + 0.05
        else:
            closest_idx = np.argmin(np.abs(x1 - 1.0))
            mini = y1[closest_idx]
            temp = np.abs(mini)
            y1 = y1 + temp + 0.05

        if debug:
            _print_summary('te_stage_after_shift_before_left_smooth', x1, y1)
            _save_npz(debug_dir, '{}_06_te_after_shift_before_left_smooth.npz'.format(debug_case_name),
                      x1=x1, y1=y1)
            _save_plot(debug_dir, '{}_06_te_after_shift_before_left_smooth.png'.format(debug_case_name), [
                {'x': x, 'y': y, 'style': 'o', 'label': 'clean data', 'ms': 5},
                {'x': x1, 'y': y1, 'style': '-', 'label': 'after shift', 'lw': 2}
            ], title='{} | Te after shift before left smooth'.format(debug_case_name))

    # ------------------------------------------------------
    # other branches
    # ------------------------------------------------------
    else:
        if datatype == 'ne' or datatype == 'Refl':
            x_part, y_part = front_ped_fitting(x, y)
        else:
            x_part, y_part = front_ped_fitting(x, y)

        x_sol = np.linspace(1.01, 1.20, 20)
        if datatype == 'Refl':
            y_sol = np.linspace(y[-1] * 0.999, y[-1] * 0.9, 20)
        else:
            y_sol = np.linspace(np.min(b / 10 ** q) * 0.999, np.min(b / 10 ** q) * 0.8, 20)

        x = np.concatenate((x, x_sol), axis=0)
        y = np.concatenate((y, y_sol), axis=0)

        if datatype == 'Te':
            rconnect = 0.86
        elif datatype == 'ne' or datatype == 'Refl':
            rconnect = 0.805
        else:
            rconnect = 0.86

        index = np.where((x <= 1.2) & (x > rconnect))
        Rfit, nfit, ne_ped, ne_width, zzall = ped_fitting(x, y, index)
        x1 = np.concatenate((x_part, Rfit[(Rfit <= 1.005) & (Rfit >= rconnect)]), axis=0)
        y1 = np.concatenate((y_part, nfit[(Rfit <= 1.005) & (Rfit >= rconnect)]), axis=0)

        from re_fitting import fitting_mod
        xconnect = rconnect

        for i in range(10):
            x1, y1, xconnect = fitting_mod(x1, y1, xconnect)

        from scipy.signal import savgol_filter
        y1 = savgol_filter(y1, window_length=11, polyorder=3)

    # left smoothing
    if datatype == 'neTS':
        cross_point = 0.30
    else:
        cross_point = x[0] + 0.05

    if debug:
        _save_npz(debug_dir, '{}_07_before_left_smooth.npz'.format(debug_case_name),
                  x1_before_left=x1, y1_before_left=y1, cross_point=np.array([cross_point]))
        _save_plot(debug_dir, '{}_07_before_left_smooth.png'.format(debug_case_name), [
            {'x': x, 'y': y, 'style': 'o', 'label': 'clean data', 'ms': 5},
            {'x': x1, 'y': y1, 'style': '-', 'label': 'before left smooth', 'lw': 2}
        ], title='{} | before left smooth'.format(debug_case_name))

    if not skip_left_smooth:
        x_left, y_left = new_sm_extremity(x1, y1, rmin=0.0, rmax=cross_point,
                                          mode='left', sf=1.0, neworder=2)
        mask_right = x1 >= cross_point
        x1 = np.concatenate((x_left, x1[mask_right]), axis=0)
        y1 = np.concatenate((y_left, y1[mask_right]), axis=0)

    if debug:
        _print_summary('final_output', x1, y1)
        _save_npz(debug_dir, '{}_08_final_output.npz'.format(debug_case_name),
                  final_x=x1, final_y=y1, del_x=del_x, del_y=del_y, q=np.array([q]))
        _save_plot(debug_dir, '{}_08_final_output.png'.format(debug_case_name), [
            {'x': xorigin, 'y': yorigin, 'style': 'o', 'label': 'raw data', 'ms': 5, 'alpha': 0.35},
            {'x': x, 'y': y, 'style': 'o', 'label': 'clean data', 'ms': 5},
            {'x': del_x, 'y': del_y, 'style': 'x', 'label': 'removed', 'ms': 8},
            {'x': x1, 'y': y1, 'style': '-', 'label': 'final fit', 'lw': 2}
        ], title='{} | final output'.format(debug_case_name))

    return x1, y1, del_x, del_y, q
