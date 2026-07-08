import numpy as np
from scipy.interpolate import UnivariateSpline


def _robust_sigma_mad(arr, eps=1e-12):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return eps
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    return max(1.4826 * mad, eps)


def _sort_and_merge_duplicate_x(x, y, yerr=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if yerr is not None:
        yerr = np.asarray(yerr, dtype=float)

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    if yerr is not None:
        yerr = yerr[idx]

    xu, yu, eu = [], [], []
    i = 0
    n = len(x)
    while i < n:
        j = i + 1
        while j < n and x[j] == x[i]:
            j += 1
        xu.append(x[i])
        yu.append(np.mean(y[i:j]))
        if yerr is not None:
            eu.append(np.mean(yerr[i:j]))
        i = j

    xu = np.asarray(xu, dtype=float)
    yu = np.asarray(yu, dtype=float)
    if yerr is None:
        return xu, yu, None
    return xu, yu, np.asarray(eu, dtype=float)


def _enforce_monotonic_decreasing(y_ref):
    y_ref = np.asarray(y_ref, dtype=float).copy()
    for i in range(1, len(y_ref)):
        if y_ref[i] > y_ref[i - 1]:
            y_ref[i] = y_ref[i - 1]
    return y_ref


def _robust_spline_baseline(x, y, s_factor=0.20, n_iter=6):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 4:
        return np.interp(x, x, y)

    k = min(3, len(x) - 1)
    s = max(s_factor * len(x), 1e-8)
    w = np.ones_like(y)

    spline = None
    for _ in range(n_iter):
        spline = UnivariateSpline(x, y, w=w, s=s, k=k)
        y_fit = spline(x)
        residual = y - y_fit
        sigma = _robust_sigma_mad(residual)

        c = 2.5 * sigma + 1e-12
        u = np.abs(residual) / c
        w = 1.0 / np.maximum(1.0, u ** 2)

    return spline(x)


def clean_te_profile(x, y, yerr=None,
                     window=5,
                     k_low=3.0,
                     k_high=4.5,
                     low_ratio=0.65,
                     high_ratio=1.60,
                     max_iter=15,
                     spline_s_factor=0.20,
                     scale_floor_rel=0.03,
                     scale_floor_abs=0.03,
                     use_monotonic_prior=False,
                     debug=False):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if yerr is not None:
        yerr = np.asarray(yerr, dtype=float)
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr) & (y > 0) & (yerr >= 0)
    else:
        valid = np.isfinite(x) & np.isfinite(y) & (y > 0)

    x = x[valid]
    y = y[valid]
    if yerr is not None:
        yerr = yerr[valid]

    if len(y) == 0:
        if debug:
            return x, y, 0, []
        return x, y, 0

    x, y, yerr = _sort_and_merge_duplicate_x(x, y, yerr)

    # Te 固定按 eV -> keV 缩放，避免被单个极端异常值把量级带坏
    q = 3
    scale_q = 1000.0
    ys = y / scale_q
    if yerr is not None:
        es = yerr / scale_q
    else:
        es = None

    removed_log = []

    if len(ys) < max(window, 5):
        if debug:
            return x, ys, q, removed_log
        return x, ys, q

    for it in range(max_iter):
        if len(ys) < max(window, 5):
            break

        y_ref = _robust_spline_baseline(x, ys, s_factor=spline_s_factor, n_iter=6)

        # 可选物理先验：参考曲线整体单调下降
        if use_monotonic_prior:
            y_ref = _enforce_monotonic_decreasing(y_ref)

        residual = ys - y_ref
        global_sigma = _robust_sigma_mad(residual)
        scale_floor = max(scale_floor_rel * np.median(ys), scale_floor_abs, 1e-8)

        flagged = []
        half = max(window // 2, 1)

        # 端点默认不删，降低误删中心点/边界点的风险
        for i in range(1, len(ys) - 1):
            left = max(0, i - half)
            right = min(len(ys), i + half + 1)

            loc_res = np.concatenate([residual[left:i], residual[i + 1:right]])
            if loc_res.size >= 2:
                local_sigma = _robust_sigma_mad(loc_res)
            else:
                local_sigma = global_sigma

            if es is not None:
                scale_i = max(local_sigma, es[i], scale_floor)
            else:
                scale_i = max(local_sigma, scale_floor)

            score = residual[i] / scale_i

            x0, x1, x2 = x[i - 1], x[i], x[i + 1]
            y0, y2 = ys[i - 1], ys[i + 1]
            if abs(x2 - x0) < 1e-12:
                y_lin = 0.5 * (y0 + y2)
            else:
                y_lin = y0 + (y2 - y0) * (x1 - x0) / (x2 - x0)

            low_cond = (score < -k_low) and (ys[i] < low_ratio * y_lin)
            high_cond = (score > k_high) and (ys[i] > high_ratio * y_lin)

            if low_cond or high_cond:
                flagged.append({
                    'idx': i,
                    'x': x[i],
                    'y': ys[i],
                    'y_ref': y_ref[i],
                    'y_lin': y_lin,
                    'score': score,
                    'type': 'low' if low_cond else 'high'
                })

        if not flagged:
            break

        flagged = sorted(flagged, key=lambda d: abs(d['score']), reverse=True)
        worst = flagged[0]
        rm_idx = worst['idx']
        removed_log.append({
            'iter': it,
            'idx': rm_idx,
            'x': float(worst['x']),
            'y_scaled': float(worst['y']),
            'y_ref': float(worst['y_ref']),
            'y_lin': float(worst['y_lin']),
            'score': float(worst['score']),
            'type': worst['type']
        })

        x = np.delete(x, rm_idx)
        ys = np.delete(ys, rm_idx)
        if es is not None:
            es = np.delete(es, rm_idx)

    if debug:
        return x, ys, q, removed_log
    return x, ys, q


def polynomial(X, Y):
    return clean_te_profile(
        X, Y,
        yerr=None,
        window=5,
        k_low=1.3,
        k_high=4.4,
        low_ratio=0.7,
        high_ratio=2.2,
        max_iter=8,
        spline_s_factor=0.25,
        scale_floor_rel=0.05,
        scale_floor_abs=0.05,
        use_monotonic_prior=False,   # 物理先验先默认关闭；效果好再打开
        debug=False
    )
