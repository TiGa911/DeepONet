import numpy as np
from scipy.interpolate import interp1d
from fitting_mtanh import fitting
from matplotlib import pyplot as plt
from readMDS import readmds
import matplotlib.gridspec as gridspec
import numpy as np

def deriv(x, y):
    """二阶拉格朗日插值求导"""
    x = np.array(x)
    y = np.array(y)
    def dlip(ra,r,f):
        r1,r2,r3 = r
        f1,f2,f3 = f
        return ((ra-r1)+(ra-r2))/(r3-r1)/(r3-r2)*f3 \
             + ((ra-r1)+(ra-r3))/(r2-r1)/(r2-r3)*f2 \
             + ((ra-r2)+(ra-r3))/(r1-r2)/(r1-r3)*f1
    return np.array([dlip(x[0],x[0:3],y[0:3])]
                   + list(dlip(x[1:-1],[x[0:-2],x[1:-1],x[2:]],
                                          [y[0:-2],y[1:-1],y[2:]]))
                   + [dlip(x[-1],x[-3:],y[-3:])])

def calcz(x, y):
    """计算逆尺度长度 Z = - (dy/dx) / y"""
    tmp = np.nanmin(np.abs(y[np.where(y != 0)]))
    return -deriv(x, y) / (np.abs(y) + tmp*1E-32) * np.sign(y)

def integz(x0, z0, xbc, ybc, x, clipz=True):
    """由 Z 积分恢复剖面 y"""
    x_ = x
    x = np.unique(np.hstack((x_, xbc)))
    z = interp1d(x0, z0, fill_value="extrapolate")(x)

    if clipz:
        inside = np.where((x >= min(x0)) & (x <= max(x0)))[0].astype(int)
        low = np.where(x < min(x0))[0].astype(int)
        high = np.where(x > max(x0))[0].astype(int)
        z[low] = z[inside][0]
        z[high] = z[inside][-1]

    # backward integration
    i0 = np.where(x <= xbc)[0]
    x0b = x[i0]
    z0b = z[i0]
    y0 = []
    if len(x0b):
        t0 = -(z0b[:-1] + z0b[1:]) * np.diff(x0b) / 2.
        y0 = np.cumprod([ybc] + np.exp(-t0[::-1]).tolist())[::-1]

    # forward integration
    i1 = list(range(len(x)))[max(np.hstack((0, i0))):]
    x1 = x[i1]
    z1 = z[i1]
    y1 = []
    if len(x1):
        t1 = (z1[:-1] + z1[1:]) * np.diff(x1) / 2.
        y1 = np.cumprod([ybc] + np.exp(-t1).tolist())[1:]

    return interp1d(x, np.hstack((y0, y1)), fill_value="extrapolate")(x_)

import numpy as np
from scipy.interpolate import interp1d

def nml_smooth(x, y, x_core_end, x_ped_start):
    """
    仅在芯部结束 (x_core_end) 与台基开始 (x_ped_start) 之间平滑剖面，
    芯部与台基区域保持原始值不变。

    输入:
        x           : rho 数组 (单调递增)
        y           : 剖面数值数组
        x_core_end  : 芯部结束位置
        x_ped_start : 台基开始位置

    输出:
        new_x, new_y : 平滑后的剖面
    """
    # 转换为 float 数组
    x = np.asarray(x, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()

    if x_core_end >= x_ped_start:
        raise ValueError("必须满足 x_core_end < x_ped_start")

    # 找出各区域索引
    mask_core = x <= x_core_end
    mask_ped = x >= x_ped_start
    mask_mid = (x > x_core_end) & (x < x_ped_start)

    # 保留 core 和 ped 原值
    new_y = y.copy()

    if mask_mid.any():
        # 取出中间区域
        x_mid = x[mask_mid]
        y_mid = y[mask_mid]

        # 计算逆尺度长度 Z
        def deriv(xa, ya):
            xa, ya = np.asarray(xa), np.asarray(ya)
            def dlip(ra,r,f):
                r1,r2,r3 = r
                f1,f2,f3 = f
                return ((ra-r1)+(ra-r2))/(r3-r1)/(r3-r2)*f3 \
                     + ((ra-r1)+(ra-r3))/(r2-r1)/(r2-r3)*f2 \
                     + ((ra-r2)+(ra-r3))/(r1-r2)/(r1-r3)*f1
            return np.array([dlip(xa[0],xa[0:3],ya[0:3])]
                           + list(dlip(xa[1:-1],
                                       [xa[0:-2],xa[1:-1],xa[2:]],
                                       [ya[0:-2],ya[1:-1],ya[2:]]))
                           + [dlip(xa[-1],xa[-3:],ya[-3:])])

        def calcz(xa, ya):
            tmp = np.nanmin(np.abs(ya[np.where(ya != 0)]))
            return -deriv(xa, ya) / (np.abs(ya) + tmp*1E-32) * np.sign(ya)

        def integz(x0, z0, xbc, ybc, x_eval):
            x_ = x_eval
            x = np.unique(np.hstack((x_, xbc)))
            z = interp1d(x0, z0, fill_value="extrapolate")(x)

            inside = np.where((x >= min(x0)) & (x <= max(x0)))[0]
            low = np.where(x < min(x0))[0]
            high = np.where(x > max(x0))[0]
            if inside.size > 0:
                z[low] = z[inside][0]
                z[high] = z[inside][-1]

            # backward
            i0 = np.where(x <= xbc)[0]
            x0b, z0b = x[i0], z[i0]
            y0 = []
            if len(x0b):
                t0 = -(z0b[:-1] + z0b[1:]) * np.diff(x0b) / 2.
                y0 = np.cumprod([ybc] + np.exp(-t0[::-1]).tolist())[::-1]

            # forward
            i1 = list(range(len(x)))[max(np.hstack((0, i0))):]
            x1, z1 = x[i1], z[i1]
            y1 = []
            if len(x1):
                t1 = (z1[:-1] + z1[1:]) * np.diff(x1) / 2.
                y1 = np.cumprod([ybc] + np.exp(-t1).tolist())[1:]

            return interp1d(x, np.hstack((y0, y1)),
                            fill_value="extrapolate")(x_)

        # 平滑处理
        z_mid = calcz(x_mid, y_mid)
        # 边界条件：从 core_end 点接入
        pivot = x_mid[0]
        ybc = y[mask_core][-1]  # core_end 对应的 y
        new_y[mask_mid] = integz(x_mid, z_mid, pivot, ybc, x_mid)

    return x, new_y


# shot = int(input('shot:'))#73999  106915  ### 好：131420 63948 151518
# # for i in range(50):
# #     try:
# time = float(input('time:'))#20.5
# # shot = shot + 1
# # time = float(3.5)
# data, status,real_time = readmds(shot, time)
# x, y, datatype = data['Te']['TS']['Rho'], data['Te']['TS']['data'], data['Te']['TS']['type']
# x, y, x_del, y_del,q = fitting(x, y, datatype)

# x,y = nml_smooth(x,y,x_core_end=0.85, x_ped_start=0.9)

# plt.scatter(x, y/1000., marker='.', c='r')
# plt.xlabel(r'$\rho$', fontsize=10)
# plt.ylabel('Te(TS)(KeV)', fontsize=10) # axis1 = plt.subplot(121, xlim=(1,2), ylim=(2,3))
#     # try:   
# x, y, x_del, y_del,q = fitting(x, y, datatype)
# # print(x.shape)
# # print(y.shape)
# # print(y)
# plt.plot(x, y)   
# #删除的点        
# plt.scatter(x_del, y_del/1000.,marker='x', c='g')  
# plt.xlim([0, 1.2])
# plt.ylim([min(y), max(y)+0.1])   
# plt.show()