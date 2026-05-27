# -*- coding: utf-8 -*-
# This module should make the left boundary derivative of the fitted data tend to 0.

import numpy as np

# M = m1*sf+(m2+m3)/2*(1-sf)
def new_sm_extremity(rawx, rawy, rmin, rmax, mode, sf, neworder):
    mode = 'left'
    newx = rawx[(rawx >= rmin) & (rawx <= rmax)]
    newy = rawy[(rawx >= rmin) & (rawx <= rmax)]

    new_derivative1 = (newy[-1] - newy[-2]) / (newx[-1] - newx[-2])
    nds = np.sign(new_derivative1)
    newdx = newx[-1] - newx[0]
    newdy = newy[-1] - newy[0]
    new_derivative2 = (newy[-1] - newy[-2]) / (newx[-1] - newx[-2])
    new_derivative = 1.0 * sf * min(abs(new_derivative1), abs(new_derivative1)) * nds
    if (new_derivative == 0.0):
        newy2 = newx * 0.0 + newy[-1]
    else:
        a = new_derivative / (1.0 * neworder * ((newx[-1] - newx[0]) ** (neworder - 1)))
        a = a * sf
        c = newy[-1] - 1.0 * a * (newx[-1] - newx[0]) ** (neworder)
        newy2 = a * (newx - newx[0]) ** (neworder) + c
    return (newx, newy2)

# rawx = ne_rho
# rawy = ne_data
#
# x, y = new_sm_extremity(rawx, rawy, rmin=0.0, rmax=0.20, mode='left', sf=1.0, neworder=2)
# plt.plot(x, y)
# plt.plot(rawx, rawy)
# plt.show()