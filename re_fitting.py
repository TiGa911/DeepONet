# -*- coding: utf-8 -*-
# This module uses the method of fitting module to iterate the data smoothly,
# so that the data fitting curve is smooth.

import numpy as np
from numpy import exp
from scipy.optimize import leastsq
from matplotlib import pyplot as plt


def mtanh(x, factor):
    # mtanh fuction
    A=factor[0]
    B=factor[1]
    alpha=factor[2]
    x_sys=factor[3]
    w=factor[4]
    beta=factor[5]
    #gamma=a[7]
    z=(x_sys-x)/w
    z = np.clip(z, -500, 500)
    mtanh_fun=((1+alpha*z)*exp(z)-(1+beta*z)*exp(-z))/(exp(z)+exp(-z))
    y=A*mtanh_fun+B
    return y

def residuals(p, y, x):
    # Error function when using the least square method next.
    return y - mtanh(x, p)

def ped_fitting(R,n):
    # About the fitting of mtanh function using least square method in the platform base area.
    ped_pos=0.91#pedestal position
    h=3.59  #height
    w=0.05    #width
    slope1=2   #slope of the core part
    slope2=-3  #slope of edge part
    fac_init=np.array((h/2, h/2, slope1, ped_pos, w, slope2))

    index=np.where((R<=1.2)&(R>0.75))
    R1=R[index]
    n1=n[index]

    factor=leastsq(residuals, fac_init, args=(n1, R1), maxfev=5000)

    ne_ped=factor[0][0]+factor[0][1]
    ne_width =2* factor[0][4]
    
    x1=np.arange(np.min(R1),1.2,0.01)
    y1=mtanh(x1,factor[0])
    return np.array((x1,y1,ne_ped,ne_width,factor), dtype=object)


def fitting_mod(x, y, xconnect):
    # The pedestal fitting function used in iterative fitting is similar to that of the fitting module.
    # print(y[x<=1.0][-1])
    x_sol=np.linspace(1.01,1.20,20)
    y_sol = np.linspace(np.min(y)*0.999, np.min(y)*0.9, 20)
    #y_sol=np.linspace(y[-1]*0.999,y[-1]*0.98,20)
    # y_sol=np.linspace(0.04,0.02,20)

    x=np.concatenate((x,x_sol),axis=0)
    y=np.concatenate((y,y_sol),axis=0)
    #print(y)

    Rfit,nfit,ne_ped,ne_width,zzall = ped_fitting(x,y)
    # print(zzall)
    # plt.plot(x,y,'--',label='ne',linewidth=4.0,c='black')
    # plt.plot(Rfit,nfit,'-',label='mtanh',linewidth=4.0,c='red')

    rconect=xconnect
    xnew=np.concatenate((x[(x<=rconect)&(x>=0.)],Rfit[(Rfit<=1.005)&(Rfit>=rconect)]),axis=0)
    ynew=np.concatenate((y[(x<=rconect)&(x>=0.)],nfit[(Rfit<=1.005)&(Rfit>=rconect)]),axis=0)

    #print(Rfit);print(xnew);

    # plt.plot(xnew,ynew,'-',label='mtanh',linewidth=4.0,c='blue')

    # plt.xlabel(r'$\rho$',fontsize=16)
    # plt.ylabel('ne (10$^{19}$ m$^{-3}$)',fontsize=16)
    # plt.show()
    a = np.intersect1d(xnew, Rfit)[0]+0.001
    # print(a)
    return xnew, ynew, a
