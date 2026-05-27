# -*- coding: utf-8 -*-
# Data cleaning module
# First, clear the quartile distance (IQR function) of the data,then judge the
#        error between the data points and the curve obtained by spline smoothing
#        of the data and make the standard deviation judgment (iter_clean function),
#        so as to clean the data iteratively;
# Last, the del_lowdata function is used to take larger data and keep it.

import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from polynomial import polynomial


def iter_clean(m, n, datatype):
    '''
    Smooth iterative cleaning of data.
    para m: Row data
    para n: Ordinate data
    para datatype: Label for identifying data
    return modified m, modified n, Order of magnitude of n
    '''
    def delete_point1(a, b):
        # roughly data cleaning of the front_pedestal
        y = UnivariateSpline(a, b, s=8)
        x = np.linspace(0, 1, 10000)
        e = []
        for i in range(len(a[a < 0.785])):
            e.append(abs(y(a[i]) - b[i]))
        index = []
        # print(np.std(e))
        if np.std(e) > 0.08:
            index1 = e.index(max(e))
            index.append(index1)

        a = np.delete(a, index)
        b = np.delete(b, index)
        return a, b, x, y
    def delete_point2(a, b):
        # roughly data cleaning of the pedestal
        y = UnivariateSpline(a[a >= 0.6], b[a >= 0.6], s=8)
        x = np.linspace(0.6, 1, 10000)
        e = []
        for i in range(len(a[a >= 0.6])):
            e.append(abs(y(a[a >= 0.6][i]) - b[a >= 0.6][i]))
        # print(e)
        # print(np.std(e))
        index = []
        std = 0
        if datatype == 'neTS':
            std = 0.1
        elif datatype == 'Ti':
            std = 0.1
        if np.std(e) > std:   # Te的点0.08正好， ne的点0.1正好
            index1 = e.index(max(e))
            index1 = a.tolist().index(a[a >= 0.6][index1])
            index.append(index1)
        a = np.delete(a, index)
        b = np.delete(b, index)
        # print(len(a), len(b))
        return a, b, x, y

    mm = m
    nn = np.nan_to_num(n)
    indexx = []
    for a in range(len(nn)):
        if nn[a] == 0:
            indexx.append(a)
    nn = np.delete(nn, indexx)
    q = len(str(int(nn[0])))-1
    n = n/10**q
    for i in range(12):
        m, n, x, y = delete_point1(m, n)
    # plt.plot(m, n, '.')
    # plt.plot(x, y(x))
    # plt.show()
    if datatype == 'neTS':
        for j in range(12):
            m, n, x, y = delete_point2(m, n)
        # plt.plot(m, n, '.')
        # plt.plot(x, y(x))
        # plt.show()
        # if datatype == 'neTS':
        #     for j in range(12):
        #         m, n, x, y = delete_point3(m, n)
                # plt.plot(m, n, '.')
                # plt.plot(x, y(x))
                # plt.show()

    return m, n, q

def IQR(a, b):
    '''
    IQR is the quartile distance data cleaning function, and the following step-by-step cleaning is performed.
    (little effect)
    para a: Row data
    para b: Ordinate data
    return modified m, modified n
    '''
    # Data is sorted according to the size of abscissa.
    def bubble_sort(a, b):
        for i in range(1, len(a) - 1):
            for j in range(0, len(a) - i):
                if a[j] > a[j + 1]:
                    a[j], a[j + 1] = a[j + 1], a[j]
                    b[j], b[j + 1] = b[j + 1], b[j]

        return a, b

    def detect_outliers2(df):
        # main fuction  of IQR
        de = []
        if type(df).__name__ == 'ndarray':
            df = df.tolist()
        # outlier_indices = []
        # 1st quartile (25%)
        if len(df) != 0:
            Q1 = np.percentile(df, 25)
            # 3rd quartile (75%)
            Q3 = np.percentile(df, 75)
            # Interquartile range (IQR)
            IQR = Q3 - Q1

            # outlier step
            outlier_step = 1.5 * IQR
            for nu in df[:]:
                if (nu < Q1 - outlier_step) | (nu > Q3 + outlier_step):
                    de.append(nu)
                    df.remove(nu)
        if type(df) == 'list' or 'tuple':
            df = np.array(df)
        return de, df

    m, n = bubble_sort(a, b)
    de = []
    m_max = np.max(m)
    fenduan = [0, m_max * 0.3, m_max * 0.6, m_max * 0.8, m_max]
    x_index = {}
    bdict = {}
    s = len(fenduan) - 1
    for h in range(s):
        if fenduan[1] == 1 or fenduan[1] == m_max:
            sss = np.where((fenduan[0] <= m) & (m <= fenduan[1]))
        else:
            sss = np.where((fenduan[0] <= m) & (m < fenduan[1]))
        x_index['x_index' + str(h)] = sss
        fenduan.pop(0)
    for i in range(s):
        bdict['b' + str(i)] = b[x_index["x_index" + str(i)]]
    for j in range(s):
        m1 = []
        de1, bdict['b' + str(j)] = detect_outliers2(bdict['b' + str(j)])
        if de1 != []:
            de = np.concatenate((de, de1))
            # print(de)
        if type(b).__name__ == 'ndarray':
            b = b.tolist()
        if len(de) != 0:
            # print(de)
            for k in de:
                index1 = b.index(k)
                m1.append(index1)
                # print(m1)
            a = np.delete(a, m1)
            b = np.delete(b, m1)
            de = []
            # print(a)
            # print(len(a), len(b))
    y = bdict['b0']
    for k in range(s):
        if k > 0:
            y = np.concatenate((y, bdict['b' + str(k)]), axis=0)
    return a, y



def del_lowdata(a, b, a_range):
    # reserve larger data
    for j in range(10):
        bpart = b[(a_range[0] <= a) & (a <= a_range[1])]
        index = []
        for i in range(1, len(bpart)-1):
            if i+1 <= len(bpart):
                if (bpart[i] < bpart[i-1]) & (bpart[i] < bpart[i+1]):
                    index.append(b.tolist().index(bpart[i]))
        b = np.delete(b, index)
        a = np.delete(a, index)
    index = []
    for i in range(len(b)-1):
        if b[i] < b[i+1]:
            index.append(i)
    b = np.delete(b, index)
    a = np.delete(a, index)
    return a, b


def clean(a, b, datatype):
    # The implementation function of this module.
    # print(datatype)
    # print(datatype)
    q=0
    if datatype != 'Refl':
        # q is the Order of magnitude of n
        if datatype == 'Ti':
            return a, b, q
        elif datatype == 'Te':
            a, b, q= polynomial(a,b)
            print(q)
        else:
            # a, b, q= polynomial(a,b)
            a,b = IQR(a,b)
            a, b, q= iter_clean(a, b, datatype)
            a, b = del_lowdata(a, b, [0.8, 1.0])
    if datatype == 'Refl':
        a, b = IQR(a, b)
        index = []
        refllen = len(b)
        for i in range(refllen):
            if np.isnan(b[i]) == True:
                index.append(i)
            if b[i] == 0:
                index.append(i)
        a = np.delete(a, index)
        b = np.delete(b, index)
    return a, b, q


# shot = 63948
# time = 6.0
# data = classify(shot, time)
# a, b, datatype = data['Te']['TS']['Rho'], data['Te']['TS']['data'], data['Te']['type']
# # a, b, datatype = data['ne']['Refl']['Rho'], data['ne']['Refl']['data'], data['ne']['type']
# # print(data.keys())
# # a, b = bubble_sort(te_rho, te_ts)
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.plot(a, b/1000, '.', linewidth=3)
# # plt.xlim(0, 1)
# # plt.ylim(0, max(b)+0.1)
# # plt.ylabel('Te kev', fontsize=16)
#
# a, b = clean(a, b, datatype)
# plt.subplot(1, 2, 2)
# plt.plot(a, b, '.', linewidth=3, c='red')
# # plt.xlim(0, 1)
# # plt.ylim(0, max(b)+0.1)
# # plt.ylabel('Te (10$^{'+str(q)+'}$ m$^{-3}$)', fontsize=16)
# # plt.ylabel('Te kev', fontsize=16)
# plt.show()

