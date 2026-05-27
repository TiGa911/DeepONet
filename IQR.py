import numpy as np
import matplotlib.pyplot as plt
from readMDS import readmds

shot = int(input('shot:'))#73999  106915

time = float(input('time:'))#20.5
data, status, real_time = readmds(shot, time)

x, y= data['Te']['TS']['Rho'], data['Te']['TS']['data']

def IQR(a, b):
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

def find_delete(xorigin, xclean, yorigin):  # -1表示需要留下来的值
    yorigin = yorigin
    new_xorigin = np.full_like(xorigin, -1)
    new_yorigin = np.full_like(yorigin, -1)

    # 替换 np.isin 为 np.in1d
    mask = ~np.in1d(xorigin, xclean)
    new_xorigin[mask] = xorigin[mask]
    new_yorigin[mask] = yorigin[mask]

    index = ~np.isnan(new_yorigin) & (new_yorigin != -1)
    new_xorigin = new_xorigin[index]
    new_yorigin = new_yorigin[index]

    return new_xorigin, new_yorigin

a, b = IQR(x,y)
del_x, del_y = find_delete(x, a, y)
plt.scatter(x,y,marker='.', c='r',label='Original Data')
plt.scatter(del_x,del_y/1000., c='g',marker='x',label='Deleted data')
plt.xlabel(r'$\rho$', fontsize=10)
plt.ylabel('Te(TS)(10$^{3}$  eV)', fontsize=10)
plt.show()
