# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from readMDS import readmds

# shot = int(input('shot:'))#73999  106915

# time = float(input('time:'))#20.5
# data, status = readmds(shot, time)



# # x, y = data['ne']['Refl']['Rho'], data['ne']['Refl']['data']
# x, y= data['Te']['TS']['Rho'], data['Te']['TS']['data']
#        window_size - 移动窗口大小
#        threshold - 用于判断异常的阈值
def SMA(ws,thd,x,y):

    def moving_average(data, window_size):
        """
        计算移动平均值
        参数:
        data - 输入数据   1D 数组
        window_size - 移动窗口大小

        返回:
        移动平均值数组
        """
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    def detect_anomalies(data, window_size, threshold):
        """
        使用移动平均法检测异常值
        参数:
        data - 输入数据   1D 数组
        window_size - 移动窗口大小
        threshold - 用于判断异常的阈值
        
        返回:
        anomaly_indices - 异常值的索引列表
        """

        if len(data) < window_size:
            print("数据长度不足以计算移动平均")
            return []  # 或其他默认返回值
        moving_avg = moving_average(data, window_size)
        diff = np.abs(data[window_size-1:] - moving_avg)
        
        # 检测异常值，若数据点与移动平均值的差值大于阈值，则认为是异常
        anomaly_indices = np.where(diff > threshold)[0] + (window_size - 1)
        return anomaly_indices

    # 参数设定
    window_size = ws
    threshold = thd

    # 检测异常值
    anomalies = detect_anomalies(y, window_size, threshold)
    y = np.delete(y, anomalies)
    x = np.delete(x, anomalies)
    return x,y


def SMA_ped(ws, thd, x, y):
    def moving_average(data, window_size):
        """
        计算移动平均值
        参数:
        data - 输入数据 1D 数组
        window_size - 移动窗口大小
        返回:
        移动平均值数组
        """
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    def detect_anomalies(data, window_size, threshold):
        """
        使用移动平均法检测异常值
        参数:
        data - 输入数据 1D 数组
        window_size - 移动窗口大小
        threshold - 用于判断异常的阈值
        返回:
        anomaly_indices - 异常值的索引列表
        """
        if len(data) < window_size:
            print("数据长度不足以计算移动平均")
            return []  # 或其他默认返回值
        moving_avg = moving_average(data, window_size)
        diff = np.abs(data[window_size - 1:] - moving_avg)
        
        # 检测异常值，若数据点与移动平均值的差值大于阈值，则认为是异常
        anomaly_indices = np.where(diff > threshold)[0] + (window_size - 1)
        return anomaly_indices

    # 筛选出 x > 0.785 的数据
    mask = x > 0.785
    x_filtered = x[mask]
    y_filtered = y[mask]

    # 参数设定
    window_size = ws
    threshold = thd

    # 检测异常值
    anomalies = detect_anomalies(y_filtered, window_size, threshold)
    y_filtered = np.delete(y_filtered, anomalies)
    x_filtered = np.delete(x_filtered, anomalies)

    # 将处理后的数据返回
    return x_filtered, y_filtered



# filtered_x, filtered_y, anomalies = SMA(3,1.2,x,y)
# # 绘图
# plt.figure(figsize=(10, 6))
# plt.scatter(x, y,marker='.', c='r',label='Original Data')
# # plt.plot(x[window_size-1:], moving_average(y, window_size), label='Moving Average', color='red')
# # plt.scatter(filtered_x,filtered_y)
# plt.scatter(x[anomalies], y[anomalies], color='g', label='Anomalies', marker='x')
# plt.legend()
# plt.title('Moving Average Method for Anomaly Detection')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()

