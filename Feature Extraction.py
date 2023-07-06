"""
-*- coding: utf-8 -*-
Desc: 信息熵算法特征提取
Auth: Zhou Ziqi
GitHub：RyanZzzzq
Intro：使用信息熵算法对数据集进行特征提取，选出最具有代表性的特征，从而达到降维的目的，便于之后的分类操作
Version：v1.0
Time：2023.7.6
"""


import pandas as pd  # 引用pandas库
import numpy as np  # 引用numpy库


# 数据处理
def main_processing(data, m, n):
    data_feature = np.zeros(shape=(6, n))  # 初始化特征二维矩阵
    for i in range(n):  # 对数据集中的每列
        data_ave = np.mean(data[:, i])  # 求均值
        data_std = np.std(data[:, i])  # 求标准差
        data_max = np.max(data[:, i])  # 求最大值
        data_min = np.min(data[:, i])  # 求最小值
        data_energy = np.sum(np.abs(data[:, i]))  # 求能量（数据绝对值之和）
        data_normal = (data[:, i] - data_min) / (data_max - data_min)  # 数据归一化(0,1)
        segment = int(0.5 * m)
        data_entropy = entropy(data_normal, segment)  # 求信息熵
        data_feature[:, i] = [data_ave, data_std, data_max, data_min, data_energy, data_entropy]  # 特征二维数组
    return data_feature


# 结果写入表格
def output_to_file(data_feature):
    data_f = pd.DataFrame(data_feature)  # 写入数据
    data_f.index = ['ave', 'std', 'max', 'min', 'energy', 'entropy']  # 行标题
    writer = pd.ExcelWriter('./result/SCZ/SCZ_f21.xlsx')  # 写入路径
    data_f.to_excel(writer, 'data_feature', float_format='%.20f')  # data_feature为表格sheet名，float_format为数值精度
    writer.save()  # 保存


# 初始化
def initial():
    csv_file_path = './test/SCZ/s01_block021_SCZ.csv'  # 导入数据地址
    raw_data = pd.read_csv(csv_file_path)  # 导入数据
    height, width = raw_data.shape  # 获得导入表格的高度和宽度
    raw_data = raw_data.values  # 取值，且转化为二维数组
    data = np.array(raw_data)  # 转化为二维数组(矩阵)，转化为numpy数据标准 ,不改变维度
    (m, n) = raw_data.shape  # 矩阵的行数m列数n，返回一个元组
    print(height, width, type(raw_data))
    return [data, m, n]


# 求信息熵
def entropy(vector, segment):  # 自定义一个求解信息熵的函数，vector为向量，segment分段数值
    x_min = np.min(vector)
    x_max = np.max(vector)
    x_dis = np.abs(x_max - x_min)
    x_lower = x_min
    seg = 1.0 / segment
    internal = x_dis * seg
    basic_list = []
    full_list = []
    #
    for i in range(len(vector)):
        if vector[i] >= x_lower + internal:
            basic_list.append(vector[i])
    length_list = len(basic_list)
    full_list.append(length_list)
    #
    for j in range(1, segment):
        basic_list = []
        for i in range(len(vector)):
            if x_lower + j * internal <= vector[i] < x_lower + (j + 1) * internal:
                basic_list.append(vector[i])
        length_list = len(basic_list)
        full_list.append(length_list)
    #
    basic_list = []
    for i in range(len(vector)):
        if vector[i] >= x_lower + (segment - 1) * internal:
            basic_list.append(vector[i])
    length_list = len(basic_list)
    full_list.append(length_list)
    full_list = full_list / np.sum(full_list)
    # y = 0
    full_y = []
    for i in range(segment):
        if full_list[i] == 0:
            y = 0
            full_y.append(y)
        else:
            y = -full_list[i] * np.log2(full_list[i])
            full_y.append(y)
    result = np.sum(full_y)
    return result


# 主函数
if __name__ == '__main__':
    # 导入数据及初始化
    main_data, data_height, data_width = initial()
    # 求信息熵等主要处理
    features = main_processing(main_data, data_height, data_width)
    # 写出数据到表格文件
    output_to_file(features)
