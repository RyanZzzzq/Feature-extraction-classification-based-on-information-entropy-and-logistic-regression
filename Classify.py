"""
-*- coding: utf-8 -*-
Desc: 逻辑回归分类数据集
Auth: Zhou Ziqi
GitHub：RyanZzzzq
Intro：对已经使用信息熵算法提取特征的数据集进行逻辑回归分类，得到分类结果
Version：v1.0
Time：2023.7.6
"""


import numpy as np
import matplotlib.pyplot as plt


# 初始化
def initial():
    # 引入数据集
    file_path = './test/Test.csv'  # 数据集位置
    # 数据初步预处理
    raw_data = np.loadtxt(file_path, delimiter=',', skiprows=0)
    height, width = raw_data.shape
    data = np.array(raw_data)  # 用np.array()将二维矩阵转化为numpy数据标准
    print("矩阵高度为：", height)  # 输出矩阵的高度和宽度
    print("矩阵宽度为：", width)
    print("输入数据集为：", data)  # 输出数据集
    return data


def logistic(data):
    # 划分数据集
    train_x = data[:, 0:2]  # 划分训练集和特征集，训练集为前两列，特征集为后一列
    train_y = data[:, 2]
    print("训练集x矩阵宽高为：", train_x.shape)
    print("特征集y矩阵宽高为：", train_y.shape)

    # 初始化参数
    theta = np.random.rand(4)  # 随机生成0-1之间的随机数
    # 准确度的记录
    accuracies = []
    # 标准化
    mu = train_x.mean(axis=0)
    sigma = train_x.std(axis=0)

    # 标准化
    def standardize(x):
        return (x - mu) / sigma

    train_z = standardize(train_x)
    # 增加x0

    def to_matrix(x):
        x0 = np.ones([x.shape[0], 1])
        x3 = x[:, 0, np.newaxis]**2
        return np.hstack([x0, x, x3])

    x_trained = to_matrix(train_z)
    print("标准化后的训练集为：", train_z)
    # 绘制标准化后的训练数据图
    plt.plot(train_z[train_y == 0, 0], train_z[train_y == 0, 1], 'o')
    plt.plot(train_z[train_y == 1, 0], train_z[train_y == 1, 1], 'x')
    plt.show()

    # Sigmoid函数
    def f(x):
        return 1 / (1 + np.exp(-np.dot(x, theta)))

    # 分类函数
    def classify(x):
        return (f(x) >= 0.5).astype(int)
    classify_result = classify(to_matrix(standardize(train_x)))
    print("分类的结果为：", classify_result)

    # 学习率
    eta = 1e-3
    # 重复次数
    epoch = 2000
    count = 0
    # 重复学习
    for _ in range(epoch):
        p = np.random.permutation(x_trained.shape[0])
        for x, y in zip(x_trained[p, :], train_y[p]):
            theta = theta - eta * (f(x) - y) * x
            # theta = theta - eta * np.dot(f(x_trained) - train_y, x_trained)
        # 日志输出
        count += 1
        print(' 第{} 次: theta = {}'.format(count, theta))
        result = classify(x_trained) == train_y
        accuracy = len(result[result == True]) / len(result)
        accuracies.append(accuracy)
    # 精度绘图
    plot_accuracy = np.arange(len(accuracies))
    plt.plot(plot_accuracy, accuracies)
    plt.show()

    # 绘图确认
    x_1 = np.linspace(-4, 4, 1000)  # 在[-1，4]生成1000个数
    x_2 = -(theta[0] + theta[1] * x_1 + theta[3] * x_1 ** 2) / theta[2]
    plt.plot(train_z[train_y == 0, 0], train_z[train_y == 0, 1], 'o')
    plt.plot(train_z[train_y == 1, 0], train_z[train_y == 1, 1], 'x')
    plt.plot(x_1, x_2, linestyle='dashed')
    plt.show()


if __name__ == '__main__':
    # 导入数据集并简单预处理
    Main_data = initial()
    # 逻辑回归二分类
    logistic(Main_data)
