# -*- coding: utf-8 -*-
"""
Created on Fri May 29 10:17:53 2020

@author: LZY
"""

#! -*- coding: utf-8 -*-
# py37
# 时间:2019年4月30日09:56:17
# DIP：形态学操作处理应用（粒子测度）
# 粒子测度是一种对图像中粒子的尺度分布进行测量的操作
# 原理：开操作对输入图像中与结构元尺度相似的粒子的亮区域影响最大

import cv2
import matplotlib.pyplot as plt


def particleMeasure(imgBin):
    sizes = []
    values = []
    # maxSize = int(imgBin.shape[0] / 4)  # 只分析比图像的高度小一半的圆
    maxSize = 100

    #  不断增加kernel的大小，计算开运算对图像的影响
    for size in range(3, maxSize, 2):
        # 定义结构元的类型和大小
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size - 2, size - 2))
        imgOpen_1 = cv2.morphologyEx(imgBin, cv2.MORPH_OPEN, kernel=kernel)
        # 定义结构元的类型和大小
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        imgOpen_2 = cv2.morphologyEx(imgBin, cv2.MORPH_OPEN, kernel=kernel)

        diffImg = imgOpen_2 - imgOpen_1

        # image, cnts, hierarchy = cv2.findContours(diffImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓
        diffValue = cv2.sumElems(diffImg)[0] / 255  # 统计一个通道腐蚀导致减少的点的数量

        sizes.append(size)  # 结构元的size 画图时的横坐标
        # if diffValue / (len(cnts) + 0.01) > 1:  # 总减少的面积/总的个数，等于每个减少的面积，只统计开操作使面积减少较大的结构元size
        values.append(diffValue / size)  # 区域的面积除以结构元的大小，画图时的纵坐标
        # else:
        #     values.append(0)
    MAX = max(values)
    values = [i if i > 0.03 * MAX else 0 for i in values]  # 去掉较小的数据
    values = [i / sum(values) for i in values]  # 计算各个尺寸占的比例
    return sizes, values


if __name__ == "__main__":
    fileName = "Chapter5_1.bmp"  # 作业要求的测试图片
#    fileName = "images5_1.png"  # 自己制作的测试图片
    img = cv2.imread(fileName)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 转成灰度图
    print(gray.shape)  # (274, 279)

    img2 = cv2.GaussianBlur(gray, (5, 5), 0.05)  # 平滑去燥
    _, imgBin = cv2.threshold(img2, 80, 255, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 设阈值二值化

    sizes, values = particleMeasure(imgBin)  # 计算开操作对二值图的影响

    plt.bar(sizes, values)   # 绘制粒子测度图
    plt.text(30, 0.12, "remove low ratio data")
    plt.ylabel("ratio")
    plt.xlabel("rectangle kernel size")
    plt.show()
    print("END ")
