import copy

import cv2
import math
import os.path

import numpy as np

import datetime


# 内接圆计算
def calCircleIn(img, contours_arr):
    # 计算到轮廓的距离
    raw_dist = np.empty((img.shape[0], img.shape[1]), dtype=np.float32)
    start_i = contours_arr[0, 0, 1]  # 起始行
    end_i = contours_arr[-1, 0, 1] + 1  # 结束行
    for i in range(start_i, end_i):
        for j in range(img.shape[1]):
            raw_dist[i, j] = cv2.pointPolygonTest(contours_arr, (j, i), True)

    # 获取最大值即内接圆半径，中心点坐标
    _min_val, max_val, _, _max_dist_pt = cv2.minMaxLoc(raw_dist)
    max_val = abs(max_val)
    radius = np.int_(max_val)

    return radius, _max_dist_pt  # 返回半径和中心点坐标


# 内接圆绘制
def drawCircleIn(filename, save_path, img, contours_arr):  # 画出最大内接圆 避免出事
    img = copy.copy(img)  # 防止指向同一内存
    radius, center = calCircleIn(img, contours_arr)
    cv2.circle(img, center, radius, (0, 255, 0), 1)
    cv2.circle(img, center, 1, (0, 255, 0), 1)
    cv2.imwrite(os.path.join(save_path, filename), img)

    return radius


# 外接圆计算
def calCircleOut(contour):
    cnt = contour
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    radius = int(radius)  # 半径
    return radius, (x, y)


# 外接圆绘制
def drawCircleOut(filename, save_path, img, contour):
    img = copy.copy(img)  # 防止指向同一内存
    radius, (x, y) = calCircleOut(contour)
    center = (int(x), int(y))  # 最小内接圆圆心
    radius = int(radius)  # 半径
    cv2.circle(img, center, radius, (0, 255, 0), 1)
    cv2.circle(img, center, 1, (0, 255, 0), 1)
    cv2.imwrite(os.path.join(save_path, filename), img)

    return radius


# 矩形度计算
def calRectangleDegree(contours_area, contours):
    bound_rect = cv2.minAreaRect(contours)  # 获取最小外接矩形
    box = cv2.boxPoints(bound_rect)  # 转化为矩形点集
    area_rect = cv2.contourArea(box)
    return contours_area / area_rect  # 图像面积除以矩形面积


# 圆度计算
def calCircleDegree(contours_area, contours_length):
    return 4 * math.pi * contours_area / (contours_length ** 2)
