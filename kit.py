import cv2
import math
import os.path

import numpy as np


# 内接圆计算
def calCircleIn(img, contours_arr):
    # 计算到轮廓的距离
    raw_dist = np.empty(img.shape, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            raw_dist[i, j] = cv2.pointPolygonTest(contours_arr, (j, i), True)

    # 获取最大值即内接圆半径，中心点坐标
    _min_val, max_val, _, _max_dist_pt = cv2.minMaxLoc(raw_dist)
    max_val = abs(max_val)
    radius = np.int_(max_val)
    return radius


# 内接圆绘制
def drawCircleIn(filename, save_path, img, contours_arr):
    # 计算到轮廓的距离
    raw_dist = np.empty(img.shape, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            raw_dist[i, j] = cv2.pointPolygonTest(contours_arr, (j, i), True)

    # 获取最大值即内接圆半径，中心点坐标
    _min_val, max_val, _, max_dist_pt = cv2.minMaxLoc(raw_dist)
    max_val = abs(max_val)

    # 画出最大内接圆 避免出事
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    radius = np.int_(max_val)
    cv2.circle(result, max_dist_pt, radius, (0, 0, 255), 1, 1, 0)
    cv2.imwrite(os.path.join(save_path, filename), result)


# 外接圆计算
def calCircleOut(contour):
    cnt = contour
    (_, _), radius = cv2.minEnclosingCircle(cnt)
    radius = int(radius)  # 半径
    return radius


# 外接圆绘制
def drawCircleOut(filename, save_path, img, contours):
    cnt = contours

    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))  # 最小内接圆圆心
    radius = int(radius)  # 半径
    cv2.circle(img, center, radius, (0, 255, 0), 1)
    cv2.circle(img, center, 1, (0, 255, 0), 1)
    cv2.imwrite(os.path.join(save_path, filename), img)


# 矩形度计算
def calRectangleDegree(contours_area, contours):
    bound_rect = cv2.minAreaRect(contours)  # 获取最小外接矩形
    box = cv2.boxPoints(bound_rect)  # 转化为矩形点集
    area_rect = cv2.contourArea(box)
    return contours_area / area_rect  # 图像面积除以矩形面积


# 圆度计算
def calCircleDegree(contours_area, contours_length):
    return 4 * math.pi * contours_area / (contours_length ** 2)
