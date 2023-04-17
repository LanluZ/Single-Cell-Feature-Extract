import cv2
import math

import numpy as np


# 内接圆绘制
def draw_circle_in(filename, img, contours_arr):
    # 计算到轮廓的距离
    raw_dist = np.empty(img.shape, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            raw_dist[i, j] = cv2.pointPolygonTest(contours_arr[1], (j, i), True)

    # 获取最大值即内接圆半径，中心点坐标
    min_val, max_val, _, max_dist_pt = cv2.minMaxLoc(raw_dist)
    max_val = abs(max_val)

    # 画出最大内接圆 避免出事
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    radius = np.int_(max_val)
    cv2.circle(result, max_dist_pt, radius, (0, 0, 255), 1, 1, 0)
    cv2.imwrite('./Out/CircleIn/' + filename, result)

    return radius * 2


# 外接圆绘制
def draw_circle_out(filename, img, contours_arr):
    cnt = contours_arr[1]

    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))  # 最小内接圆圆心
    radius = int(radius)  # 半径
    cv2.circle(img, center, radius, (0, 255, 0), 1)
    cv2.circle(img, center, 1, (0, 255, 0), 1)
    cv2.imwrite('./Out/CircleOut/' + filename, img)

    return radius * 2


# 矩形度计算
def cal_rectangle_degree(contours_area, contours_arr):
    bound_rect = cv2.minAreaRect(contours_arr[1])  # 获取最小外接矩形
    box = cv2.boxPoints(bound_rect)  # 转化为矩形点集
    area_rect = cv2.contourArea(box)
    return contours_area / area_rect  # 图像面积除以矩形面积


# 圆度计算
def cal_circle_degree(contours_area, contours_length):
    return 4 * math.pi * contours_area / (contours_length ** 2)
