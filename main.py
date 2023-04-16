import csv
import cv2
import os

import numpy as np


# 内接圆绘制
def circle_in(filename, img, contours_arr):
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
def circle_out(filename, img, contours_arr):
    cnt = contours_arr[1]

    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))  # 最小内接圆圆心
    radius = int(radius)  # 半径
    cv2.circle(img, center, radius, (0, 255, 0), 1)
    cv2.circle(img, center, 1, (0, 255, 0), 1)
    cv2.imwrite('./Out/CircleOut/' + filename, img)

    return radius * 2


originFile = []  # 存储将要处理的图片
currentPath = os.getcwd().replace('\\', '/') + '/Data'  # 获取当前所在目录
for fileName in os.listdir(currentPath):  # 获取目录下文件名称
    if fileName[-3:len(fileName)] == 'tif' and fileName[0:4] != 'draw':  # 筛选需要处理的图片
        originFile.append(fileName)

# 数据存储对象
csvfile = open('./Out/Data.csv', mode='w', newline='')
fieldnames = ['filename', 'area', 'inscribedCircle', 'circumscribedCircle', 'specificValue']
write = csv.DictWriter(csvfile, fieldnames=fieldnames)
write.writeheader()

# 遍历图片文件
for fileName in originFile:
    dataDic = dict()  # 数据记录
    dataDic['filename'] = fileName[0:-4]
    im = cv2.imread('./Data/' + fileName)
    imGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # 图像二值化
    imGray = cv2.blur(imGray, (3, 3))  # 图像滤波

    # 图像预处理
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    opened = cv2.morphologyEx(imGray, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    ret, binary = cv2.threshold(closed, 180, 255, cv2.THRESH_BINARY)

    # 图像轮廓获取
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(im, contours, 1, (0, 0, 255), 1)
    cv2.imwrite('./Out/Draw-' + fileName, im)  # 图像轮廓输出 用于检查 防止出事

    # 内接圆绘制
    dataDic['inscribedCircle'] = circle_in(fileName, closed, contours)

    # 外接圆绘制
    dataDic['circumscribedCircle'] = circle_out(fileName, closed, contours)

    # 像素面积获取
    dataDic['area'] = cv2.contourArea(contours[1])

    # 最小外接圆与最大内接圆直径比值
    dataDic['specificValue'] = dataDic['inscribedCircle'] / dataDic['circumscribedCircle']

    # 数据写入
    write.writerow(dataDic)
