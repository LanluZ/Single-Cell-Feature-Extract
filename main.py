import csv
import os

from kit import *

originFile = []  # 存储将要处理的图片
currentPath = os.getcwd().replace('\\', '/') + '/Data'  # 获取当前所在目录
for fileName in os.listdir(currentPath):  # 获取目录下文件名称
    if fileName[-3:len(fileName)] == 'tif' and fileName[0:4] != 'draw':  # 筛选需要处理的图片
        originFile.append(fileName)

# 删除先前处理文件
outPath = currentPath[0:-5] + '/Out'
for fileName in os.listdir(outPath + '/CircleIn'):  # 获取目录下文件名称
    os.remove(outPath + '/CircleIn/' + fileName)
for fileName in os.listdir(outPath + '/CircleOut'):  # 获取目录下文件名称
    os.remove(outPath + '/CircleOut/' + fileName)
for fileName in os.listdir(outPath):  # 获取目录下文件名称
    if fileName[-3:len(fileName)] == 'tif':  # 筛选需要处理的图片
        os.remove(outPath + '/' + fileName)

# 数据存储对象
csvfile = open('./Out/Data.csv', mode='w', newline='')
fieldnames = ['filename', 'length', 'area', 'inscribedCircle', 'circumscribedCircle', 'specificValue',
              'rectangleDegree', 'circleDegree']
write = csv.DictWriter(csvfile, fieldnames=fieldnames)
write.writeheader()

# 遍历图片文件
for fileName in originFile:
    dataDic = dict()  # 数据记录
    dataDic['filename'] = fileName[0:-4]
    im = cv2.imread('./Data/' + fileName)
    imGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # 以灰度读入图片
    imGray = cv2.blur(imGray, (3, 3))  # 图像滤波

    # 图像预处理
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    opened = cv2.morphologyEx(imGray, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    ret, binary = cv2.threshold(imGray, 180, 255, cv2.THRESH_BINARY)
    # 反色
    binary = cv2.bitwise_not(binary)

    # 图像轮廓获取
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(im, contours, 0, (0, 0, 255), 1)
    cv2.imwrite('./Out/Draw-' + fileName, im)  # 图像轮廓输出 用于检查 防止出事

    # 像素面积获取
    dataDic['area'] = cv2.contourArea(contours[0])

    # 轮廓周长
    dataDic['length'] = cv2.arcLength(contours[0], True)

    # 内接圆绘制
    # dataDic['inscribedCircle'] = draw_circle_in(fileName, closed, contours[0])

    # 外接圆绘制
    # dataDic['circumscribedCircle'] = draw_circle_out(fileName, closed, contours[0])

    # 内接圆计算
    dataDic['inscribedCircle'] = cal_circle_in(imGray, contours[0])

    # 外接圆计算
    dataDic['circumscribedCircle'] = cal_circle_out(contours[0])

    # 最小外接圆与最大内接圆直径比值
    dataDic['specificValue'] = dataDic['inscribedCircle'] / dataDic['circumscribedCircle']

    # 矩形度计算
    dataDic['rectangleDegree'] = cal_rectangle_degree(dataDic['area'], contours[0])

    # 圆度计算
    dataDic['circleDegree'] = cal_circle_degree(dataDic['area'], dataDic['length'])

    # 数据写入
    write.writerow(dataDic)
