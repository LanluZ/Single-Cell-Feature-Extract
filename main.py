import csv
import os

from kit import *


def main():
    origin_file = []  # 存储将要处理的图片
    current_path = os.getcwd().replace('\\', '/') + '/Data'  # 获取当前所在目录
    for filename in os.listdir(current_path):  # 获取目录下文件名称
        if filename[-3:len(filename)] == 'tif' and filename[0:4] != 'draw':  # 筛选需要处理的图片
            origin_file.append(filename)

    # 删除先前处理文件
    out_path = current_path[0:-5] + '/Out'
    for filename in os.listdir(out_path + '/CircleIn'):  # 获取目录下文件名称
        os.remove(out_path + '/CircleIn/' + filename)
    for filename in os.listdir(out_path + '/CircleOut'):  # 获取目录下文件名称
        os.remove(out_path + '/CircleOut/' + filename)
    for filename in os.listdir(out_path):  # 获取目录下文件名称
        if filename[-3:len(filename)] == 'tif':  # 筛选需要处理的图片
            os.remove(out_path + '/' + filename)

    # 数据存储对象
    csvfile = open('./Out/Data.csv', mode='w', newline='')
    fieldnames = ['filename', 'length', 'area', 'inscribedCircle', 'circumscribedCircle', 'specificValue',
                  'rectangleDegree', 'circleDegree']
    write = csv.DictWriter(csvfile, fieldnames=fieldnames)
    write.writeheader()

    # 遍历图片文件
    for filename in origin_file:
        data_dic = dict()  # 数据记录
        data_dic['filename'] = filename[0:-4]
        im = cv2.imread('./Data/' + filename)
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # 以灰度读入图片
        im_gray = cv2.blur(im_gray, (3, 3))  # 图像滤波

        ret, binary = cv2.threshold(im_gray, 180, 255, cv2.THRESH_BINARY)
        # 反色
        binary = cv2.bitwise_not(binary)

        # 图像轮廓获取
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(im, contours, 0, (0, 0, 255), 1)
        cv2.imwrite('./Out/Draw-' + filename, im)  # 图像轮廓输出 用于检查 防止出事

        # 像素面积获取
        data_dic['area'] = cv2.contourArea(contours[0])

        # 轮廓周长
        data_dic['length'] = cv2.arcLength(contours[0], True)

        # 内接圆绘制
        # data_dic['inscribedCircle'] = draw_circle_in(filename, closed, contours[0])

        # 外接圆绘制
        # data_dic['circumscribedCircle'] = draw_circle_out(filename, closed, contours[0])

        # 内接圆计算
        data_dic['inscribedCircle'] = calCircleIn(im_gray, contours[0])

        # 外接圆计算
        data_dic['circumscribedCircle'] = calCircleOut(contours[0])

        # 最小外接圆与最大内接圆直径比值
        data_dic['specificValue'] = data_dic['inscribedCircle'] / data_dic['circumscribedCircle']

        # 矩形度计算
        data_dic['rectangleDegree'] = calRectangleDegree(data_dic['area'], contours[0])

        # 圆度计算
        data_dic['circleDegree'] = calCircleDegree(data_dic['area'], data_dic['length'])

        # 数据写入
        write.writerow(data_dic)


if __name__ == '__main__':
    main()
