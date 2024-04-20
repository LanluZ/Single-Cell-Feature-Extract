import csv

from kit import *
from move import *


def main():
    # 请自行修改目标存储位置
    aim_dir = './Finish'
    # 记录存储预处理图片存放的文件夹
    dir_list = os.listdir('./Data')

    # 批量处理
    for dirname in dir_list:

        file_arr = []  # 存储将要处理的图片
        current_path = os.getcwd().replace('\\', '/')  # 获取当前所在目录
        for filename in os.listdir(current_path + '/Data/' + dirname):  # 获取目录下文件名称
            file_arr.append(filename)

        # 删除先前处理文件
        delDirFile(current_path + '/Out')

        # 数据存储对象
        csvfile = open('./Out/Data.csv', mode='w', newline='')
        fieldnames = ['filename', 'length', 'area', 'inscribedCircle', 'circumscribedCircle', 'specificValue',
                      'rectangleDegree', 'circleDegree']
        write = csv.DictWriter(csvfile, fieldnames=fieldnames)
        write.writeheader()

        # 遍历图片文件
        i = 0
        for filename in file_arr:
            data_dic = dict()  # 数据记录
            im = cv2.imread('./Data/' + dirname + '/' + filename)
            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # 以灰度读入图片
            im_gray = cv2.blur(im_gray, (3, 3))  # 图像滤波

            ret, binary = cv2.threshold(im_gray, 180, 255, cv2.THRESH_BINARY)
            # 反色
            binary = cv2.bitwise_not(binary)

            print(f"开始处理:{str(filename)} 图形尺寸:{binary.shape}", end=' ')

            # 图像轮廓获取
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            index = getMaxCounterIndex(contours)
            print(f"轮廓获取完成", end=' ')

            # 轮廓绘制
            cv2.drawContours(im, contours, index, (118, 215, 234), 1)
            cv2.imwrite('./Out/Draw-' + filename, im)  # 图像轮廓输出 用于检查 防止出事

            # 内接圆绘制
            if not os.path.isdir('./Out/CI'):
                os.mkdir('./Out/CI')
            inscribed_circle = drawCircleIn(filename, './Out/CI', im_gray, contours[index])

            print(f"内接圆获取完成", end=' ')

            # 外接圆绘制
            if not os.path.isdir('./Out/CO'):
                os.mkdir('./Out/CO')
            circumscribed_circle = drawCircleOut(filename, './Out/CO', im_gray, contours[index])

            print(f"外接圆获取完成", end=' ')

            # 图像序号
            data_dic['filename'] = str(filename)

            # 像素面积获取
            data_dic['area'] = cv2.contourArea(contours[index])

            # 轮廓周长
            data_dic['length'] = cv2.arcLength(contours[index], True)

            # 内接圆计算
            data_dic['inscribedCircle'] = inscribed_circle

            # 外接圆计算
            data_dic['circumscribedCircle'] = circumscribed_circle

            # 最小外接圆与最大内接圆直径比值
            data_dic['specificValue'] = inscribed_circle / circumscribed_circle

            # 矩形度计算
            data_dic['rectangleDegree'] = calRectangleDegree(data_dic['area'], contours[index])

            # 圆度计算
            data_dic['circleDegree'] = calCircleDegree(data_dic['area'], data_dic['length'])

            # 数据写入
            write.writerow(data_dic)

            print('数据写出完成')

        csvfile.close()

        moveFile('./Out', aim_dir, dirname)


# 寻找图像中最大的轮廓
def getMaxCounterIndex(contours):
    index = 0
    max_area = 0
    max_index = 0
    for i in contours:
        current_area = cv2.contourArea(i)
        if current_area > max_area:
            max_area = current_area
            max_index = index
        index += 1

    return max_index


# 按特定结构删除指定文件夹下文件
def delDirFile(path):
    for filename in os.listdir(path):  # 获取目录下文件名称
        if filename[-3:len(filename)] == 'tif':  # 筛选需要处理的图片
            os.remove(path + '/' + filename)


if __name__ == '__main__':
    main()
