import csv
import shutil

from kit import *
from move import *

file_suffix = 'png'  # 预处理文件后缀


def main():
    # 请自行修改目标存储位置
    aim_dir = './Finish'
    # 记录存储预处理图片存放的文件夹
    dir_list = os.listdir('./Data')

    # 批量处理
    for dirname in dir_list:

        file_arr = []  # 存储将要处理的图片
        current_path = os.path.dirname(__file__)  # 获取当前所在目录
        for filename in os.listdir(current_path + '/Data/' + dirname):  # 获取目录下文件名称
            file_arr.append(filename)

        # 删除并创建先前处理文件
        try:
            shutil.rmtree(os.path.join(current_path, 'Out'))
        except FileNotFoundError:
            pass

        os.mkdir('./Out')

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
            img = cv2.imread(os.path.join(current_path, 'Data', dirname, filename), cv2.IMREAD_UNCHANGED)
            img = cv2.blur(img, (1, 1))  # 图像滤波

            img_binary = None
            if file_suffix == 'png':
                # 获取透明度通道
                alpha_channel = img[:, :, 3]
                # 将透明部分填充为黑色
                img[alpha_channel == 0] = [0, 0, 0, 255]  # 将RGB通道值设置为黑色，透明度设置为255
                # 转化为8UC3三通道图像
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                # 图像二值化
                ret, img_binary = cv2.threshold(img, 12, 255, cv2.THRESH_BINARY)
                # 转化为8UC1单通道图像
                img_binary = cv2.cvtColor(img_binary, cv2.COLOR_BGR2GRAY)
            elif file_suffix == 'jpg':
                # 图像二值化
                ret, img_binary = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
                # 转化为8UC1单通道图像
                img_binary = cv2.cvtColor(img_binary, cv2.COLOR_BGR2GRAY)
                # 反色
                img_binary = cv2.bitwise_not(img_binary)

            print(f"开始处理:{str(filename)} 图形尺寸:{img_binary.shape}", end=' ')

            # 图像轮廓获取
            contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            index = getMaxCounterIndex(contours)
            print(f"轮廓获取完成", end=' ')

            # 轮廓绘制
            cv2.drawContours(img_binary, contours, index, (118, 215, 234), 1)
            cv2.imwrite('./Out/Draw-' + filename, img_binary)  # 图像轮廓输出 用于检查 防止出事

            # 内接圆绘制
            if not os.path.isdir('./Out/CI'):
                os.mkdir('./Out/CI')
            inscribed_circle = drawCircleIn(filename, './Out/CI', img, contours[index])

            print(f"内接圆获取完成", end=' ')

            # 外接圆绘制
            if not os.path.isdir('./Out/CO'):
                os.mkdir('./Out/CO')
            circumscribed_circle = drawCircleOut(filename, './Out/CO', img, contours[index])

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


if __name__ == '__main__':
    main()
