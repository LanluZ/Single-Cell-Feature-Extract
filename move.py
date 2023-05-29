import os
import shutil


def moveFile(origin_path, aim_path, name):
    files = os.listdir(origin_path)
    try:
        os.makedirs(aim_path + '/' + name)
    except FileExistsError:
        pass
    for filename in files:
        shutil.move(origin_path + '/' + filename, aim_path + '/' + name + '/' + filename)
