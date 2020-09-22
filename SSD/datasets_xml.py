try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import cv2
import os
import shutil
import random
import colorsys
import matplotlib.pyplot as plt
import matplotlib.image as imgplt
import numpy as np


def convert(s, box):
    """
    :param s: 图像总大小，二维元组或列表 (w, h)
    :param box: 四维元组或列表 (xmin, xmax, ymin, ymax)
    :return: (x, y, w, h)
    """
    dw = 1. / s[0]
    dh = 1. / s[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def de_convert(x, y, w, h) -> list:
    """
    :return: box (xmin, ymin, xmax, ymax) 注意顺序
    """
    s = [500, 500]
    dw = 1. / s[0]
    dh = 1. / s[1]
    x = x / dw
    w = w / dw
    y = y / dh
    h = h / dh
    box = [0, 0, 0, 0]
    box[0] = int((2 * x - w) / 2)
    box[2] = int((2 * x + w) / 2)
    box[1] = int((2 * y - h) / 2)
    box[3] = int((2 * y + h) / 2)
    return box


def draw(image, box):
    """
    :param image: 图片路径
    :param box: 左上和右下坐标 (xmin, ymin, xmax, ymax) 注意顺序!
    :return: None
    """
    image = imgplt.imread(image)
    assert image is not None, 'Image is not found, No such file or directory'
    # plt.imshow(image)
    # plt.show()
    hsv_tuples = [(1.0 * x / 1, 1., 1.) for x in range(1)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(0)
    random.shuffle(colors)
    random.seed(None)
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), colors[0], 2)
    plt.imshow(image)
    plt.show()


class BboxError(object):
    def __init__(self):
        self.error_list = []

    def error_process(self, num, name):
        if num[0] >= num[2] or num[1] >= num[3]:
            tempt = num[0:2]
            num[0] = num[2]
            num[1] = num[3]
            num[2] = tempt[0]
            num[3] = tempt[1]
        for elem in num:
            if elem < 0:
                self.error_list.append(name)
            elif elem > 500:
                self.error_list.append(name)
        return num

    def error_summary(self):
        print('[INFO] total error: {}'.format(len(self.error_list)))


workspace_dir = 'D:/D_Python code/darknet-master-opencv340-cuda10.1/build/darknet/x64/data/VOCdevkit/VOC2007'
save_image_dir = workspace_dir + '/JPEGImages'
save_xml_dir = workspace_dir + '/Annotations'
save_label_dir = workspace_dir + '/labels'

if __name__ == '__main__':
    assert os.path.isdir(workspace_dir), "[ERROR] workspace_dir:{} is not found!".format(workspace_dir)
    assert os.path.isdir(save_image_dir), "[ERROR] save_image_dir:{} is not found!".format(save_image_dir)
    assert os.path.isdir(save_xml_dir), "[ERROR] save_xml_dir:{} is not found!".format(save_xml_dir)
    assert os.path.isdir(save_label_dir), "[ERROR] save_label_dir:{} is not found!".format(save_label_dir)

    print("[INFO] loading image files into index...")
    image_list = os.listdir(save_image_dir)
    label_list = os.listdir(save_label_dir)
    assert len(image_list) == len(label_list), "[ERROR] the amounts of images and labels are not equal!"
    print("[INFO] load {} files in total! ".format(len(image_list)))

    # convert label to xml
    error = BboxError()
    print('\n=================================')
    for label in label_list:
        print('[INFO] converting: {}'.format(label))
        with open(save_label_dir + '/' + label, 'r') as f:
            data = f.read()
            # 从label提取bbox
            data = list(map(float, data[2:-1].split()))
            # bbox逆转换
            data = de_convert(data[0], data[1], data[2], data[3])
            # 处理错误
            data = error.error_process(data, os.path.splitext(label)[0])
            # draw(save_image_dir + '/%s.jpg' % os.path.splitext(label)[0], data)

        # 从sample模板复制sml
        shutil.copy(workspace_dir + '/sample.xml', save_xml_dir + '/%s.xml' % os.path.splitext(label)[0])
        # 读取xml
        tree = ET.ElementTree(file=save_xml_dir + '/%s.xml' % os.path.splitext(label)[0])
        # 获取根节点
        child = tree.getroot()
        child[1].text = os.path.splitext(label)[0] + '.jpg'  # filename
        child[4][0].text = child[4][1].text = '500'  # size-width/height
        child[4][2].text = '3'  # size-depth
        child[6][0].text = 'face'  # object-name
        child[6][4][0].text = str(data[0])
        child[6][4][1].text = str(data[1])
        child[6][4][2].text = str(data[2])
        child[6][4][3].text = str(data[3])  # bbox
        tree.write(save_xml_dir + '/%s.xml' % os.path.splitext(label)[0], encoding='UTF-8')
    print('[INFO] convert complete!')
    print('\n=================================')
    # 处理错误
    error.error_summary()
    for f in error.error_list:
        print('[INFO] delet {}'.format(f))
        os.remove(save_xml_dir + '/{}.xml'.format(f))

# [INFO] delet 2546
# [INFO] delet 3494
# [INFO] delet 4188
# [INFO] delet 5392
# [INFO] delet 6476
# [INFO] delet 7687
# total: 9766-6=9760
# 1003?

