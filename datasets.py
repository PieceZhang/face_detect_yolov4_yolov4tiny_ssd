import cv2
import json
import os
import random
import colorsys
import matplotlib.pyplot as plt
import matplotlib.image as imgplt
import numpy as np

def convert(s, box):
    """
    :param s: 二维元组或列表 (w, h)
    :param box: 四维元组或列表 (xmin, xmax, ymin, ymax)
    :return: (x, y, w, h)
    """
    dw = 1. / s[0]
    dh = 1. / s[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return x, y, w, h

def de_convert(s, x, y, w, h) -> list:
    dw = 1. / s[0]
    dh = 1. / s[1]
    x = x/dw
    w = w/dw
    y = y/dh
    h = h/dh
    box = [0, 0, 0, 0]
    box[0] = int((2*x-w)/2)
    box[1] = int((2*x+w)/2)
    box[2] = int((2*y-h)/2)
    box[3] = int((2 * y + h) / 2)
    return box

def draw(image, box):
    """
    :param image: 图片
    :param box: 左上和右下坐标
    :return: None
    """
    image = imgplt.imread(image)
    assert image is not None, 'Image is not found, No such file or directory'
    plt.imshow(image)
    plt.show()
    hsv_tuples = [(1.0 * x / 1, 1., 1.) for x in range(1)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(0)
    random.shuffle(colors)
    random.seed(None)
    cv2.rectangle(image, (box[0], box[2]), (box[1], box[3]), colors[0], 2)
    plt.imshow(image)
    plt.show()

image_dir = 'D:/A_科研/毕业设计/celebrity_test_200'
bg_dir = 'D:/A_科研/datasets/indoorCVPR_09'
workspace_dir = './datasets'
save_image_dir = workspace_dir + '/JPEGImages'
save_label_dir = workspace_dir + '/labels'


if __name__ == '__main__':
    if not os.path.isdir(workspace_dir):
        os.mkdir(workspace_dir)
    if not os.path.isdir(save_image_dir):
        os.mkdir(save_image_dir)
    if not os.path.isdir(save_label_dir):
        os.mkdir(save_label_dir)

    bbox = []
    bg_size = [500, 500]
    face_size = [0, 0]
    face_location = [0, 0]  # 融合的中点坐标

    bg_filist = os.listdir(bg_dir)
    for faces_idx in range(170, 201):
        if not os.path.isdir(image_dir + '/Faces (%s)' % faces_idx):
            continue
        print('Images in Faces ({})'.format(faces_idx))
        for face_file in os.listdir(image_dir + '/Faces (%s)' % faces_idx):
            # 处理jpg
            if os.path.splitext(face_file)[-1] == '.jpg':
                # 随机选取背景
                bg_image = imgplt.imread(bg_dir + '/' + str(random.choice(bg_filist)))
                bg_image = cv2.resize(bg_image, (bg_size[0], bg_size[1]), interpolation=cv2.INTER_CUBIC)
                # 读取face图片
                face_image = imgplt.imread(image_dir + '/Faces (%s)' % faces_idx + '/' + face_file)
                # 随机缩放0.3-1.1倍
                face_size[0] = face_size[1] = random.randint(120, 440)
                face_image = cv2.resize(face_image, (face_size[0], face_size[1]), interpolation=cv2.INTER_CUBIC)
                # 融合face至背景上随机位置
                face_location[0] = random.randint(int(face_size[0] / 2) + 1, bg_size[0] - int(face_size[0] / 2) - 1)
                face_location[1] = random.randint(int(face_size[1] / 2) + 1, bg_size[1] - int(face_size[1] / 2) - 1)
                face_image = cv2.seamlessClone(face_image, bg_image,
                                               255 * np.ones(face_image.shape, face_image.dtype),
                                               (face_location[0], face_location[1]), cv2.NORMAL_CLONE)
                plt.imsave(save_image_dir + '/' + face_file, face_image)
            # 处理json
            else:
                print(os.path.splitext(face_file)[0])
                # 转换为比例
                multiplier = face_size[0] / 400
                # 读取json
                image_json = json.loads(open(image_dir + '/Faces (%s)' % faces_idx + '/' + face_file, 'r').read())
                # 解码json
                bbox.append(int(image_json['faces'][0]['face_rectangle']['left'] * multiplier
                            + face_location[0] - face_size[0] / 2))  # xmin
                bbox.append(int(image_json['faces'][0]['face_rectangle']['width'] * multiplier + bbox[0]))  # xmax
                bbox.append(int(image_json['faces'][0]['face_rectangle']['top'] * multiplier
                            + face_location[1] - face_size[1] / 2))  # ymin
                bbox.append(int(image_json['faces'][0]['face_rectangle']['height'] * multiplier + bbox[2]))  # ymax
                # show image and draw box #
                # draw(face_image, bbox)
                # convert to lebal #
                bbox = convert(bg_size, bbox)  # convert to (x, y, w, h)
                out_file = open(save_label_dir + '/%s.txt'
                                % os.path.splitext(face_file)[0], 'w')
                out_file.write(str(0) + " " + " ".join([str(a) for a in bbox]) + '\n')
                out_file.close()
                bbox = []

    # 验证
    # draw(save_image_dir + '/' + '0.jpg',
    #      de_convert(bg_size, 0.5680000000000001, 0.658, 0.364, 0.364))
    filist = os.listdir(save_label_dir)
    for file in os.listdir(save_image_dir):
        if filist.count(os.path.splitext(file)[0] + '.txt') != 1:
            print('error in: ', file, filist.count(file))

    # 打乱顺序
    filist = os.listdir(save_label_dir)
    random.shuffle(filist)
    for idx, file in enumerate(filist):
        os.rename(save_label_dir + '/' + file, save_label_dir + '/' + str(idx) + '.txt')
        os.rename(save_image_dir + '/' + os.path.splitext(file)[0] + '.jpg',
                  save_image_dir + '/' + str(idx) + '.jpg')
