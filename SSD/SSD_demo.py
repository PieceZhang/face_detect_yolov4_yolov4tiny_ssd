"""
Input:
# Tensor("Placeholder:0", shape=(?, ?, 3), dtype=uint8)
Outputs:
# Tensor("ssd_300_vgg/block4_box/Reshape:0", shape=(1, 38, 38, 4, 4), dtype=float32)
# Tensor("ssd_300_vgg/softmax/Reshape_1:0", shape=(1, 38, 38, 4, 2), dtype=float32)
# Tensor("ssd_300_vgg/block7_box/Reshape:0", shape=(1, 19, 19, 6, 4), dtype=float32)
# Tensor("ssd_300_vgg/softmax_1/Reshape_1:0", shape=(1, 19, 19, 6, 2), dtype=float32)
# Tensor("ssd_300_vgg/block8_box/Reshape:0", shape=(1, 10, 10, 6, 4), dtype=float32)
# Tensor("ssd_300_vgg/softmax_2/Reshape_1:0", shape=(1, 10, 10, 6, 2), dtype=float32)
# Tensor("ssd_300_vgg/block9_box/Reshape:0", shape=(1, 5, 5, 6, 4), dtype=float32)
# Tensor("ssd_300_vgg/softmax_3/Reshape_1:0", shape=(1, 5, 5, 6, 2), dtype=float32)
# Tensor("ssd_300_vgg/block10_box/Reshape:0", shape=(1, 3, 3, 4, 4), dtype=float32)
# Tensor("ssd_300_vgg/softmax_4/Reshape_1:0", shape=(1, 3, 3, 4, 2), dtype=float32)
# Tensor("ssd_300_vgg/block11_box/Reshape:0", shape=(1, 1, 1, 4, 4), dtype=float32)
# Tensor("ssd_300_vgg/softmax_5/Reshape_1:0", shape=(1, 1, 1, 4, 2), dtype=float32)
"""
# python D:\OpenVINO\openvino_2020.4.287\deployment_tools\model_optimizer\mo.py --input_model SSD_300.pb --data_type FP16 --input_shape (300,300,3)

import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
import random
from collections import namedtuple
from SSD import np_methods

SSDParams = namedtuple('SSDParameters', ['img_shape',
                                         'num_classes',
                                         'no_annotation_label',
                                         'feat_layers',
                                         'feat_shapes',
                                         'anchor_size_bounds',
                                         'anchor_sizes',
                                         'anchor_ratios',
                                         'anchor_steps',
                                         'anchor_offset',
                                         'normalizations',
                                         'prior_scaling'
                                         ])


class SSDNet(object):
    """Implementation of the SSD VGG-based 300 network.

    The default features layers with 300x300 image input are:
      conv4 ==> 38 x 38
      conv7 ==> 19 x 19
      conv8 ==> 10 x 10
      conv9 ==> 5 x 5
      conv10 ==> 3 x 3
      conv11 ==> 1 x 1
    The default image size used to train this network is 300x300.
    """
    default_params = SSDParams(
        img_shape=(300, 300),
        num_classes=2,
        no_annotation_label=2,
        feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11'],
        feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
        anchor_size_bounds=[0.15, 0.90],
        # anchor_size_bounds=[0.20, 0.90],
        anchor_sizes=[(21., 45.),
                      (45., 99.),
                      (99., 153.),
                      (153., 207.),
                      (207., 261.),
                      (261., 315.)],
        # anchor_sizes=[(30., 60.),
        #               (60., 111.),
        #               (111., 162.),
        #               (162., 213.),
        #               (213., 264.),
        #               (264., 315.)],
        anchor_ratios=[[2, .5],
                       [2, .5, 3, 1. / 3],
                       [2, .5, 3, 1. / 3],
                       [2, .5, 3, 1. / 3],
                       [2, .5],
                       [2, .5]],
        anchor_steps=[8, 16, 32, 64, 100, 300],
        anchor_offset=0.5,
        normalizations=[20, -1, -1, -1, -1, -1],
        prior_scaling=[0.1, 0.1, 0.2, 0.2]
    )

    def __init__(self, params=None):
        """Init the SSD net with some parameters. Use the default ones
        if none provided.
        """
        if isinstance(params, SSDParams):
            self.params = params
        else:
            self.params = SSDNet.default_params

    def anchors(self, img_shape, dtype=np.float32):
        """Compute the default anchor boxes, given an image shape.
        """
        return np_methods.ssd_anchors_all_layers(img_shape,
                                                 self.params.feat_shapes,
                                                 self.params.anchor_sizes,
                                                 self.params.anchor_ratios,
                                                 self.params.anchor_steps,
                                                 self.params.anchor_offset,
                                                 dtype)


def draw(image, boxes, classes):
    image_h, image_w, _ = image.shape
    colors = [255, 0, 0]
    for i in range(classes.shape[0]):
        cls_id = int(classes[i])
        if cls_id >= 0:
            boxes[i, 0] = int(boxes[i, 0] * image_h)
            boxes[i, 1] = int(boxes[i, 1] * image_w)
            boxes[i, 2] = int(boxes[i, 2] * image_h)
            boxes[i, 3] = int(boxes[i, 3] * image_w)
    for box in boxes:
        x0, y0, x1, y1 = box
        left = max(0, np.floor(x0 + 0.5).astype(int))
        top = max(0, np.floor(y0 + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x1 + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y1 + 0.5).astype(int))
        bbox_thick = 1
        cv2.rectangle(image, (left, top), (right, bottom), colors, bbox_thick)
    return image


if __name__ == '__main__':
    input_shape = (300, 300)
    ssd_net = SSDNet()
    ssd_anchors = ssd_net.anchors(input_shape)

    with tf.gfile.GFile('./SSD_300.pb', "rb") as pb:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(pb.read())

    # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            name="",  # name可以自定义，修改name之后记得在下面的代码中也要改过来
        )

    # 打印网络结构
    for op in graph.get_operations():
        print(op.name, op.values())

    nodein = graph.get_tensor_by_name('Placeholder:0')
    output = ["ssd_300_vgg/block4_box/Reshape:0", "ssd_300_vgg/softmax/Reshape_1:0",
              "ssd_300_vgg/block7_box/Reshape:0", "ssd_300_vgg/softmax_1/Reshape_1:0",
              "ssd_300_vgg/block8_box/Reshape:0", "ssd_300_vgg/softmax_2/Reshape_1:0",
              "ssd_300_vgg/block9_box/Reshape:0", "ssd_300_vgg/softmax_3/Reshape_1:0",
              "ssd_300_vgg/block10_box/Reshape:0", "ssd_300_vgg/softmax_4/Reshape_1:0",
              "ssd_300_vgg/block11_box/Reshape:0", "ssd_300_vgg/softmax_5/Reshape_1:0"]
    prediction = []
    location = []
    prediction.append(graph.get_tensor_by_name(output[1]))
    prediction.append(graph.get_tensor_by_name(output[3]))
    prediction.append(graph.get_tensor_by_name(output[5]))
    prediction.append(graph.get_tensor_by_name(output[7]))
    prediction.append(graph.get_tensor_by_name(output[9]))
    prediction.append(graph.get_tensor_by_name(output[11]))
    location.append(graph.get_tensor_by_name(output[0]))
    location.append(graph.get_tensor_by_name(output[2]))
    location.append(graph.get_tensor_by_name(output[4]))
    location.append(graph.get_tensor_by_name(output[6]))
    location.append(graph.get_tensor_by_name(output[8]))
    location.append(graph.get_tensor_by_name(output[10]))

    with tf.Session(graph=graph) as sess:
        cam = cv2.VideoCapture(0)
        ifsuccess, frame_origin = cam.read()
        assert ifsuccess is True, 'camera error'

        while 1:
            ifsuccess, frame_origin = cam.read()
            frame_origin = cv2.resize(frame_origin, input_shape, interpolation=cv2.INTER_CUBIC)
            time_start = time.time()  # start
            rprediction, rlocation = sess.run([prediction, location], feed_dict={nodein: frame_origin})
            time_stop = time.time()  # stop
            cost_time = time_stop - time_start
            print('fps: ', 1 / cost_time)

            rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
                rprediction, rlocation, ssd_anchors,
                select_threshold=0.55, img_shape=input_shape, num_classes=2, decode=True)

            # 检测有没有超出检测边缘
            rbbox_img = [0, 0, 1, 1]
            rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
            rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
            # 去重，将重复检测到的目标去掉
            rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=0.01)
            # 将box的坐标重新映射到原图上（上文所有的坐标都进行了归一化，所以要逆操作一次）
            rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)

            frame_origin = draw(frame_origin, rbboxes, rclasses)
            frame_origin = cv2.putText(frame_origin, 'Model: SSD_300', (10, 25),
                                       cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 23, 255), 1)
            frame_origin = cv2.putText(frame_origin, 'Device: GPU', (10, 50),
                                       cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 23, 255), 1)
            frame_origin = cv2.putText(frame_origin, 'Cost: {:2.2f} ms'.format(cost_time),
                                       (10, 75), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 23, 255), 1)
            frame_origin = cv2.putText(frame_origin,
                                       'FPS: {:2.2f}'.format(1 / cost_time) if cost_time > 0 else 'FPS: --',
                                       (10, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 23, 255), 1)
            cv2.imshow("capture", frame_origin)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()
        sess.close()
