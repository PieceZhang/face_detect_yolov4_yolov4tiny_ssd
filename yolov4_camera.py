import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
from fastnms import fastnms
import colorsys
import random
import time
from decode_np import Decode

if __name__ == '__main__':
    # 验证时的分数阈值和nms_iou阈值
    conf_thresh = 0.50
    nms_thresh = 0.50
    input_shape = (416, 416)
    all_classes = ['face']

    image_origin1 = cv2.imread('image1.jpg')
    assert image_origin1 is not None, 'Image is not found, No such file or directory'
    # image_origin1 = cv2.resize(image_origin1, input_shape, interpolation=cv2.INTER_CUBIC)
    # image_origin1 = cv2.cvtColor(image_origin1, cv2.COLOR_BGR2RGB)
    # plt.imshow(image_origin1)
    # plt.show()
    # image1 = image_origin1.reshape(1, input_shape[0], input_shape[1], 3)

    image_origin2 = cv2.imread('image1.jpg')
    assert image_origin2 is not None, 'Image is not found, No such file or directory'
    # image_origin2 = cv2.resize(image_origin2, input_shape, interpolation=cv2.INTER_CUBIC)
    # image_origin2 = cv2.cvtColor(image_origin2, cv2.COLOR_BGR2RGB)
    # plt.imshow(image_origin2)
    # plt.show()
    # image2 = image_origin2.reshape(1, input_shape[0], input_shape[1], 3)

    with tf.gfile.GFile('yolov4.pb', "rb") as pb:
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

    """
    已实现 11.8fps
    """
    with tf.Session(graph=graph) as sess:
        # warm up
        _ = Decode(conf_thresh, nms_thresh, input_shape, all_classes, graph, iftiny=False)
        tempt = _.detect_image(image_origin1, draw_image=True)
        del tempt

        cam = cv2.VideoCapture(0)
        ifsuccess, frame_origin = cam.read()
        assert ifsuccess is True, 'camera error'
        _decode = Decode(conf_thresh, nms_thresh, input_shape, all_classes, graph, iftiny=False)

        while 1:
            ifsuccess, frame_origin = cam.read()
            time_start = time.time()  # start
            image, boxes, scores, classes = _decode.detect_image(frame_origin, draw_image=True)
            time_stop = time.time()  # stop
            print('fps: ', 1 / (time_stop-time_start))
            cv2.imshow("capture", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()
        sess.close()
