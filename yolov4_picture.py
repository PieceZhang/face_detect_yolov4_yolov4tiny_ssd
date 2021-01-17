import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
from decode_np import Decode


if __name__ == '__main__':
    # 验证时的分数阈值和nms_iou阈值
    conf_thresh = 0.60
    nms_thresh = 0.20
    input_shape = (416, 416)
    all_classes = ['face']
    timelist = []

    image_origin1 = cv2.imread('image1.jpg')
    assert image_origin1 is not None, 'Image is not found, No such file or directory'
    # image_origin1 = cv2.resize(image_origin1, input_shape, interpolation=cv2.INTER_CUBIC)
    # image_origin1 = cv2.cvtColor(image_origin1, cv2.COLOR_BGR2RGB)
    # plt.imshow(image_origin1)
    # plt.show()
    # image1 = image_origin1.reshape(1, input_shape[0], input_shape[1], 3)

    image_origin2 = cv2.imread('image2.jpg')  # TODO v4无法识别image1
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
    # for op in graph.get_operations():
    #     print(op.name, op.values())

    """
    5.3fps
    """
    with tf.Session(graph=graph) as sess:
        # warm up
        _ = Decode(conf_thresh, nms_thresh, input_shape, all_classes, graph, iftiny=False)
        tempt = _.detect_image(image_origin1, draw_image=True)
        del tempt

        _decode = Decode(conf_thresh, nms_thresh, input_shape, all_classes, graph, iftiny=False)
        time_start = time.time()  # start
        image, boxes, scores, classes = _decode.detect_image(image_origin2, draw_image=True)
        time_stop = time.time()  # stop

        cost_time = time_stop - time_start
        print('fps: ', 1 / cost_time)

        image = cv2.putText(image, 'Model: YOLOv4', (10, 25),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 23, 255), 1)
        image = cv2.putText(image, 'Device: GPU', (10, 50),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 23, 255), 1)
        image = cv2.putText(image, 'Cost: {:2.2f} ms'.format(cost_time),
                            (10, 75), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 23, 255), 1)
        image = cv2.putText(image,
                            'FPS: {:2.2f}'.format(1 / cost_time) if cost_time > 0 else 'FPS: --',
                            (10, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 23, 255), 1)

        cv2.imshow("image", image)
        cv2.waitKey()
        sess.close()

