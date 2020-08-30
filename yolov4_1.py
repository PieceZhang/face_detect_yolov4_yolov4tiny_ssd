import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
from fastnms import fastnms
import colorsys
import random
import time


def draw(image, boxes, scores, classes, all_classes, colors):
    image_h = image_w = 416
    for box, score, cl in zip(boxes, scores, classes):
        x0, y0, x1, y1 = box
        left = max(0, np.floor(x0 + 0.5).astype(int))
        top = max(0, np.floor(y0 + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x1 + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y1 + 0.5).astype(int))
        bbox_color = colors[cl]
        # bbox_thick = 1 if min(image_h, image_w) < 400 else 2
        bbox_thick = 1
        cv2.rectangle(image, (left, top), (right, bottom), bbox_color, bbox_thick)
        bbox_mess = '%.2f' % score
        t_size = cv2.getTextSize(bbox_mess, 0, 0.5, thickness=1)[0]
        cv2.rectangle(image, (left, top), (left + t_size[0], top - t_size[1] - 3), bbox_color, -1)
        cv2.putText(image, bbox_mess, (left, top - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    plt.imshow(image)
    plt.show()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('result.jpg', image)



def decode(conv_output, anchors, stride, num_class):

    anchor_per_scale = len(anchors)
    conv_shape = tf.shape(conv_output)
    batch_size = int(sess.run(conv_shape[0], feed_dict1))
    output_size = int((sess.run(conv_shape[1], feed_dict1) / anchor_per_scale) ** 0.5)

    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, anchor_per_scale, 5 + num_class))
    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5:]
    y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
    x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])
    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)
    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
    pred_wh = (tf.exp(conv_raw_dwdh) * anchors)  # exp
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)
    pred_xywh = tf.reshape(pred_xywh, (batch_size, -1, 4))  # [-1, -1, 4]
    pred_conf = tf.reshape(pred_conf, (batch_size, -1, 1))  # [-1, -1, 1]
    pred_prob = tf.reshape(pred_prob, (batch_size, -1, num_class))  # [-1, -1, 80]

    pred_xywh = sess.run(pred_xywh, feed_dict1)
    pred_conf = sess.run(pred_conf, feed_dict1)
    pred_prob = sess.run(pred_prob, feed_dict1)


    return pred_xywh, pred_conf, pred_prob


if __name__ == '__main__':
    # 验证时的分数阈值和nms_iou阈值
    conf_thresh = 0.40
    nms_thresh = 0.60
    keep_top_k = 50
    nms_top_k = 50
    input_shape = (416, 416)
    all_classes = ['face']
    anchors = np.array([
        [[12, 16], [19, 36], [40, 28]],
        [[36, 75], [76, 55], [72, 146]],
        [[142, 110], [192, 243], [459, 401]]
    ])
    dellist = []
    timelist = []

    # 定义颜色
    hsv_tuples = [(1.0 * x / 1, 1., 1.) for x in range(1)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    image_origin1 = cv2.imread('image1.jpg')
    assert image_origin1 is not None, 'Image is not found, No such file or directory'
    image_origin1 = cv2.resize(image_origin1, input_shape, interpolation=cv2.INTER_CUBIC)
    image_origin1 = cv2.cvtColor(image_origin1, cv2.COLOR_BGR2RGB)
    # plt.imshow(image_origin1)
    # plt.show()
    image1 = image_origin1.reshape(1, input_shape[0], input_shape[1], 3)

    image_origin2 = cv2.imread('image2.jpg')
    assert image_origin2 is not None, 'Image is not found, No such file or directory'
    image_origin2 = cv2.resize(image_origin2, input_shape, interpolation=cv2.INTER_CUBIC)
    image_origin2 = cv2.cvtColor(image_origin2, cv2.COLOR_BGR2RGB)
    # plt.imshow(image_origin2)
    # plt.show()
    image2 = image_origin2.reshape(1, input_shape[0], input_shape[1], 3)

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

    node_in = graph.get_tensor_by_name('inputs:0')  # shape=(?, 416, 416, 3)
    output_s = graph.get_tensor_by_name('detector/yolo-v4/Reshape_8:0')  # small: (?,507,6) ->13*13*3=507
    output_m = graph.get_tensor_by_name('detector/yolo-v4/Reshape_4:0')  # medium: (?,2028,6) ->26*26*3=2028
    output_l = graph.get_tensor_by_name('detector/yolo-v4/Reshape:0')  # large: (?,8112,6) ->52*52*3=8112

    """
    参考github例程，同时使用大中小三个尺度feature map，0.12fps
    """
    with tf.Session(graph=graph) as sess:
        feed_dict1 = {node_in: image1}
        feed_dict2 = {node_in: image2}
        # warm up
        sess.run([output_s, output_m, output_l], feed_dict1)

        timelist.append(time.time())

        output_s, output_m, output_l = sess.run([output_s, output_m, output_l], feed_dict2)

        timelist.append(time.time()), print('1',timelist[-1] - timelist[-2])  # time 1

        pred_xywh_s, pred_conf_s, pred_prob_s = decode(output_s, anchors[0], 8, 1)
        pred_xywh_m, pred_conf_m, pred_prob_m = decode(output_m, anchors[1], 16, 1)
        pred_xywh_l, pred_conf_l, pred_prob_l = decode(output_l, anchors[2], 32, 1)
        # print(pred_xywh_s, pred_conf_m, pred_prob_l)

        pred_score_s = pred_conf_s * pred_prob_s
        pred_score_m = pred_conf_m * pred_prob_m
        pred_score_l = pred_conf_l * pred_prob_l

        timelist.append(time.time()), print('2',timelist[-1] - timelist[-2])  # time 2

        # all_pred_boxes = tf.concat([pred_xywh_s, pred_xywh_m, pred_xywh_l], axis=1)  # [batch_size, -1, 4]
        # all_pred_scores = tf.concat([pred_score_s, pred_score_m, pred_score_l], axis=1)  # [batch_size, -1, 80]
        all_pred_boxes = np.concatenate((pred_xywh_s, pred_xywh_m, pred_xywh_l), axis=1)
        all_pred_scores = np.concatenate((pred_score_s, pred_score_m, pred_score_l), axis=1)

        timelist.append(time.time()), print('3',timelist[-1] - timelist[-2])  # time 3 tf: 0.004s, np: 0.0009s

        output = fastnms(all_pred_boxes, all_pred_scores, conf_thresh, nms_thresh, keep_top_k, nms_top_k)

        timelist.append(time.time()), print('4',timelist[-1] - timelist[-2])  # time 4

        output = sess.run(output, feed_dict2)
        boxes, scores, classes = output[0][0], output[1][0], output[2][0]
        print(f"boxes: {boxes} \nscores: {scores}\nclasses: {classes}")

        timelist.append(time.time()), print('5',timelist[-1] - timelist[-2])  # time 5

        # (0, 0)在左上角
        for idx1, value1 in enumerate(boxes):
            if value1[2] - value1[0] < 20 or value1[3] - value1[1] < 20:
                dellist.append(idx1)
                continue
            for value2 in value1:
                if value2 > input_shape[0]+30 or value2 < -30:
                    dellist.append(idx1)
                    break
        boxes = np.delete(boxes, dellist, 0)
        scores = np.delete(scores, dellist)
        classes = np.delete(classes, dellist)

        # stop timing
        timelist.append(time.time()), print('6',timelist[-1] - timelist[-2])  # time 6
        print('fps: ',1 / (timelist[-1] - timelist[1]))

        sess.close()

    if boxes is not None:
        draw(image_origin2, boxes, scores, classes, all_classes, colors)

