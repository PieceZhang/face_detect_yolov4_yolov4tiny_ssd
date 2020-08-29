import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
from fastnms import fastnms
import colorsys
import random
import time


def draw(image, boxes, scores, classes, all_classes, colors):
    # image_h = image_w = 416
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
        # plt.imshow(image1)
        # plt.show()
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
    conf_thresh = 0.70
    nms_thresh = 0.30
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

    # load model
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

    node_in = graph.get_tensor_by_name('inputs:0')  # shape=(?, 416, 416, 3)
    output = graph.get_tensor_by_name('output_boxes:0')  # shape=(?, 10647, 6)

    """
    直接使用outputboxes（未实现）
    """
    with tf.Session(graph=graph) as sess:
        feed_dict1 = {node_in: image1}
        feed_dict2 = {node_in: image2}

        # warm up
        _ = sess.run(output, feed_dict1)

        # TODO output_boxes输出张量大小与之前相同，但排列顺序不同（507+2028+8112=10647），需要重新排序
        '''
        output_boxes (<tf.Tensor 'output_boxes:0' shape=(?, 10647, 6) dtype=float32>,)
        all_pred_boxes: (?, 10647, 4)
        conf * prob -> all_pred_scores: (?, 10647, 1)
        '''

        # start timing...
        time_start = time.time()
        output = tf.reverse(sess.run(output, feed_dict2), [1])
        # output = tf.reverse(output, [1])  # TODO 由直接run得到数值改为操作tensor构建网络，最后再run（feed_dict2)得到结果

        all_pred_boxes, pred_conf, pred_prob = tf.split(output, [4, 1, 1], 2)
        all_pred_scores = pred_conf * pred_prob

        output = fastnms(all_pred_boxes, all_pred_scores, conf_thresh, nms_thresh, keep_top_k, nms_top_k)
        output = sess.run(output, feed_dict2)
        boxes, scores, classes = output[0][0], output[1][0], output[2][0]

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
        # print(f"boxes: {boxes} \nscores: {scores}\nclasses: {classes}")

        # stop timing
        time_end = time.time()
        print('time: ',time_end - time_start)
        print('fps: ',1 / (time_end - time_start))

        sess.close()

    if boxes is not None:
        draw(image_origin2, boxes, scores, classes, all_classes, colors)
