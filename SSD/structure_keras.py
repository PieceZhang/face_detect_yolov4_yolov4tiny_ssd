import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Input
import numpy as np
from tensorflow.python.keras.models import Model
from SSD.anchor import ssd_anchors_all_layers
from SSD.layer_diy import pad2d, ssd_multibox_layer
from SSD.loss import ssd_losses

class SSDnet(object):
    def __init__(self):
        self.inputshape = [300, 300]
        self.input = Input(shape=[self.inputshape[0], self.inputshape[1], 3])
        self.endpoints = {}
        self.num_classes = 1  # TODO 类别数+背景 ?
        self.no_annotation_label = 1
        self.feat_layers = ["conv4_3", "conv7", "conv8_2", "conv9_2", "conv10_2", "conv11_2"]  # 要进行检测的特征图name
        self.feat_shapes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]  # 特征图大小
        self.anchor_size_bounds = [0.15, 0.90]  # 特征图尺度范围
        self.anchor_sizes = [(21., 45.),
                             (45., 99.),
                             (99., 153.),
                             (153., 207.),
                             (207., 261.),
                             (261., 315.)]  # 不同特征图的先验框尺度（第一个值是s_k，第2个值是s_k+1）
        self.anchor_ratios = [[2, .5],
                              [2, .5, 3, 1. / 3],
                              [2, .5, 3, 1. / 3],
                              [2, .5, 3, 1. / 3],
                              [2, .5],
                              [2, .5]]  # 特征图先验框所采用的长宽比（每个特征图都有2个正方形先验框）
        self.anchor_steps = [8, 16, 32, 64, 100, 300]  # 特征图的单元大小
        self.anchor_offset = 0.5  # 偏移值，确定先验框中心
        self.normalizations = [20, -1, -1, -1, -1, -1]  # l2 norm
        self.prior_scaling = [0.1, 0.1, 0.2, 0.2]  # variance
        # build model
        self.model = self._build()

    def _build(self):
        with tf.variable_scope('ssd_vgg'):
            # block1
            with tf.variable_scope('block1'):
                net = Conv2D(64, 3, 1, activation='relu', padding='same', name='conv1_1')(self.input)
                net = Conv2D(64, 3, 1, activation='relu', padding='same', name='conv1_2')(net)
                net = MaxPool2D(2, 2, padding='same', name='pool1')(net)
            # block2
            with tf.variable_scope('block2'):
                net = Conv2D(128, 3, 1, activation='relu', padding='same', name='conv2_1')(net)
                net = Conv2D(128, 3, 1, activation='relu', padding='same', name='conv2_2')(net)
                net = MaxPool2D(2, 2, padding='same', name='pool2')(net)
            # block3
            with tf.variable_scope('block3'):
                net = Conv2D(256, 3, 1, activation='relu', padding='same', name='conv3_1')(net)
                net = Conv2D(256, 3, 1, activation='relu', padding='same', name='conv3_2')(net)
                net = Conv2D(256, 3, 1, activation='relu', padding='same', name='conv3_3')(net)
                net = MaxPool2D(2, 2, padding='same', name='pool3')(net)
            # block 4
            with tf.variable_scope('block4'):
                net = Conv2D(512, 3, 1, activation='relu', padding='same', name='conv4_1')(net)
                net = Conv2D(512, 3, 1, activation='relu', padding='same', name='conv4_2')(net)
                net = Conv2D(512, 3, 1, activation='relu', padding='same', name='conv4_3')(net)
                self.endpoints['conv4_3'] = net
                net = MaxPool2D(2, 2, padding='same', name='pool4')(net)
            # block 5
            with tf.variable_scope('block5'):
                net = Conv2D(512, 3, 1, activation='relu', padding='same', name='conv5_1')(net)
                net = Conv2D(512, 3, 1, activation='relu', padding='same', name='conv5_2')(net)
                net = Conv2D(512, 3, 1, activation='relu', padding='same', name='conv5_3')(net)
                net = MaxPool2D(3, 1, padding='same', name='pool5')(net)

            # SSD layers
            # block 6
            with tf.variable_scope('block6'):
                net = Conv2D(1024, 3, 1, activation='relu', padding='same', dilation_rate=6, name='conv6')(net)
            # block 7
            with tf.variable_scope('block7'):
                net = Conv2D(1024, 1, 1, activation='relu', padding='same', name='conv7')(net)
                self.endpoints["conv7"] = net
            # block 8
            with tf.variable_scope('block8'):
                net = Conv2D(256, 1, 1, activation='relu', padding='same', name="conv8_1")(net)
                net = Conv2D(512, 3, 2, activation='relu', padding='valid', name="conv8_2")(pad2d(net, 1))
                self.endpoints["conv8_2"] = net
            # block 9
            with tf.variable_scope('block9'):
                net = Conv2D(128, 1, 1, activation='relu', padding='same', name="conv9_1")(net)
                net = Conv2D(256, 3, 2, activation='relu', padding='valid', name="conv9_2")(pad2d(net, 1))
                self.endpoints["conv9_2"] = net
            # block 10
            with tf.variable_scope('block10'):
                net = Conv2D(128, 1, 1, activation='relu', padding='same', name="conv10_1")(net)
                net = Conv2D(256, 3, 1, activation='relu', padding='valid', name="conv10_2")(net)
                self.endpoints["conv10_2"] = net
            # block 11
            with tf.variable_scope('block11'):
                net = Conv2D(128, 1, 1, activation='relu', padding='same', name="conv11_1")(net)
                net = Conv2D(256, 3, 1, activation='relu', padding='valid', name="conv11_2")(net)
                self.endpoints["conv11_2"] = net

            predictions = []
            classes = []  # logits
            locations = []
            for i, layer in enumerate(self.feat_layers):
                cls, loc = ssd_multibox_layer(self.endpoints[layer], self.num_classes,
                                              self.anchor_sizes[i],
                                              self.anchor_ratios[i],
                                              self.normalizations[i], name=layer+"_box")
                predictions.append(tf.nn.softmax(cls))
                classes.append(cls)
                locations.append(loc)
            model = Model(inputs=self.input, outputs=classes + locations)
            model.summary()
            print('predictions:\n', predictions)
            print('classes:\n', classes)
            print('locations:\n', locations)
            # model.compile()
            return model

    def anchors(self):
        """Get sSD anchors"""
        return ssd_anchors_all_layers(self.inputshape,
                                      self.feat_shapes,
                                      self.anchor_sizes,
                                      self.anchor_ratios,
                                      self.anchor_steps,
                                      self.anchor_offset,
                                      np.float32)


if __name__ == '__main__':
    SSD = SSDnet()
    anchor = SSD.anchors()
    print()
