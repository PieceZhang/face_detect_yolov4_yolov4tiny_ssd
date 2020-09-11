import tensorflow as tf
from tensorflow.contrib import slim


def pad2d(x, pad):
    return tf.pad(x, paddings=[[0, 0], [pad, pad], [pad, pad], [0, 0]])


class SSDnet(object):
    def __init__(self):
        self.inputshape = [384, 512]
        self.input = tf.placeholder(tf.float32, shape=[None, self.inputshape[0], self.inputshape[1], 3])
        self.num_classes = 1
        self._build()

    def _build(self):
        with tf.variable_scope('ssd_vgg'):
            # block1
            net = slim.conv2d(self.input, 64, 3, 1, scope='conv1_1')
            net = slim.conv2d(net, 64, 3, 1, scope='conv1_2')
            net = slim.max_pool2d(net, 2, 2, scope='pool1')
            # block2
            net = slim.conv2d(net, 128, 3, 1, scope='conv2_1')
            net = slim.conv2d(net, 128, 3, 1, scope='conv2_2')
            net = slim.max_pool2d(net, 2, 2, scope='pool2')
            # block3
            net = slim.conv2d(net, 256, 3, 1, scope='conv3_1')
            net = slim.conv2d(net, 256, 3, 1, scope='conv3_2')
            net = slim.conv2d(net, 256, 3, 1, scope='conv3_3')
            net = slim.max_pool2d(net, 2, 2, scope='pool3')
            # block 4
            net = slim.conv2d(net, 512, 3, 1, scope='conv4_1')
            net = slim.conv2d(net, 512, 3, 1, scope='conv4_2')
            net = slim.conv2d(net, 512, 3, 1, scope='conv4_3')
            net = slim.max_pool2d(net, 2, 2, scope='pool4')
            # block 5
            net = slim.conv2d(net, 512, 3, 1, scope='conv5_1')
            net = slim.conv2d(net, 512, 3, 1, scope='conv5_2')
            net = slim.conv2d(net, 512, 3, 1, scope='conv5_3')
            net = slim.max_pool2d(net, 3, stride=1, scope='pool4')
            print(net)

            # SSD layers
            net = slim.conv2d(net, 1024, 3, dilation_rate=6, scope="conv6")
            self.end_points["block6"] = net
            # net = dropout(net, is_training=self.is_training)
            # block 7
            net = slim.conv2d(net, 1024, 1, scope="conv7")
            self.end_points["block7"] = net
            # block 8
            net = slim.conv2d(net, 256, 1, scope="conv8_1x1")
            net = slim.conv2d(pad2d(net, 1), 512, 3, stride=2, scope="conv8_3x3",
                              padding="valid")
            self.end_points["block8"] = net
            # block 9
            net = slim.conv2d(net, 128, 1, scope="conv9_1x1")
            net = slim.conv2d(pad2d(net, 1), 256, 3, stride=2, scope="conv9_3x3",
                              padding="valid")
            self.end_points["block9"] = net
            # block 10
            net = slim.conv2d(net, 128, 1, scope="conv10_1x1")
            net = slim.conv2d(net, 256, 3, scope="conv10_3x3", padding="valid")
            self.end_points["block10"] = net
            # block 11
            net = slim.conv2d(net, 128, 1, scope="conv11_1x1")
            net = slim.conv2d(net, 256, 3, scope="conv11_3x3", padding="valid")
            self.end_points["block11"] = net


if __name__ == '__main__':
    SSDnet()
