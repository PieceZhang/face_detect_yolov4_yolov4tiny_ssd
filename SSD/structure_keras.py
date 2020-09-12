import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Input
from tensorflow.python.keras.models import Model

def pad2d(x, pad):
    return tf.pad(x, paddings=[[0, 0], [pad, pad], [pad, pad], [0, 0]])

def l2norm(x, scale, trainable=True, scope="L2Normalization"):
    n_channels = x.get_shape().as_list()[-1]
    l2_norm = tf.nn.l2_normalize(x, [3], epsilon=1e-12)
    with tf.variable_scope(scope):
        gamma = tf.get_variable("gamma", shape=[n_channels, ], dtype=tf.float32,
                                initializer=tf.constant_initializer(scale),
                                trainable=trainable)
        return l2_norm * gamma

class SSDnet(object):
    def __init__(self):
        self.inputshape = [384, 512]  # 3:4
        self.input = Input(shape=[self.inputshape[0], self.inputshape[1], 3])
        self.model = None  # from _build
        self.endpoints = {}
        self.num_classes = 1
        self._build()

    def _build(self):
        with tf.variable_scope('ssd_vgg'):
            # block1
            with tf.variable_scope('block1'):
                net = Conv2D(64, 3, 1, activation='relu', padding='same', name='conv1_1')(self.input)
                net = Conv2D(64, 3, 1, activation='relu', padding='same', name='conv1_2')(net)
                net = MaxPool2D(2, 2, name='pool1')(net)
            # block2
            with tf.variable_scope('block2'):
                net = Conv2D(128, 3, 1, activation='relu', padding='same', name='conv2_1')(net)
                net = Conv2D(128, 3, 1, activation='relu', padding='same', name='conv2_2')(net)
                net = MaxPool2D(2, 2, name='pool2')(net)
            # block3
            with tf.variable_scope('block3'):
                net = Conv2D(256, 3, 1, activation='relu', padding='same', name='conv3_1')(net)
                net = Conv2D(256, 3, 1, activation='relu', padding='same', name='conv3_2')(net)
                net = Conv2D(256, 3, 1, activation='relu', padding='same', name='conv3_3')(net)
                net = MaxPool2D(2, 2, name='pool3')(net)
            # block 4
            with tf.variable_scope('block4'):
                net = Conv2D(512, 3, 1, activation='relu', padding='same', name='conv4_1')(net)
                net = Conv2D(512, 3, 1, activation='relu', padding='same', name='conv4_2')(net)
                net = Conv2D(512, 3, 1, activation='relu', padding='same', name='conv4_3')(net)
                net = MaxPool2D(2, 2, name='pool4')(net)
                print(net)
                self.endpoints['conv4_3'] = net
            # block 5
            with tf.variable_scope('block5'):
                net = Conv2D(512, 3, 1, activation='relu', padding='same', name='conv5_1')(net)
                net = Conv2D(512, 3, 1, activation='relu', padding='same', name='conv5_2')(net)
                net = Conv2D(512, 3, 1, activation='relu', padding='same', name='conv5_3')(net)
                net = MaxPool2D(3, 1, name='pool5')(net)

            # SSD layers
            # block 6
            with tf.variable_scope('block6'):
                net = Conv2D(1024, 3, 1, activation='relu', padding='same', dilation_rate=6, name='conv6')(net)
            # block 7
            with tf.variable_scope('block7'):
                net = Conv2D(1024, 1, 1, activation='relu', padding='same', name='conv7')(net)
                print(net)
                self.endpoints["conv7"] = net
            # block 8
            with tf.variable_scope('block8'):
                net = Conv2D(256, 1, 1, activation='relu', padding='same', name="conv8_1")(net)
                net = Conv2D(512, 3, 2, activation='relu', padding='valid', name="conv8_2")(pad2d(net, 1))
                print(net)
                self.endpoints["conv8_2"] = net
            # block 9
            with tf.variable_scope('block9'):
                net = Conv2D(128, 1, 1, activation='relu', padding='same', name="conv9_1")(net)
                net = Conv2D(256, 3, 2, activation='relu', padding='valid', name="conv9_2")(pad2d(net, 1))
                print(net)
                self.endpoints["conv9_2"] = net
            # block 10
            with tf.variable_scope('block10'):
                net = Conv2D(128, 1, 1, activation='relu', padding='same', name="conv10_1")(net)
                net = Conv2D(256, 3, 2, activation='relu', padding='valid', name="conv10_2")(pad2d(net, 1))
                print(net)
                self.endpoints["conv10_2"] = net
            # block 11
            with tf.variable_scope('block11'):
                net = Conv2D(128, 1, 1, activation='relu', padding='same', name="conv11_1")(net)
                net = Conv2D(256, 3, 2, activation='relu', padding='valid', name="conv11_2")(net)
                print(net)
                self.endpoints["conv11_2"] = net


if __name__ == '__main__':
    SSDnet()
