import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, Lambda, ZeroPadding2D, Reshape
from tensorflow.python.keras.backend import variable, constant, l2_normalize

def pad2d(x, pad):
    return ZeroPadding2D(padding=[[pad, pad], [pad, pad]])(x)
    # return tf.pad(x, paddings=[[0, 0], [pad, pad], [pad, pad], [0, 0]])

def _tf_l2_normalize(x):
    return tf.nn.l2_normalize(x, [3])

def _l2norm(x):
    scale = 20
    scope = "L2Normalization"
    n_channels = x.get_shape().as_list()[-1]
    l2_norm = Lambda(_tf_l2_normalize)(x)
    with tf.variable_scope(scope):
        gamma = variable(value=constant(scale, dtype=tf.float32, shape=[n_channels, ]),
                         dtype=tf.float32, name='gamma')
        return l2_norm * gamma

def ssd_multibox_layer(x, num_classes, sizes, ratios, normalization=-1, name="multibox"):
    pre_shape = x.get_shape().as_list()[1:-1]
    # pre_shape = [-1] + pre_shape
    with tf.variable_scope(name):
        # l2 norm
        if normalization > 0:
            x = Lambda(_l2norm)(x)
        print(x)

        # numbers of anchors
        n_anchors = len(sizes) + len(ratios)

        # location predictions
        loc_pred = Conv2D(n_anchors, 1 if name=='conv11_2_box' else 3,
                          activation=None, name=name+'loc')(x)  # n_anchors*4
        # loc_pred = tf.reshape(loc_pred, pre_shape + [n_anchors, 4])
        loc_pred = Reshape(pre_shape + [n_anchors, 4])(loc_pred)

        # class prediction
        cls_pred = Conv2D(n_anchors*num_classes, 1 if name=='conv11_2_box' else 3,
                          activation=None, name=name+'cls')(x)
        # cls_pred = tf.reshape(cls_pred, pre_shape + [n_anchors, num_classes])
        cls_pred = Reshape(pre_shape + [n_anchors, num_classes])(cls_pred)

        return cls_pred, loc_pred
