from functools import wraps

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers import Lambda, DepthwiseConv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

from .utils import compose, iou


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def DarknetConv2D_BN(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization())


@wraps(DepthwiseConv2D)
def DarknetDWConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for DepthWise Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return DepthwiseConv2D(*args, **darknet_conv_kwargs)


def DarknetDWConv2D_BN(*args, **kwargs):
    """Darknet DepthWise Convolution2D followed by BatchNormalization."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetDWConv2D(*args, **no_bias_kwargs),
        BatchNormalization())


def channel_split(x, name):
    """Split the number of channel into half."""
    channel = x.shape.as_list()[-1]
    sc = channel // 2
    x_left = Lambda(lambda c: c[..., 0:sc], name='{}/split_{}'.format(name, 1))(x)
    x_right = Lambda(lambda c: c[..., sc:], name='{}/split_{}'.format(name, 2))(x)
    return x_left, x_right


def group_block(x, num_filters, block):
    """A group block starting with a Convolution2d and followed by channel split Convolution2D"""
    name = 'group_{}'.format(block)
    x = DarknetConv2D_BN_Leaky(num_filters // 4, (1, 1))(x)
    x_l, x_r = channel_split(x, name)
    filters = num_filters - (num_filters // 4)
    x_l = DarknetConv2D(filters // 2, (3, 3))(x_l)
    x_r = DarknetConv2D(filters // 2, (3, 3))(x_r)
    x = Concatenate()([x_l, x_r, x])
    # x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    return x


def resblock_body(x, num_filters, num_blocks):
    """A series of resblocks starting with a downsampling Convolution2D"""
    # Darknet uses top and left padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Leaky(num_filters // 2, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters, (3, 3)))(x)
        x = Add()([x, y])
    return x


def darknet_body(x):
    """Darknent body having 52 Convolution2D layers"""
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)             # layers[1] to layers[3]
    x = resblock_body(x, num_filters=64, num_blocks=1)    # layers[4] to layers[14]
    x = resblock_body(x, num_filters=128, num_blocks=2)   # layers[15] to layers[32]
    x = resblock_body(x, num_filters=256, num_blocks=8)   # layers[33] to layers[92]
    x = resblock_body(x, num_filters=512, num_blocks=8)   # layers[93] to layers[152]
    x = resblock_body(x, num_filters=1024, num_blocks=4)  # layers[153] to layers[184]
    return x


def make_last_layers(x, num_filters, out_filters):
    """6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer"""
    x = compose(
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)
    y = compose(
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D(out_filters, (1, 1)))(x)
    return x, y


def yolo_body(inputs, num_anchors, num_classes):
    """Create YOLOv3 model CNN body in Keras."""
    darknet = Model(inputs, darknet_body(inputs))
    x, y1 = make_last_layers(darknet.output, 512, num_anchors * (5 + num_classes))

    x = compose(
        DarknetConv2D_BN_Leaky(256, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors * (5 + num_classes))

    x = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[92].output])
    _, y3 = make_last_layers(x, 128, num_anchors * (5 + num_classes))

    return Model(inputs, [y1, y2, y3])  # layers[0] to layers[251]


def make_last_layers_lite(x, num_filters, out_filters):
    """6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer"""
    x = compose(
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetDWConv2D(kernel_size=(3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetDWConv2D(kernel_size=(3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)

    y = compose(
        DarknetDWConv2D(kernel_size=(3, 3)),
        DarknetConv2D(out_filters, (1, 1)))(x)
    return x, y


def lite_yolo_body(inputs, num_anchors, num_classes):
    """Create YOLOv3-lite model CNN body in Keras."""
    darknet = Model(inputs, darknet_body(inputs))
    x, y1 = make_last_layers_lite(darknet.output, 256, num_anchors * (5 + num_classes))

    x = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[152].output])
    x, y2 = make_last_layers_lite(x, 128, num_anchors * (5 + num_classes))

    x = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[92].output])
    _, y3 = make_last_layers_lite(x, 64, num_anchors * (5 + num_classes))

    return Model(inputs, [y1, y2, y3])


def tiny_yolo_body(inputs, num_anchors, num_classes):
    """Create YOLOv3-tiny model CNN body in Keras."""
    x1 = compose(
        DarknetConv2D_BN_Leaky(16, (3, 3)),                              # layers[1] to layers[3]
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),  # layers[4]
        DarknetConv2D_BN_Leaky(32, (3, 3)),                              # layers[5] to layers[7]
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),  # layers[8]
        DarknetConv2D_BN_Leaky(64, (3, 3)),                              # layers[9] to layers[11]
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),  # layers[12]
        DarknetConv2D_BN_Leaky(128, (3, 3)),                             # layers[13] to layers[15]
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),  # layers[16]
        DarknetConv2D_BN_Leaky(256, (3, 3)))(inputs)                     # layers[17] to layers[19]
    x2 = compose(
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),  # layers[20]
        DarknetConv2D_BN_Leaky(512, (3, 3)),                             # layers[21] to layers[23]
        MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'),  # layers[24]
        DarknetConv2D_BN_Leaky(1024, (3, 3)),                            # layers[25] to layers[27]
        DarknetConv2D_BN_Leaky(256, (1, 1)))(x1)                         # layers[28] to layers[30]
    y1 = compose(
        DarknetConv2D_BN_Leaky(512, (3, 3)),
        DarknetConv2D(num_anchors * (5 + num_classes), (1, 1)))(x2)

    x2 = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),                             # layers[31] to layers[33]
        UpSampling2D(2))(x2)                                             # layers[34]
    y2 = compose(
        Concatenate(),                                                   # layers[35]
        DarknetConv2D_BN_Leaky(256, (3, 3)),
        DarknetConv2D(num_anchors * (5 + num_classes), (1, 1)))([x2, x1])

    return Model(inputs, [y1, y2])  # layers[0] to layers[43]


def mini_yolo_body(inputs, num_anchors, num_classes):
    """Create YOLOv3-mini model CNN body in Keras."""
    x = ZeroPadding2D(((1, 0), (1, 0)))(inputs)
    x1 = DarknetConv2D_BN_Leaky(16, (3, 3), strides=(2, 2))(x)
    # x1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x1 = group_block(x1, num_filters=32, block=1)
    x1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x1)
    x1 = group_block(x1, num_filters=64, block=2)
    x1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x1)
    x1 = group_block(x1, num_filters=128, block=3)
    x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x1)
    x2 = group_block(x2, num_filters=256, block=4)

    x3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x2)
    x3 = group_block(x3, num_filters=512, block=5)
    x3 = compose(
        # MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
        DarknetConv2D_BN_Leaky(256, (1, 1)),
        DarknetDWConv2D(kernel_size=(3, 3)),
        DarknetConv2D(512, (1, 1)),
        DarknetConv2D_BN_Leaky(256, (1, 1)))(x3)

    y1 = compose(
        DarknetDWConv2D(kernel_size=(3, 3)),
        DarknetConv2D(256, (1, 1)),
        DarknetConv2D(num_anchors * (5 + num_classes), (1, 1)))(x3)

    x3 = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x3)

    x1 = compose(
        ZeroPadding2D(((1, 0), (1, 0))),
        DarknetConv2D(32, (1, 1)),
        DarknetDWConv2D(kernel_size=(3, 3), strides=(2, 2)))(x1)

    y2 = compose(
        Concatenate(),
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        DarknetDWConv2D(kernel_size=(3, 3)),
        DarknetConv2D(128, (1, 1)),
        DarknetConv2D(num_anchors * (5 + num_classes), (1, 1)))([x3, x2, x1])

    return Model(inputs, [y1, y2])


def fast_yolo_body(inputs, num_anchors, num_classes):
    """Create YOLOv3-fast model CNN body in Keras."""
    x = ZeroPadding2D(((1, 0), (1, 0)))(inputs)
    x1 = DarknetConv2D_BN_Leaky(16, (3, 3), strides=(2, 2))(x)
    x1 = group_block(x1, num_filters=32, block=1)
    x1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x1)
    x1 = group_block(x1, num_filters=64, block=2)
    x1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x1)
    x1 = group_block(x1, num_filters=128, block=3)
    x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x1)
    x2 = group_block(x2, num_filters=256, block=4)

    x3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x2)
    x3 = group_block(x3, num_filters=512, block=5)
    x3 = compose(
        DarknetConv2D_BN_Leaky(256, (1, 1)),
        DarknetDWConv2D(kernel_size=(3, 3)),
        DarknetConv2D(512, (1, 1)),
        DarknetConv2D_BN_Leaky(256, (1, 1)))(x3)

    y1 = compose(
        DarknetDWConv2D(kernel_size=(3, 3)),
        DarknetConv2D(256, (1, 1)),
        DarknetConv2D(num_anchors * (5 + num_classes), (1, 1)))(x3)

    x3 = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x3)

    x2 = compose(
        Concatenate(),
        DarknetConv2D_BN_Leaky(128, (1, 1)))([x3, x2])

    y2 = compose(
        DarknetDWConv2D(kernel_size=(3, 3)),
        DarknetConv2D(128, (1, 1)),
        DarknetConv2D(num_anchors * (5 + num_classes), (1, 1)))(x2)

    x2 = compose(
        DarknetConv2D_BN_Leaky(64, (1, 1)),
        UpSampling2D(2))(x2)

    y3 = compose(
        Concatenate(),
        DarknetConv2D_BN_Leaky(64, (1, 1)),
        DarknetDWConv2D(kernel_size=(3, 3)),
        DarknetConv2D(64, (1, 1)),
        DarknetConv2D(num_anchors * (5 + num_classes), (1, 1)))([x2, x1])

    return Model(inputs, [y1, y2, y3])


def process_yolo_outputs(feats, anchors, num_classes, input_shape, calc_loss=False):
    """
    Process YOLO predictions to training input format.
    :param feats: tensor, shape=(batch_size, gird_h, grid_w, anchors_per_scale * (5 + num_classes))
            where 5=[d_cx d_cy d_w d_h confidence_score]
    :param anchors: ndarray, shape=(anchors_per_scale, 2) where 2=[width height]
    :param num_classes: int, classes numbers
    :param input_shape: tensor, (height, width)
    :param calc_loss: bool, calculate loss or not
    :return: tuple of tensors
    """
    num_anchors = len(anchors)
    # Reshape the format to the same as y_true.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    # Create spatial grid.
    grid_shape = K.shape(feats)[1:3]  # grid_h, grid_w
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])  # shape=(grid_h, grid_w, 1, 2)
    grid = K.cast(grid, K.dtype(feats))

    # Reshape to batch_size, grid_h, grid_w, num_anchors, box_params
    # where box_params = d_cx, d_cy, d_w, d_h, confidence score, class probabilities.
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, 5 + num_classes])

    # Adjust predictions to each spatial grid point and anchor size.
    # Normalize center_x, center_y, width, height, 0 to 1.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    """Get original corrected boxes."""
    box_yx = box_xy[..., ::-1]  # 0 to 1, reverse slice for NMS
    box_hw = box_wh[..., ::-1]  # 0 to 1, reverse slice for NMS
    input_shape = K.cast(input_shape, K.dtype(box_yx))  # (h, w)
    image_shape = K.cast(image_shape, K.dtype(box_yx))  # (h, w)
    new_shape = K.round(image_shape * K.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale
    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        box_mins[..., 0:1],   # y_min
        box_mins[..., 1:2],   # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]   # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    """Process Conv layer output."""
    box_xy, box_wh, box_confidence, box_class_probs = process_yolo_outputs(feats, anchors,
                                                                           num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_eval(yolo_outputs, anchors, num_classes, image_shape,
              max_boxes=20, score_threshold=.6, iou_threshold=.5):
    """
    Evaluate YOLO model on given input and return filtered boxes.
    :param yolo_outputs: list of tensors,
            [(batch_size, grid_h, gird_w, anchors_per_scale * (5 + num_classes)), ...]
            where 5=[d_cx d_cy d_w d_h confidence_score]
    :param anchors: ndarray, shape=(clusters, 2) where 2=[width height]
    :param num_classes: int, classes numbers
    :param image_shape: tensor, (original_image_height, original_image_width)
    :param max_boxes: int, maximum number of class boxes to be selected by NMS
    :param score_threshold: float, class_confidence_score threshold of box
    :param iou_threshold: float, IoU threshold of NMS
    :return: tuple of tensors
    """
    num_out_layers = len(yolo_outputs)
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_out_layers == 3 else [[3, 4, 5], [0, 1, 2]]
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(num_out_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l], anchors[anchor_mask[l]],
                                                    num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)            # shape = (batch_total, 4)
    box_scores = K.concatenate(box_scores, axis=0)  # shape = (batch_total, num_classes)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])                  # shape=(class_num_boxes, 4), rank=2
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])  # shape=(class_num_boxes, ), rank=1
        nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores,
                                                 max_boxes_tensor, iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)      # rank=2
    scores_ = K.concatenate(scores_, axis=0)    # rank=1
    classes_ = K.concatenate(classes_, axis=0)  # rank=1

    return boxes_, scores_, classes_


def preprocess_true_boxes(true_boxes, input_shape, anchors, anchors_per_scale, num_classes):
    """
    Pre-process true boxes to training input format.
    :param true_boxes: ndarray, shape=(batch_size, max_boxes, 5)
            where 5=[rx_min ry_min rx_max ry_max class_id]
    :param input_shape: tuple, (height, width)
    :param anchors: ndarray, shape=(clusters, 2) where 2=[width height]
    :param anchors_per_scale: int, divide up the clusters across scales
    :param num_classes: int, classes numbers
    :return: list of ndarrays, [y_true1, y_true2, ...], all value 0 to 1
            ndarray shape=(batch_size, grid_h, grid_w, anchors_per_scale, 5 + num_classes)
            where 5=[n_cx n_cy n_w n_h confidence_score]
    """
    assert (true_boxes[..., 4] < num_classes).all(), 'Class id must be less than num_classes.'
    num_out_layers = len(anchors) // anchors_per_scale
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_out_layers == 3 else [[3, 4, 5], [0, 1, 2]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')

    # Replace with center_x, center_y, width, height.
    # Normalize center_x, center_y, width, height, 0 to 1.
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    batch_size = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_out_layers)]
    y_true = [np.zeros(shape=(batch_size, grid_shapes[l][0], grid_shapes[l][1],
                              len(anchor_mask[l]), 5 + num_classes),
                       dtype='float32') for l in range(num_out_layers)]

    # Label True for the width > 0 and False for the width <= 0.
    valid_mask = boxes_wh[..., 0] > 0  # shape=(batch_size, image_max_boxes)

    for b in range(batch_size):
        # Discard zero rows.
        valid_boxes_wh = boxes_wh[b, valid_mask[b]]  # shape=(image_valid_boxes, 2)
        if len(valid_boxes_wh) == 0: continue
        iou_ = iou(valid_boxes_wh, anchors)          # shape=(image_valid_boxes, clusters)

        # Find best anchor for each true box.
        best_anchor = np.argmax(iou_, axis=-1)       # shape=(image_valid_boxes, ) about index

        # Label image true boxes at right position of y_true.
        for v, n in enumerate(best_anchor):
            for l in range(num_out_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b, v, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b, v, 1] * grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b, v, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, v, 0:4]  # n_cx, n_cy, n_w, n_h
                    y_true[l][b, j, i, k, 4] = 1                        # confidence score
                    y_true[l][b, j, i, k, 5 + c] = 1                    # class probability

    return y_true


def box_iou(b1, b2):
    """
    Calculates the IoU between predicted boxes and true boxes of a image,
    need to consider the boxes position in the feature map.
    :param b1: tensor, shape=(i1,...,iN, 4) where 4=[n_cx n_cy n_w n_h]
    :param b2: tensor, shape=(j, 4) where 4=[n_cx n_cy n_w n_h]
    :return: tensor, shape=(i1,...,iN, j)
    """
    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)  # shape=(i1, ..., iN, 1, 4)
    b1_xy = b1[..., :2]  # position
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)  # shape=(1, j, 4)
    b2_xy = b2[..., :2]  # position
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    # intersect=positive or non-intersect=0
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou_ = intersect_area / (b1_area + b2_area - intersect_area)

    return iou_


def yolo_loss(args, anchors, anchors_per_scale, num_classes, ignore_thresh=.5, print_loss=False):
    """
    Calculates YOLO Loss.
    :param args: list of tensors, [yolo_outputs + y_true]
            yolo_outputs: [(batch_size, gird_h, grid_w, anchors_per_scale * (5 + num_classes)), ...]
                    where 5=[d_cx d_cy d
                    _w d_h confidence_score]
            y_true: [(batch_size, gird_h, grid_w, anchors_per_scale, 5 + num_classes), ...]
                    where 5=[n_cx n_cy n_w n_h confidence_score]
    :param anchors: ndarray, shape=(clusters, 2) where 2=[width height]
    :param anchors_per_scale: int, divide up the clusters across scales
    :param num_classes: int, classes numbers
    :param ignore_thresh: float, the IoU threshold whether to ignore object confidence loss
    :param print_loss: bool, print loss or not
    :return: tensor, shape=(1,)
    """
    num_out_layers = len(anchors) // anchors_per_scale
    yolo_outputs = args[:num_out_layers]
    y_true = args[num_out_layers:]
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_out_layers == 3 else [[3, 4, 5], [0, 1, 2]]

    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_out_layers)]

    loss = 0
    batch_size = K.shape(yolo_outputs[0])[0]
    batch_size_f = K.cast(batch_size, K.dtype(yolo_outputs[0]))

    for l in range(num_out_layers):
        object_mask = y_true[l][..., 4:5]      # confidence score
        true_class_probs = y_true[l][..., 5:]  # class probabilities

        grid, raw_pred, pred_xy, pred_wh = process_yolo_outputs(yolo_outputs[l], anchors[anchor_mask[l]],
                                                                num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])
        # grid, shape=(grid_h, grid_w, 1, 2)
        # raw_pred, shape=(batch_size, grid_h, grid_w, anchors_per_scale, 5 + num_classes)
        # pred_box, shape=(batch_size, grid_h, grid_w, anchors_per_scale, 4)

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        def loop_body(b, ignore_mask):
            true_boxes = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
            iou_ = box_iou(pred_box[b], true_boxes)
            best_iou = K.max(iou_, axis=-1)
            # true_boxes, shape=(image_label_boxes, 4), rank=rank(t)-rank(m)+1=2
            # iou_, shape=(grid_h, grid_w, anchors_per_scale, image_label_boxes)
            # best_iou, shape=(grid_h, grid_w, anchors_per_scale)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_boxes)))
            return b + 1, ignore_mask

        _, ignore_mask = K.control_flow_ops.while_loop(lambda b, *args: b < batch_size,
                                                       loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()  # shape=(batch_size, grid_h, grid_w, anchors_per_scale)
        ignore_mask = K.expand_dims(ignore_mask, -1)  # shape=(batch_size, grid_h, grid_w, anchors_per_scale, 1)

        # Darknet raw true box to calculate loss.
        raw_true_xy = y_true[l][..., :2] * grid_shapes[l][::-1] - grid                          # 0 to 1
        raw_true_wh = K.log(y_true[l][..., 2:4] * input_shape[::-1] / anchors[anchor_mask[l]])  # positive or negative
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))             # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]                          # 2-w*h, 1 to 2

        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2],
                                                                       from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                              from_logits=True) + \
                          (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                                    from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

        xy_loss = K.sum(xy_loss) / batch_size_f
        wh_loss = K.sum(wh_loss) / batch_size_f
        confidence_loss = K.sum(confidence_loss) / batch_size_f
        class_loss = K.sum(class_loss) / batch_size_f
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)],
                            message='loss: ')
    return loss
