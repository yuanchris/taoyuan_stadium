import os
import numpy as np
from functools import reduce
from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import cv2


def iou(boxes_wh, anchors_wh):
    """
    Calculates the IoU between r boxes and k clusters.
    :param boxes_wh: ndarray, shape=(boxes, 2)
    :param anchors_wh: ndarray, shape=(clusters, 2)
    :return: ndarray, shape=(boxes, clusters)
    """
    # Expand dimension to apply broadcasting.
    boxes_wh = np.expand_dims(boxes_wh, -2)                               # shape=(boxes, 1, 2)
    anchors_wh = np.expand_dims(anchors_wh, 0)                            # shape=(1, clusters, 2)

    # Ignore the coordinate of boxes.
    intersect_wh = np.minimum(boxes_wh, anchors_wh)                       # shape=(boxes, clusters, 2)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]          # shape=(boxes, clusters)
    if np.count_nonzero(intersect_area == 0) > 0:
        raise ValueError("Box has no area.")

    boxes_area = boxes_wh[..., 0] * boxes_wh[..., 1]                      # shape=(boxes, 1)
    anchors_area = anchors_wh[..., 0] * anchors_wh[..., 1]                # shape=(1, clusters)
    iou_ = intersect_area / (boxes_area + anchors_area - intersect_area)  # shape=(boxes, clusters)

    return iou_


def get_lines(file_path):
    """
    Load lines from a file.
    :param file_path: str, 'file path'
    :return: list of strings
    """
    file_path = os.path.expanduser(file_path)
    with open(file_path, encoding='utf-8') as f:
        lines = f.readlines()
        return list(map(str.strip, lines))


def get_anchors(anchors_path):
    """
    Load the anchors from a file.
    :param anchors_path: 'anchors file path'
    :return: ndarray, shape=(anchors, 2)
    """
    anchors_path = os.path.expanduser(anchors_path)
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
        anchors = [float(a.strip()) for a in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)


def compose(*funcs):
    """
    Reference: https://mathieularose.com/function-composition-in-python/
    Compose arbitrarily many functions, evaluated left to right.
    """
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def letterbox_image(image, input_shape):
    """Resize image with unchanged aspect ratio using padding."""
    iw, ih = image.size
    h, w = input_shape
    scale = min(float(w) / float(iw), float(h) / float(ih))
    nw = int(round(iw * scale))
    nh = int(round(ih * scale))
    dx = (w - nw) // 2
    dy = (h - nh) // 2

    # image = image.resize((nw, nh), Image.BICUBIC)
    image = image.resize((nw, nh), Image.NEAREST)
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    return new_image


def letterbox_image_cv(image, input_shape):
    """Resize image with unchanged aspect ratio using padding."""
    ih, iw = (image.shape[0], image.shape[1])
    h, w = input_shape
    scale = min(float(w) / float(iw), float(h) / float(ih))
    nw = int(round(iw * scale))
    nh = int(round(ih * scale))
    dx = (w - nw) // 2
    dy = (h - nh) // 2

    # image = image.resize((nw, nh), Image.BICUBIC)
    r_image = cv2.resize(image, (nw, nh), cv2.INTER_CUBIC)
    new_image = np.zeros((h, w, 3), dtype='float32')
    new_image.fill(128)
    new_image[dy:dy+r_image.shape[0], dx:dx+r_image.shape[1], :] = r_image
    return new_image


def rand(a=0., b=1.):
    return np.random.rand() * (b - a) + a  # [a, b) or (b, a]


def get_data(annotation_line, input_shape,
             max_boxes=20, augmentation=True, jitter=.3,
             hue=.1, sat=1.5, val=1.5, proc_img=True):
    """
    Pre-processing resize data or real-time data augmentation.
    :param annotation_line: str, 'image_path x_min,y_min,x_max,y_max,class_id ...'
    :param input_shape: tuple, (height, width)
    :param max_boxes: int, in order to generate fit dimension
    :param augmentation: bool, data augmentation or not
    :param jitter: float, random jitter
    :param hue: float, random hue
    :param sat: float, random Saturation
    :param val: float, random value
    :param proc_img: bool, protect image or not
    :return: tuple, (image_data, box_data)
            image_data: ndarray, shape=(height, width, 3)
                    where 3=[R G B] and all value 0 to 1
            box_data: ndarray, shape=(max_boxes, 5)
                    where 5=[rx_min ry_min rx_max ry_max class_id]
    """
    line = annotation_line.split()
    image = Image.open(line[0])  # mode='RGBA'
    iw, ih = image.size
    h, w = input_shape
    boxes = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
    # boxes shape=(image_label_boxes, 5)

    if not augmentation:
        # Resize image with unchanged aspect ratio using padding.
        scale = min(float(w) / float(iw), float(h) / float(ih))
        nw = int(round(iw * scale))
        nh = int(round(ih * scale))
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        image_data = 0
        if proc_img:
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image) / 255.  # normalize ndarray, 0 to 1, shape=(h, w, 3)

        # correct boxes
        box_data = np.zeros((max_boxes, 5))
        if len(boxes) > 0:
            np.random.shuffle(boxes)
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * nw / iw + dx   # x_min, x_max
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * nh / ih + dy   # y_min, y_max
            box_w = boxes[:, 2] - boxes[:, 0]                    # shape=(image_label_boxes, )
            box_h = boxes[:, 3] - boxes[:, 1]                    # shape=(image_label_boxes, )
            boxes = boxes[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
            if len(boxes) > max_boxes: boxes = boxes[:max_boxes]
            box_data[:len(boxes)] = boxes                        # ndarray, shape=(max_boxes, 5)

        return image_data, box_data

    # Data augmentation.
    # resize image
    new_ar = (w / h) * (rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter))
    scale = rand(.25, 2.)
    if new_ar < 1:
        nh = int(round(h * scale))
        nw = int(round(nh * new_ar))
    else:
        nw = int(round(w * scale))
        nh = int(round(nw / new_ar))

    # place image
    dx = int(rand(0, w - nw))  # >= 0 or <= 0
    dy = int(rand(0, h - nh))  # >= 0 or <= 0

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = rand() < .5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = rgb_to_hsv(np.array(image) / 255.)  # ndarray, 0 to 1
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x)  # ndarray, 0 to 1, shape=(h, w, 3)

    # correct boxes
    box_data = np.zeros((max_boxes, 5))
    if len(boxes) > 0:
        np.random.shuffle(boxes)
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * nw / iw + dx
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * nh / ih + dy
        if flip: boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        boxes[:, 0:2][boxes[:, 0:2] < 0] = 0                 # x_min, y_min
        boxes[:, 2][boxes[:, 2] > w] = w                     # x_max
        boxes[:, 3][boxes[:, 3] > h] = h                     # y_max
        box_w = boxes[:, 2] - boxes[:, 0]                    # shape=(image_label_boxes, )
        box_h = boxes[:, 3] - boxes[:, 1]                    # shape=(image_label_boxes, )
        boxes = boxes[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        if len(boxes) > max_boxes: boxes = boxes[:max_boxes]
        box_data[:len(boxes)] = boxes                        # ndarray, shape=(max_boxes, 5)

    return image_data, box_data
