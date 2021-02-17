import os, sys
import colorsys
import numpy as np
import cv2
from timeit import default_timer as timer
from PIL import Image, ImageFont, ImageDraw

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

from keras import backend as K
from keras.layers import Input
import tensorflow as tf
from shutil import copyfile
from keras.utils import multi_gpu_model


from .yolov3.model import mini_yolo_body, tiny_yolo_body, yolo_body, yolo_eval, fast_yolo_body, lite_yolo_body
from .yolov3.utils import letterbox_image, letterbox_image_cv
from . import data

# condition
SIZE = 640
MODEL_NAME = 'yolov3'

# Detect frame
FRAME_FOLDER = 'nas_5_20-30'
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
# FRAME_FOLDER = 'slide'

# Detect a test image.
IMG_NAME = '1'
IMAGE_FILE = 'dataset/test/{}.jpg'.format(IMG_NAME)
SAVE_FILE = 'test_result/{}_baseball/{}_0.5_{}.png'.format(MODEL_NAME, IMG_NAME, SIZE)

# Detect val images for mAP.
ANNOTATIONS_FILE = data.NEW_VAL_ANNOTATIONS  # 2765 samples
DETECT_OUT_FOLDER = 'mAP/input/detection-results/'

# Detect video.
# VIDEO_FILE = 0 # camera
VIDEO_FILE = 'dataset/video/test1-1.mp4'


class YOLO(object):
    def __init__(self, model_name, classes_path, anchors_path, weights_path, gpu_id):
        K.clear_session()
        self.model_name = model_name
        self.classes_path = classes_path
        self.anchors_path = anchors_path
        self.weights_path = weights_path
        self.gpu_id = int(gpu_id)

        self.max_boxes = 200
        # self.score_threshold = 0.2   # tiny
        # self.iou_threshold = 0.05    # tiny
        self.score_threshold = SCORE_THRESHOLD
        self.iou_threshold = IOU_THRESHOLD
        self.class_names = self._get_class()   # list
        self.anchors = self._get_anchors()     # ndarray
        self.model_image_shape = (SIZE, SIZE)  # (height, width)
        self.anchors_per_scale = 3
        self.colors = self.__get_colors(self.class_names)  # list

        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=config)
        # notice graph and device position
        with self.graph.as_default(), tf.device('/gpu:{}'.format(self.gpu_id)):
            with self.sess.as_default():
                self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path, encoding='utf-8') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path, encoding='utf-8') as f:
            anchors = f.readline()
        anchors = [float(x.strip()) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    @staticmethod
    def __get_colors(names):
        hsv_tuples = [(float(x) / len(names), 1., 1.) for x in range(len(names))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))                          # RGB, 0 to 1
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))  # RGB, 0 to 255
        # colors = list([(10, 70, 150)])
        np.random.seed(110101)
        np.random.shuffle(colors)
        np.random.seed(None)
        return colors

    def generate(self):
        weights_path = os.path.expanduser(self.weights_path)
        assert weights_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        h, w = self.model_image_shape
        image_input = Input(shape=(h, w, 3))  # (?, h, w, 3)
        num_classes = len(self.class_names)
        num_anchors = len(self.anchors)
        # with tf.device("/gpu:0"):
        print('\n============= Start Reload Weights =============')
        body = None
        if self.model_name == 'yolov3-mini':
            body = mini_yolo_body
        elif self.model_name == 'yolov3-tiny':
            body = tiny_yolo_body
        elif self.model_name == 'yolov3':
            body = yolo_body
        elif self.model_name == 'yolov3-fast':
            body = fast_yolo_body
        elif self.model_name == 'yolov3-lite':
            body = lite_yolo_body
        if not body: raise ValueError('{} load error'.format(self.model_name))
        self.yolo_model = body(image_input, self.anchors_per_scale, num_classes)  # model
        self.yolo_model.load_weights(weights_path)
        print('Create {} with {} anchors and {} classes.'.format(self.model_name, num_anchors, num_classes))
        print('============ Model Weights Roloaded ============\n')

        self.original_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors, num_classes,
                                           self.original_image_shape, self.max_boxes,
                                           self.score_threshold, self.iou_threshold)
        return boxes, scores, classes

    def detect_image(self, image, draw_box=False, convert_dr=False):
        start = timer()

        ih, iw = (image.shape[0], image.shape[1])
        if self.model_image_shape != (None, None):
            assert self.model_image_shape[0] % 32 == 0, 'Multiples of 32 required.'  # height
            assert self.model_image_shape[1] % 32 == 0, 'Multiples of 32 required.'  # width
            boxed_image = letterbox_image_cv(image, self.model_image_shape)
            boxed_image = cv2.cvtColor(boxed_image, cv2.COLOR_BGR2RGB)
        else:
            new_image_size = (ih - (ih % 32),
                              iw - (iw % 32))
            boxed_image = letterbox_image_cv(image, new_image_size)
            boxed_image = cv2.cvtColor(boxed_image, cv2.COLOR_BGR2RGB)

        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.  # ndarray, 0 to 1
        # Expand batch dim, shape=(1, height, width, RGB).
        image_data = np.expand_dims(image_data, 0)

        # Feed test image to the model.
        with self.sess.as_default() as sess, self.graph.as_default():
            out_boxes, out_scores, out_classes = \
                sess.run(fetches=[self.boxes, self.scores, self.classes],
                         feed_dict={self.yolo_model.input: image_data,
                                    self.original_image_shape: [ih, iw],
                                    K.learning_phase(): 0})

        # print('Found {} boxes from image.'.format(len(out_boxes)))

        boxes_l = []
        info_l = []  # c_x c_y x_min x_max w h

        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            top, left, bottom, right = box  # y_min, x_min, y_max, x_max
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(ih, np.floor(bottom + 0.5).astype('int32'))
            right = min(iw, np.floor(right + 0.5).astype('int32'))
            c_x = int((left + right) // 2)
            c_y = int((top + bottom) // 2)
            w = right - left
            h = bottom - top
            info_l.append('{} {} {} {} {} {} {}'.format(c_x, c_y, left, top, w, h, c))

            if draw_box:
                thickness = (iw + ih) // 512
                label = '{} {:.2f}'.format(predicted_class, score)
                if w < 80:
                    text_info = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 1, 2)
                else:
                    text_info = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 1, 2)

                label_size = np.array(text_info[0], dtype='int32')  # (w, h)
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                # My kingdom for a good redistributable image drawing library.
                for p in range(thickness):  # box
                    cv2.rectangle(image, (left + p, top + p), (right - p, bottom - p), self.colors[c], 1)

                cv2.rectangle(image, tuple(text_origin), tuple(text_origin + label_size), self.colors[c],
                              -1)  # text background
                if w < 80:
                    cv2.putText(image, label, tuple(text_origin + (0, 20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                else:
                    cv2.putText(image, label, tuple(text_origin + (0, 20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            if convert_dr:
                box_l = [predicted_class, np.around(score, 6), left, top, right, bottom]
                box_l = list(map(str, box_l))
                boxes_l.append(' '.join(box_l))

        end = timer()
        detect_time = end - start
        if not convert_dr: return image, detect_time, info_l
        assert len(out_boxes) == len(boxes_l), 'Detect boxes miss.'
        return image, detect_time, '\n'.join(boxes_l)

    def close_session(self):
        self.sess.close()


def model_path(model_name):
    """Choose the model which you want to test."""
    print(os.getcwd())
    relative_path = "module_package/yolov3_package/"
    paths_dict = {'model_name': model_name}

    if model_name == 'yolov3-mini':
        paths_dict['classes_path'] = os.path.join(relative_path, 'cfg/wider_face_classes.txt')
        paths_dict['anchors_path'] = os.path.join(relative_path, 'cfg/wider_face_anchors_filter_100-10_110011-10.txt')
        paths_dict['weights_path'] = os.path.join(relative_path, 'logs/yolov3-mini_006-9/ep121-loss15.259-val_loss16.976.h5')

    elif model_name == 'yolov3-tiny':
        paths_dict['classes_path'] = os.path.join(relative_path, 'cfg/wider_face_classes.txt')
        paths_dict['anchors_path'] = os.path.join(relative_path, 'cfg/wider_face_anchors_filter_100-10_110011-10.txt')
        paths_dict['weights_path'] = os.path.join(relative_path, 'logs/yolov3-tiny_007/ep017-loss17.401-val_loss19.391.h5')
        # paths_dict['anchors_path'] = 'cfg/wider_face_anchors.txt'
        # paths_dict['weights_path'] = 'model_data/yolov3-tiny_weights.h5'

    elif model_name == 'yolov3':
        # paths_dict['classes_path'] = 'cfg/coco_classes.txt'
        paths_dict['classes_path'] = os.path.join(relative_path, 'cfg/baseball_classes.txt')
        paths_dict['anchors_path'] = os.path.join(relative_path, 'cfg/coco_anchors.txt')
        # paths_dict['weights_path'] = 'model_data/yolov3_weights.h5'
        paths_dict['weights_path'] = os.path.join(relative_path, 'logs/yolov3_freeze0_c2/ep083-loss17.367-val_loss16.106.h5')

    elif model_name == 'yolov3-lite' or 'yolov3-fast':
        paths_dict['classes_path'] = os.path.join(relative_path, 'cfg/baseball_classes.txt')
        paths_dict['anchors_path'] = os.path.join(relative_path, 'cfg/coco_anchors.txt')
        paths_dict['weights_path'] = os.path.join(relative_path, 'logs/yolov3-fast_freeze0_c2/ep191-loss16.697-val_loss14.310.h5')

    if model_name in ['yolov3-mini', 'yolov3-tiny', 'yolov3',  'yolov3-lite', 'yolov3-fast']:
        for path in paths_dict.items():
            print('{}: {}'.format(path[0], path[1]))
        return paths_dict
    else:
        raise ValueError("Model not exist.")


def initial_model(model_name='yolov3', gpu_id=0):
    """
    Create model and load weight.
    :param gpu_id: int, choose gpu
    :param model_name: str, 'model name' # ex:'yolov3'
    :return: object, model
    """
    paths_kwargs = model_path(model_name)
    paths_kwargs['gpu_id'] = gpu_id
    yolo = YOLO(**paths_kwargs)
    return yolo


def detect_frame(yolo, frame_path, save_info=False, save_img=False):
    """
    Detect images from frame folder path.
    :param yolo: object, model
    :param frame_path: str, 'frame folder path'
    :param save_info: bool, save detect information or not
    :param save_img: bool, save detect image or not
    """
    img_paths = os.listdir(frame_path)
    save_dir = 'test_result/{}_baseball/{}_{}_freeze0_c2_{}!!!!!!'.format(MODEL_NAME, SIZE,
                                                                          SCORE_THRESHOLD, FRAME_FOLDER)
    os.makedirs(save_dir)

    for img_path in img_paths:
        image = cv2.imread(frame_path + '/' + img_path)
        image, detect_time, info = yolo.detect_image(image)
        img_path = img_path.split('.')[0] + '.' + img_path.split('.')[1]

        if save_info:
            # out_path = 'out_{}_{}_freeze0_new_{}/{}.txt'.format(SIZE, SCORE_THRESHOLD, FRAME_FOLDER, img_path)
            out_path = '{}.txt'.format(FRAME_FOLDER)
            info = '\n'.join(info)
            with open(out_path, 'a+', encoding='utf-8') as f:
                f.write(info)
        print('detect time: {:.3f} ms\n'.format(detect_time * 1000))
        if save_img:
            cv2.imwrite(save_dir + '/{}.jpg'.format(img_path), image)
    yolo.close_session()


def detect_img(yolo, image, save_img=False):
    """
    Detect a image from image path.
    :param yolo: object, model
    :param img_path: str, 'frame folder path'
    :param save: bool, save detect image or not
    :return: list of strings, ['c_x c_y x_min y_min w h', ...]
    """
    image, detect_time, info = yolo.detect_image(image)
    # print('detect time: {:.3f} ms\n'.format(detect_time * 1000))
    if save_img:
        cv2.imwrite(SAVE_FILE, image)
    # yolo.close_session()
    return info


def detect_imgs_for_map(model_name, annotations_file, detect_out_folder):
    """
    Detect images and convert the detect result of each image to dr_img_path.txt.
    format of dr_img_path.txt: 'class_name confidence left top right bottom'
    """
    print('detect images from:', annotations_file, '\n')
    paths_kwargs = model_path(model_name)
    yolo = YOLO(**paths_kwargs)
    annotation_lines = read_file(annotations_file)
    num_images = len(annotation_lines)
    total_detect_time = 0

    for c, annotation_line in enumerate(annotation_lines, 1):
        line = annotation_line.split()
        image = Image.open(line[0])
        draw_box = False
        image, detect_time, detect_result = yolo.detect_image(image, draw_box, convert_dr=True)
        # print('detect time: {:.3f} ms\n'.format(detect_time * 1000))
        total_detect_time += detect_time
        img_path = line[0].split('\\')
        img_path = img_path[-1].split('.')
        detect_file = '{}-{}'.format(c, img_path[0])
        detect_out_file = detect_out_folder + detect_file + '.txt'
        write_dr(detect_out_file, detect_result)
        if draw_box:
            image.save('mAP/mAP_info/{}_{}/draw_dr/{}.png'.format(yolo.score_threshold,
                                                                  yolo.iou_threshold,
                                                                  detect_file))
    print('\nValidation samples:', num_images)
    print('Total detect time:', total_detect_time * 1000, 'ms')
    print('Average detect time:', (total_detect_time / num_images) * 1000, 'ms')
    yolo.close_session()


def detect_video(model_name, video_file):
    paths_kwargs = model_path(model_name)
    yolo = YOLO(**paths_kwargs)
    if video_file == 0:
        print('Use camera.')
    else:
        print('Use video from {}.'.format(video_file))
    vid = cv2.VideoCapture(video_file)
    # vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    # out = cv2.VideoWriter('dataset/video/output.avi', fourcc, 20.0, (640, 480))

    while True:
        return_bool, frame = vid.read()
        if return_bool:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            thickness = (image.width + image.height) // 1024
        else:
            raise ValueError('No frame.')
        image, detect_time = yolo.detect_image(image)
        image = np.asarray(image)
        detect_time = 'detect time: {:.3f} ms'.format(detect_time * 1000)
        cv2.putText(image, detect_time, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), thickness)
        cv2.namedWindow('detect result', cv2.WINDOW_NORMAL)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('detect result', image)
        # out.write(image)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    vid.release()
    # out.release()
    cv2.destroyAllWindows()


def read_file(data_file):
    """Read lines of list from a file."""
    with open(data_file, encoding='utf-8') as f:
        lines = f.readlines()
        return list(map(str.strip, lines))


def write_dr(file_name, ground_truth):
    """Write the ground truth of each image to a output file."""
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(ground_truth)


if __name__ == '__main__':
#     import threading
#     tracking_thread0 = threading.Thread(target=initial_model, args=('yolov3', 0,))
#     tracking_thread1 = threading.Thread(target=initial_model, args=('yolov3', 1,))
#     tracking_thread0.start()
#     tracking_thread1.start()
#     tracking_thread0.join()
#     tracking_thread1.join()
    yolov3 = initial_model(MODEL_NAME)
    # detect_frame(yolov3, FRAME_FOLDER)
    t = detect_img(yolov3, IMAGE_FILE)
    # detect_imgs_for_map(MODEL_NAME, ANNOTATIONS_FILE, DETECT_OUT_FOLDER)
    # detect_video(MODEL_NAME, VIDEO_FILE)
