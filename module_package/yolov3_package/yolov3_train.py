import os
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import plot_model

from yolov3.model import preprocess_true_boxes, mini_yolo_body, tiny_yolo_body, yolo_body, yolo_loss
from yolov3.utils import get_lines, get_anchors, get_data
import data


POWER = 'GPU'
MODEL_NAME = 'yolov3'
INPUT_SHAPE = data.INPUT_SHAPE  # (height, width)
ANCHORS_PER_SCALE = 3     # divide up the clusters across scales


def _main(power, model_name, input_shape, anchors_per_scale, load_pretrained):
    if power == 'GPU':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.3
        sess = tf.Session(config=config)
        K.set_session(sess)
    elif power == 'CPU':
        num_cores = 0  # auto
        config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                                inter_op_parallelism_threads=num_cores,
                                allow_soft_placement=True, device_count={'CPU': 4})
        sess = tf.Session(config=config)
        K.set_session(sess)

    classes_path, anchors_path, annotations_path, pretrained_path, log_dir = choose_model(model_name)

    class_names = get_lines(classes_path)    # list
    num_classes = len(class_names)           # int
    anchors = get_anchors(anchors_path)      # ndarray

    model = create_model(model_name, input_shape, anchors, anchors_per_scale, num_classes,
                         load_pretrained, freeze_body=1, weights_path=pretrained_path)

    # Callbacks for training.
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True,
                                 save_best_only=True, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1)

    # Get train and validation annotations.
    train_annotations = get_lines(annotations_path)
    np.random.seed(47)
    np.random.shuffle(train_annotations)
    np.random.seed(None)
    num_train = len(train_annotations)
    val_annotations = get_lines('dataset/baseball_val.txt')
    num_val = len(val_annotations)

    if False:
        print('\n====================== Train Model Start ======================')

        model.compile(optimizer=Adam(lr=1e-3),
                      loss={'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 32
        print('Train on {} samples, val on {} samples, with batch size {}.\n'.format(num_train, num_val, batch_size))
        model.fit_generator(generator=data_generator_wrapper(
                                            train_annotations, batch_size, input_shape,
                                            anchors, anchors_per_scale, num_classes),
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=data_generator_wrapper(
                                            val_annotations, batch_size, input_shape,
                                            anchors, anchors_per_scale, num_classes),
                            validation_steps=max(1, num_val // batch_size),
                            epochs=20,
                            initial_epoch=0,
                            callbacks=[logging, checkpoint])

        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

        print('======================= Train Model End =======================\n')

    if True:
        print('\n======================= Tune All Start =======================')

        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        print('Unfreeze all of the layers.')

        model.compile(optimizer=Adam(lr=1e-4),
                      loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        batch_size = 4  # note that more memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.\n'.format(num_train, num_val, batch_size))
        model.fit_generator(generator=data_generator_wrapper(
                                            train_annotations, batch_size, input_shape,
                                            anchors, anchors_per_scale, num_classes),
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=data_generator_wrapper(
                                            val_annotations, batch_size, input_shape,
                                            anchors, anchors_per_scale, num_classes),
                            validation_steps=max(1, num_val // batch_size),
                            epochs=300,
                            initial_epoch=127,
                            callbacks=[logging, checkpoint, reduce_lr, early_stopping])

        model.save_weights(log_dir + 'trained_weights_final.h5')

        print('======================== Tune All End ========================\n')


def create_model(model_name, input_shape, anchors, anchors_per_scale,
                 num_classes, load_pretrained=True, freeze_body=2,
                 weights_path='model_data/yolov3-tiny_weights.h5'):
    """
    Create a model for training new or fine tuning.
    :param model_name: str, 'model name'
    :param input_shape: tuple, (height, width)
    :param anchors: ndarray, shape=(clusters, 2) where 2=[width height]
    :param anchors_per_scale: int, divide up the clusters across scales
    :param num_classes: int, classes numbers
    :param load_pretrained:
            True = fine tune
            False = train new
    :param freeze_body: if fine tune must notice freeze or not
            0 = use initial weights to tune all
            1 = freeze mini13 or tiny5 or darknet53 body
            2 = freeze all but 2 or 3 output conv1x1 layers
    :param weights_path: str, 'weights file path'
    :return: a model
    """
    K.clear_session()
    h, w = input_shape

    image_input = Input(shape=(h, w, 3))  # (?, h, w, 3)
    num_anchors = len(anchors)
    num_out_layers = num_anchors // anchors_per_scale

    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l],
                           w // {0: 32, 1: 16, 2: 8}[l],
                           anchors_per_scale, 5 + num_classes)) for l in range(num_out_layers)]

    print('\n\n================= Start Create Model =================')

    body = mini_yolo_body
    if model_name == 'yolov3-tiny': body = tiny_yolo_body
    elif model_name == 'yolov3': body = yolo_body
    model_body = body(image_input, anchors_per_scale, num_classes)  # model
    print('Create {} with {} anchors and {} classes.'.format(model_name, num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights from {}'.format(weights_path))
        print('Will fine tune model !!')
        if freeze_body in [1, 2]:
            freeze_1 = 40  # default yolov3-mini
            if model_name == 'yolov3-tiny': freeze_1 = 20
            elif model_name == 'yolov3': freeze_1 = 185
            freeze_2 = len(model_body.layers) - num_out_layers

            num = (freeze_1, freeze_2)[freeze_body - 1]
            for i in range(num):
                model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
        else:
            print('Use initial weights to fine tune whole model.')
    else:
        print('Will train new model !!')

    model_loss = Lambda(function=yolo_loss,
                        output_shape=(1,),
                        name='yolo_loss',
                        arguments={'anchors': anchors,
                                   'anchors_per_scale': anchors_per_scale,
                                   'num_classes': num_classes,
                                   'ignore_thresh': 0.5}
                        )(model_body.output + y_true)

    model = Model(inputs=[model_body.input] + y_true, outputs=model_loss)

    print('=================== Model Created ====================\n')

    # plot_model(model, to_file='model_data/{}.png'.format(model_name),
    #            show_shapes=True, show_layer_names=True)
    model.summary()
    print('\n\n')

    return model


def data_generator(annotation_lines, batch_size, input_shape, anchors, anchors_per_scale, num_classes):
    """Generate the batch_inputs and batch_targets for fit_generator."""
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:  # will shuffle the data first at every epoch
                np.random.shuffle(annotation_lines)
            image, boxes = get_data(annotation_lines[i], input_shape,
                                    max_boxes=25, augmentation=True)
            image_data.append(image)
            box_data.append(boxes)
            i = (i + 1) % n
        image_data = np.array(image_data)  # shape = (batch_size, height, width, 3)
        box_data = np.array(box_data)      # shape = (batch_size, max_boxes, 5)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors,
                                       anchors_per_scale, num_classes)  # list
        yield [image_data] + y_true, np.zeros(batch_size)  # (inputs, targets)


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, anchors_per_scale, num_classes):
    """Check some conditions."""
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0:
        raise ValueError('Miss annotation lines or batch size <= 0.')
    return data_generator(annotation_lines, batch_size, input_shape,
                          anchors, anchors_per_scale, num_classes)


def choose_model(model_name):
    """
    Choose the model which you want to train.
    :param model_name: str, 'model name'
    :return: tuple of strings,
            ('classes path', 'anchors path',
            'annotation path', 'pretrained path', 'logging directory')
    """
    paths_dict = {'model_name': model_name}

    if model_name == 'yolov3-mini':
        paths_dict['classes_path'] = 'cfg/wider_face_classes.txt'
        paths_dict['anchors_path'] = 'cfg/wider_face_anchors_filter_100-10_110011-10.txt'
        paths_dict['annotations_path'] = 'dataset/WIDER_train_100-10_110011.txt'  # various subset
        paths_dict['pretrained_path'] = 'logs/yolov3-mini_006-9/ep121-loss15.259-val_loss16.976.h5'
        # paths_dict['annotations_path'] = 'dataset/WIDER_train_100-10_000000.txt'  # clear subset
        # paths_dict['pretrained_path'] = 'logs/yolov3-mini_006/ep011-loss10.055-val_loss26.923.h5'

    elif model_name == 'yolov3-tiny':
        paths_dict['classes_path'] = 'cfg/wider_face_classes.txt'
        paths_dict['anchors_path'] = 'cfg/wider_face_anchors_filter_100-10_110011-10.txt'
        paths_dict['annotations_path'] = 'dataset/WIDER_train_100-10_110011.txt'
        paths_dict['pretrained_path'] = 'logs/yolov3-tiny_007/ep017-loss17.401-val_loss19.391.h5'
        # paths_dict['anchors_path'] = 'cfg/wider_face_anchors.txt'
        # paths_dict['annotations_path'] = 'dataset/WIDER_train.txt'
        # paths_dict['pretrained_path'] = 'model_data/yolov3-tiny_weights.h5'

    elif model_name == 'yolov3':
        paths_dict['classes_path'] = 'cfg/baseball_classes.txt'
        paths_dict['anchors_path'] = 'cfg/coco_anchors.txt'
        paths_dict['annotations_path'] = 'dataset/baseball_train.txt'
        # paths_dict['pretrained_path'] = 'model_data/yolov3_weights.h5'
        paths_dict['pretrained_path'] = 'logs/yolov3_freeze1/ep127-loss21.789-val_loss20.778.h5'

    paths_dict['log_dir'] = 'logs/{}_freeze1_new_7/'.format(model_name)

    if model_name in ['yolov3-mini', 'yolov3-tiny', 'yolov3']:
        for path in paths_dict.items():
            print('{}: {}'.format(path[0], path[1]))
        return paths_dict['classes_path'], paths_dict['anchors_path'], \
               paths_dict['annotations_path'], paths_dict['pretrained_path'], paths_dict['log_dir']
    else:
        raise ValueError("Model not exist.")


if __name__ == '__main__':
    _main(power=POWER,
          model_name=MODEL_NAME,
          input_shape=INPUT_SHAPE,
          anchors_per_scale=ANCHORS_PER_SCALE,
          load_pretrained=True)
