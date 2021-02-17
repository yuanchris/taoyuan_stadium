# WIDER FACE data path


# image folder
TRAIN_FOLDER = 'dataset/WIDER_train'
VAL_FOLDER = 'dataset/WIDER_val'


# original train and val file
TRAIN_BBX = 'dataset/wider_face_split/wider_face_train_bbx_gt.txt'
TRAIN_ANNOTATIONS = 'dataset/WIDER_train.txt'
VAL_BBX = 'dataset/wider_face_split/wider_face_val_bbx_gt.txt'
VAL_ANNOTATIONS = 'dataset/WIDER_val.txt'

# new train file by filter boxes of a image >= 100
NEW_TRAIN_BBX = 'dataset/wider_face_split/wider_face_train_bbx_gt_100.txt'
NEW_TRAIN_ANNOTATIONS = 'dataset/WIDER_train_100.txt'
# new val file by filter wh blur expression illumination invalid occlusion pose
NEW_VAL_BBX = 'dataset/wider_face_split/wider_face_val_bbx_gt-10_110011.txt'
NEW_VAL_ANNOTATIONS = 'dataset/WIDER_val-10_110011.txt'


# filter train file
# filter wh blur expression illumination invalid occlusion pose
FILTER_BBX = 'dataset/wider_face_split_new_100_filter/wider_face_train_bbx_gt_100-10_110011.txt'
FILTER_INFO = 'dataset/wider_face_split_new_100_filter/image_info/filter_train_delete_100-10_110011.txt'
FILTER_ANNOTATIONS = 'dataset/WIDER_train_100-10_110011.txt'
FILTER_ANCHORS = 'cfg/wider_face_anchors_filter_100-10_110011-10.txt'

INPUT_SHAPE = (416, 416)  # (height, width)
