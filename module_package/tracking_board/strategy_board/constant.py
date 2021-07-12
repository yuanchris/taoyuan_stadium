import cv2
import numpy as np
from easydict import EasyDict as edict

cfg = edict()
#result size is 557 492
cfg.field_strategy     = np.array([[15, 385], [111, 385], [111, 289], [15, 289]], dtype=np.float32)
cfg.result_strategy    = np.array([[278, 456], [350, 384], [278, 312], [206, 384]], dtype=np.float32)
cfg.east_strategy      = np.array([[440, 1422], [2661, 1422], [2225, 822], [678,806]], dtype=np.float32)
cfg.west_strategy      = np.array([[3540, 1434], [3262, 751], [1522, 792], [1044,1472]], dtype=np.float32)
cfg.mound_ew           = np.array([[1457,1150],[2392,1150]], dtype=np.float32)
cfg.mound              = np.array([60,340], dtype=np.float32)

cfg.east_perspective   = cv2.getPerspectiveTransform(cfg.east_strategy, cfg.field_strategy)
cfg.west_perspective   = cv2.getPerspectiveTransform(cfg.west_strategy, cfg.field_strategy)
cfg.result_perspective = cv2.getPerspectiveTransform(cfg.field_strategy, cfg.result_strategy)

cfg.score_diff         = 0.15
cfg.H_limit            = 40
cfg.H_left_limit       = 35
cfg.H_right_limit      = 91
cfg.score_limit        = 0.3
cfg.stuck_frames       = 7
cfg.H_border           = 130
cfg.always_in_ground   = ['P','C','H','1B','2B','3B','SS','LF','CF','RF','1H','2H','3H']
cfg.no_H_in_D_range    = 20