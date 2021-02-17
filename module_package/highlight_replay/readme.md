# Highlight Replay 

## << Problem >>
1. If sec/frame > 0.1:

    i.  It need to extend region

    ii. Reduce condition1 & condition2 count

2. bounging box is missing:

    i.  edit yolov2_predict_gpus_cv.py
      
        SCORE_THRESHOLD = 0.5 >> 0.3
