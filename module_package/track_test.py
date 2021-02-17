import queue
import threading
import time
import cv2
import numpy as np
from PIL import Image
import json
import websockets, asyncio

FIELD_IMAGE_FILE = "./module_package/src/field.png"

class Camera_System:
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.fieldIamge = cv2.imread(FIELD_IMAGE_FILE)
        self.defPoseList = {'C':[207 ,341], 'P':[207,281], '1B': [260, 253], 
        '2B':[235,238],'SS': [186, 236], '3B': [151,266], 'LF': [112,118],
        'CF':[200,75],  'RF': [328, 114]}
        self.send_pic = None


    def execute(self, yoloResults):
        self.fieldIamge = cv2.imread(FIELD_IMAGE_FILE)
        # print("-------------- Tracking board ----------------------------")
        a = int(time.time())
        if a % 4 == 1:
            self.defPoseList = {'C':[207 ,341], 'P':[207 ,281 + 20], '1B': [260, 253], 
                '2B':[235,238],'SS': [186, 236], '3B': [151,266], 'LF': [112,118],
                'CF':[200,75],  'RF': [328, 114]}
        elif a % 4 == 2:
            self.defPoseList = {'C':[207 ,341], 'P':[207 - 20 ,281], '1B': [260, 253], 
                '2B':[235,238],'SS': [186, 236], '3B': [151,266], 'LF': [112,118],
                'CF':[200,75],  'RF': [328, 114]}
        elif a % 4 == 3:
            self.defPoseList = {'C':[207 ,341], 'P':[207 + 20 ,281], '1B': [260, 253], 
                '2B':[235,238],'SS': [186, 236], '3B': [151,266], 'LF': [112,118],
                'CF':[200,75],  'RF': [328, 114]}
        else:
            self.defPoseList = {'C':[207 ,341], 'P':[207,281], '1B': [260, 253], 
                '2B':[235,238],'SS': [186, 236], '3B': [151,266], 'LF': [112,118],
                'CF':[200,75],  'RF': [328, 114]}


        for key in self.defPoseList:
            # print(self.defPoseList)
            # cv2.circle(影像, 圓心座標, 半徑, 顏色, 線條寬度)
            cv2.circle(self.fieldIamge, point2fToIntTuple(self.defPoseList[key]), 3, (200, 0, 0), 3)
            # cv2.imshow("Field", self.fieldIamge)
            # cv2.waitKey(0)
            self.send_pic = cv2.imencode('.jpg', self.fieldIamge)[1].tostring()
            # print('self.send_pic[0]:',self.send_pic[0])
        return [self.defPoseList, self.send_pic]

def point2fToIntTuple(point2f):
    """Convert point2f (np.array) to tuple.
    Args:
        point2f:    np.array((x, y))
    Returns:
        tuple
    """
    return (int(point2f[0]), int(point2f[1]))


if __name__ == '__main__':
    a = Camera_System(2)
    a.execute(1)

