import queue
import threading
import time
import cv2
import numpy as np
from PIL import Image
import json
import websockets, asyncio
import random
FIELD_IMAGE_FILE = "./module_package/src/field.png"

class Camera_System:
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.fieldIamge = cv2.imread(FIELD_IMAGE_FILE)
        self.send_pic = None


    def execute(self, yoloResults, game_return_data, now_state):
        # self.fieldIamge = cv2.imread(FIELD_IMAGE_FILE)

        pic = self.fieldIamge.copy()
        if game_return_data['pause_start'] == 'false' or now_state == False:
            return [{'pause_start':'false'}, pic]

        self.defPoseList = {'C':[207 ,341], 'P':[207,281], '1B': [260, 253], 
        '2B':[235,238],'SS': [186, 236], '3B': [151,266], 'LF': [112,118],
        'CF':[200,75],  'RF': [328, 114], 'H':[207 ,331]}
        for key in game_return_data:
            if key == 'CurrentBat':
                self.defPoseList['H'].append(game_return_data[key])
            try:
                self.defPoseList[key.split('_')[1]].append(game_return_data[key])
            except:
                continue
        # print("-------------- Tracking board ----------------------------")
        a = int(time.time())
        if a % 4 == 1:
            for key in self.defPoseList:
                # self.defPoseList[key][0] += 5*random.random()
                self.defPoseList[key][0] += 3*random.random()
        elif a % 4 == 2:
            for key in self.defPoseList:
                self.defPoseList[key][0] -= 3*random.random()
        elif a % 4 == 3:
            for key in self.defPoseList:
                self.defPoseList[key][1] += 3*random.random()
        else:
            for key in self.defPoseList:
                self.defPoseList[key][1] -= 3*random.random()

        # for key in self.defPoseList:
        #     # cv2.circle(影像, 圓心座標, 半徑, 顏色, 線條寬度)
        #     if key == 'H':
        #         cv2.circle(pic, point2fToIntTuple(self.defPoseList[key]), 3, (0, 0, 200), -1)
        #         cv2.putText(pic, key+' '+ self.defPoseList[key][2], textLocation(self.defPoseList[key]), cv2.FONT_HERSHEY_DUPLEX,
        #             0.3,(0, 0, 200), 0, cv2.LINE_AA)
        #     else:
        #         cv2.circle(pic, point2fToIntTuple(self.defPoseList[key]), 3, (200, 0, 0), -1)
        #         cv2.putText(pic, key+' '+ self.defPoseList[key][2], textLocation(self.defPoseList[key]), cv2.FONT_HERSHEY_DUPLEX,
        #             0.3,(200, 0, 0), 0, cv2.LINE_AA)
        # self.send_pic = cv2.imencode('.jpg', pic)[1].tostring()
        # return [self.defPoseList, self.send_pic]
        
            
        return [self.defPoseList, pic]


def point2fToIntTuple(point2f):
    """Convert point2f (np.array) to tuple.
    Args:
        point2f:    np.array((x, y))
    Returns:
        tuple
    """
    return (int(point2f[0]), int(point2f[1]))

def textLocation(point2f):
    """Convert point2f (np.array) to tuple.
    Args:
        point2f:    np.array((x, y))
    Returns:
        tuple
    """
    return (int(point2f[0] - 5 ), int(point2f[1] - 5))


if __name__ == '__main__':
    a = Camera_System(2)
    a.execute(1)

