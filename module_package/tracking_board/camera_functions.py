#!/usr/bin/env python
# -*- coding: UTF-8 -*- 

'''
=========================================================
File Name   :   camera_functions.py
Copyright   :   Chao-Chuan Lu
Create date :   Sep. 3, 2019
Description :   Functions for Baseball Player Tracking
TODO: Add functioin for different player height (PLAYER_HEIGHT) for homography
=========================================================
'''

import math
import glob as gb
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
import numpy as np
import copy
import pickle
from collections import Counter
import json

from . import team_classifier as TC
from . import player_position_match as PPMatch

NAS_INDEX               =   5

FIELD_NAME              =   "./osense_baseball_player_tracking/src/field.png"
PATH_OF_IMAGE_FOLDER    =   "../../NAS_data/NAS_0{}_20190929-19-43/*.jpg".format(NAS_INDEX)
PATH_OF_REFERENCE       =   "./osense_baseball_player_tracking/reference/"

# MOVEMENT_THRESHOLD      =   40
# PLAYER_HEIGHT           =   50
# DEFENDER_THRESHOLD      =   0.40

# # --- FOR isInField ---
# # IN_FIELD_BASE           =   (255, 1640)
# # IN_FIELD_LEFTBORDER     =   (182, 915)
# # IN_FIELD_RIGHTBORDER    =   (4081, 1212)
# DEF_IN_FIELD_BASE           =   (216, 1524)
# DEF_IN_FIELD_LEFTBORDER     =   (131, 808)
# DEF_IN_FIELD_RIGHTBORDER    =   (4068, 1038)
# DEF_BASE_CIRCLE             =   300   
# OFS_IN_FIELD_BASE           =   (20, 1650)
# OFS_IN_FIELD_LEFTBORDER     =   (10, 1050)
# OFS_IN_FIELD_RIGHTBORDER    =   (2600, 1315)
# OFS_BASE_CIRCLE             =   450
# OFS_RIGHT_OUT_DISTANCE      =   40
# OFS_LEFT_OUT_DISTANCE       =   20


class Player():
    def __init__(self):
        self.id = -1
        self.detectedPose = np.zeros(2)
        self.trackedPose = np.zeros(2)
        self.defaultCameraPose = np.zeros(2)
        self.defaultPose = np.zeros(2)
        self.detectedCameraPose = np.zeros(2)
        self.tracker = None
        self.detectedID = -1
        self.detectedBox = []
        self.boundingBox = []

class DefensivePosition():
    def __init__(self, positionID):
        self.positionID = positionID
        self.playerID = -1
        player = Player()
        self.player = player

class OffensivePosition():
    def __init__(self, positionID):
        self.positionID = positionID
        self.playerID = -1
        player = Player()
        self.player = player

def findHomographyBetweenCameraAndField(NAS_config):
    """Find homegraphy relationship between one camera and field.
    TODO:   Adjust the points of camera and field to the bases?
            Replace the field image with bigger one.
    """
    test1 = cv2.imread("{}camera_NAS_{}.jpg".format(PATH_OF_REFERENCE, NAS_config['NAS_INDEX']))
    test2 = cv2.imread("{}field.png".format(PATH_OF_REFERENCE))
    
    cv2.namedWindow("camera_for_homography_NAS_{}.jpg".format(NAS_config['NAS_INDEX']), cv2.WINDOW_NORMAL)
    cv2.imshow("camera_for_homography_NAS_{}.jpg".format(NAS_config['NAS_INDEX']), test1)
    cv2.namedWindow("field_for_homography", cv2.WINDOW_NORMAL)
    cv2.imshow("field_for_homography", test2)
    cv2.waitKey(0)
    
    # === Neo Solution ===
    # test1_pt = np.array([   [265, 1632],    # 本壘
    #                         [190, 886],     # 扇形左邊頂點
    #                         [2350, 860],    # 扇形中間弧頂點
    #                         [4096, 1008]],  # 扇形右邊頂點
    #                         dtype=np.float32).reshape(-1, 1, 2)

    # test2_pt = np.array([   [207, 335],     # 本壘
    #                         [13, 147],      # 扇形左邊頂點
    #                         [208, 7],       # 扇形中間弧頂點
    #                         [388, 118]],    # 扇形右邊頂點
    #                         dtype=np.int32).reshape(-1, 1, 2)
    
    # === Steven Solution ===
    # NAS_05
    # test1_pt = np.array([   [215, 1524],    # 本壘
    #                         [3028, 1171],   # 一壘延伸綠土交界
    #                         [1816, 940],   # 二壘延伸綠土交界
    #                         [154, 1028]],   # 三壘延伸綠土交界
    #                         dtype=np.float32).reshape(-1, 1, 2)

    # test2_pt = np.array([   [207, 343],     # 本壘
    #                         [288, 262],     # 一壘延伸綠土交界
    #                         [207, 208],     # 二壘延伸綠土交界
    #                         [126, 262]],    # 三壘延伸綠土交界
    #                         dtype=np.int32).reshape(-1, 1, 2)
    
    # === Steven Solution Config file ===
    test1_pt = np.array([   NAS_config['CAMERA_HOMOGRAPHY_POINTS'][0],    # 本壘
                            NAS_config['CAMERA_HOMOGRAPHY_POINTS'][1],   # 一壘延伸綠土交界
                            NAS_config['CAMERA_HOMOGRAPHY_POINTS'][2],   # 二壘延伸綠土交界
                            NAS_config['CAMERA_HOMOGRAPHY_POINTS'][3]],   # 三壘延伸綠土交界
                            dtype=np.float32).reshape(-1, 1, 2)

    test2_pt = np.array([   NAS_config['FIELD_HOMOGRAPHY_POINTS'][0],     # 本壘
                            NAS_config['FIELD_HOMOGRAPHY_POINTS'][1],     # 一壘延伸綠土交界
                            NAS_config['FIELD_HOMOGRAPHY_POINTS'][2],     # 二壘延伸綠土交界
                            NAS_config['FIELD_HOMOGRAPHY_POINTS'][3]],    # 三壘延伸綠土交界
                            dtype=np.int32).reshape(-1, 1, 2)

    H = cv2.findHomography(test1_pt, test2_pt)
    fs = cv2.FileStorage(PATH_OF_REFERENCE + "Homography_NAS_{}.yaml".format(NAS_config['NAS_INDEX']), flags=cv2.FILE_STORAGE_WRITE)
    fs.write(name="Homography", val=H[0])
    fs.release()
    print(H[0])
    print("Homography Saved!!!")

    for i in range(np.size(test1_pt,0)):
        cv2.circle(test1, tuple(test1_pt[i][0]), 10, (0, 0 ,255), -1)
        cv2.circle(test2, tuple(test2_pt[i][0]), 10, (0, 0 ,255), -1)
    cv2.namedWindow("camera_for_homography_NAS_{}.jpg".format(NAS_config['NAS_INDEX']), cv2.WINDOW_NORMAL)
    cv2.imshow("camera_for_homography_NAS_{}.jpg".format(NAS_config['NAS_INDEX']), test1)
    cv2.namedWindow("field_for_homography", cv2.WINDOW_NORMAL)
    cv2.imshow("field_for_homography", test2)
    cv2.waitKey(0)
    
    cv2.imwrite(PATH_OF_REFERENCE + "camera_for_homography_NAS_{}.jpg".format(NAS_config['NAS_INDEX']), test1)
    cv2.imwrite(PATH_OF_REFERENCE + "field_for_homography.jpg", test2)

def loadHomography(NAS_config):
    """Load Homography from Homography.yaml
    Returns:
        H:  numpy.array((3, 3), dtype='float64')
    """
    fs = cv2.FileStorage(PATH_OF_REFERENCE + "Homography_NAS_{}.yaml".format(NAS_config['NAS_INDEX']), cv2.FILE_STORAGE_READ)
    HNode = fs.getNode('Homography')
    H = HNode.mat()    
    return H

def setupDefensivePose(NAS_config):
    """Initial 9 defensive player.
    Returns:
        defPoseList:    [DefensivePosition] 
    """
    defPoseList = []

    for idx in range(len(NAS_config['DEFAULT_DEF_INDEX'])):
        dfp = DefensivePosition(NAS_config['DEFAULT_DEF_INDEX'][idx])
        dfp.playerID = -1
        defaultPlayer = Player()
        defaultPlayer.defaultCameraPose = np.array(NAS_config['DEFAULT_DEF_CAMERA_POSE'][idx])
        defaultPlayer.defaultPose = np.array(NAS_config['DEFAULT_DEF_POSE'][idx])
        defaultPlayer.detectedPose = np.array((-1, -1))
        defaultPlayer.detectedCameraPose = np.array((-1, -1))
        defaultPlayer.trackedPose = np.array((-1, -1))
        defaultPlayer.detectedID = -1
        defaultPlayer.detectedBox = [(-1, -1), (1, -1)]
        defaultPlayer.boundingBox = [-1, -1, -1, -1]
        dfp.player = defaultPlayer
        defPoseList.append(dfp)
        
    # # 本壘
    # dfp = DefensivePosition(0)
    # dfp.playerID = -1
    # defaultPlayer = Player()
    # defaultPlayer.defaultCameraPose = np.array((83, 1529))
    # defaultPlayer.defaultPose = np.array((207, 350))
    # defaultPlayer.detectedPose = np.array((-1, -1))
    # defaultPlayer.detectedCameraPose = np.array((-1, -1))
    # defaultPlayer.trackedPose = np.array((-1, -1))
    # defaultPlayer.detectedID = -1
    # defaultPlayer.detectedBox = [(-1, -1), (1, -1)]
    # defaultPlayer.boundingBox = [-1, -1, -1, -1]
    # dfp.player = defaultPlayer
    # defPoseList.append(dfp)

    # # 一壘
    # dfp = DefensivePosition(1)
    # dfp.playerID = -1
    # defaultPlayer = Player()
    # defaultPlayer.defaultCameraPose = np.array((2582, 1105))
    # defaultPlayer.defaultPose = np.array((265, 265))
    # defaultPlayer.detectedPose = np.array((-1, -1))
    # defaultPlayer.detectedCameraPose = np.array((-1, -1))
    # defaultPlayer.trackedPose = np.array((-1, -1))
    # defaultPlayer.detectedID = -1
    # defaultPlayer.detectedBox = [(-1, -1), (1, -1)]
    # defaultPlayer.boundingBox = [-1, -1, -1, -1]
    # dfp.player = defaultPlayer
    # defPoseList.append(dfp)

    # # 二壘
    # dfp = DefensivePosition(2)
    # dfp.playerID = -1
    # defaultPlayer = Player()
    # defaultPlayer.defaultCameraPose = np.array((2222, 926))
    # defaultPlayer.defaultPose = np.array((230, 230))
    # defaultPlayer.detectedPose = np.array((-1, -1))
    # defaultPlayer.detectedCameraPose = np.array((-1, -1))
    # defaultPlayer.trackedPose = np.array((-1, -1))
    # defaultPlayer.detectedID = -1
    # defaultPlayer.detectedBox = [(-1, -1), (1, -1)]
    # defaultPlayer.boundingBox = [-1, -1, -1, -1]
    # dfp.player = defaultPlayer
    # defPoseList.append(dfp)

    # # 游擊手
    # dfp = DefensivePosition(3)
    # dfp.playerID = -1
    # defaultPlayer = Player()
    # defaultPlayer.defaultCameraPose = np.array((1019, 911))
    # defaultPlayer.defaultPose = np.array((182, 231))
    # defaultPlayer.detectedPose = np.array((-1, -1))
    # defaultPlayer.detectedCameraPose = np.array((-1, -1))
    # defaultPlayer.trackedPose = np.array((-1, -1))
    # defaultPlayer.detectedID = -1
    # defaultPlayer.detectedBox = [(-1, -1), (1, -1)]
    # defaultPlayer.boundingBox = [-1, -1, -1, -1]
    # dfp.player = defaultPlayer
    # defPoseList.append(dfp)

    # # 三壘
    # dfp = DefensivePosition(4)
    # dfp.playerID = -1
    # defaultPlayer = Player()
    # defaultPlayer.defaultCameraPose = np.array((440, 985))
    # defaultPlayer.defaultPose = np.array((154, 264))
    # defaultPlayer.detectedPose = np.array((-1, -1))
    # defaultPlayer.detectedCameraPose = np.array((-1, -1))
    # defaultPlayer.trackedPose = np.array((-1, -1))
    # defaultPlayer.detectedID = -1
    # defaultPlayer.detectedBox = [(-1, -1), (1, -1)]
    # defaultPlayer.boundingBox = [-1, -1, -1, -1]
    # dfp.player = defaultPlayer
    # defPoseList.append(dfp)

    # # 右外野
    # dfp = DefensivePosition(5)
    # dfp.playerID = -1
    # defaultPlayer = Player()
    # defaultPlayer.defaultCameraPose = np.array((3593, 822))
    # defaultPlayer.defaultPose = np.array((348, 144))
    # defaultPlayer.detectedPose = np.array((-1, -1))
    # defaultPlayer.detectedCameraPose = np.array((-1, -1))
    # defaultPlayer.trackedPose = np.array((-1, -1))
    # defaultPlayer.detectedID = -1
    # defaultPlayer.detectedBox = [(-1, -1), (1, -1)]
    # defaultPlayer.boundingBox = [-1, -1, -1, -1]
    # dfp.player = defaultPlayer
    # defPoseList.append(dfp)

    # # 中外野
    # dfp = DefensivePosition(6)
    # dfp.playerID = -1
    # defaultPlayer = Player()
    # defaultPlayer.defaultCameraPose = np.array((2257, 741))
    # defaultPlayer.defaultPose = np.array((207, 83))
    # defaultPlayer.detectedPose = np.array((-1, -1))
    # defaultPlayer.detectedCameraPose = np.array((-1, -1))
    # defaultPlayer.trackedPose = np.array((-1, -1))
    # defaultPlayer.detectedID = -1
    # defaultPlayer.detectedBox = [(-1, -1), (1, -1)]
    # defaultPlayer.boundingBox = [-1, -1, -1, -1]
    # dfp.player = defaultPlayer
    # defPoseList.append(dfp)

    # # 左外野
    # dfp = DefensivePosition(7)
    # dfp.playerID = -1
    # defaultPlayer = Player()
    # defaultPlayer.defaultCameraPose = np.array((962, 768))
    # defaultPlayer.defaultPose = np.array((87, 145))
    # defaultPlayer.detectedPose = np.array((-1, -1))
    # defaultPlayer.detectedCameraPose = np.array((-1, -1))
    # defaultPlayer.trackedPose = np.array((-1, -1))
    # defaultPlayer.detectedID = -1
    # defaultPlayer.detectedBox = [(-1, -1), (1, -1)]
    # defaultPlayer.boundingBox = [-1, -1, -1, -1]
    # dfp.player = defaultPlayer
    # defPoseList.append(dfp)

    # # 投手
    # dfp = DefensivePosition(8)
    # dfp.playerID = -1
    # defaultPlayer = Player()
    # defaultPlayer.defaultCameraPose = np.array((1110, 1103))
    # defaultPlayer.defaultPose = np.array((207, 281))
    # defaultPlayer.detectedPose = np.array((-1, -1))
    # defaultPlayer.detectedCameraPose = np.array((-1, -1))
    # defaultPlayer.trackedPose = np.array((-1, -1))
    # defaultPlayer.detectedID = -1
    # defaultPlayer.detectedBox = [(-1, -1), (1, -1)]
    # defaultPlayer.boundingBox = [-1, -1, -1, -1]
    # dfp.player = defaultPlayer
    # defPoseList.append(dfp)

    return defPoseList

def setupOffensivePose(NAS_config):
    """Initial 1 offensive player.
    Returns:
        ofsPoseList:    [OffensivePosition] 
    """
    ofsPoseList = []

    for i in range(9):
        # 打擊者
        ofp = OffensivePosition(i)
        ofp.playerID = -1
        defaultPlayer = Player()
        defaultPlayer.defaultPose = np.array(NAS_config['DEFAULT_OFS_POSE'][0])
        defaultPlayer.detectedPose = np.array((-1, -1))
        defaultPlayer.detectedCameraPose = np.array((-1, -1))
        defaultPlayer.trackedPose = np.array((-1, -1))
        defaultPlayer.detectedID = -1
        defaultPlayer.detectedBox = [(-1, -1), (1, -1)]
        defaultPlayer.boundingBox = [-1, -1, -1, -1]
        ofp.player = defaultPlayer
        ofsPoseList.append(ofp)

    # # 打擊者
    # ofp = OffensivePosition(0)
    # ofp.playerID = -1
    # defaultPlayer = Player()
    # defaultPlayer.defaultPose = np.array((208, 330))
    # defaultPlayer.detectedPose = np.array((-1, -1))
    # defaultPlayer.detectedCameraPose = np.array((-1, -1))
    # defaultPlayer.trackedPose = np.array((-1, -1))
    # defaultPlayer.detectedID = -1
    # defaultPlayer.detectedBox = [(-1, -1), (1, -1)]
    # defaultPlayer.boundingBox = [-1, -1, -1, -1]
    # ofp.player = defaultPlayer
    # ofsPoseList.append(ofp)
    
    # # 一壘跑者
    # ofp = OffensivePosition(1)
    # ofp.playerID = -1
    # defaultPlayer = Player()
    # defaultPlayer.defaultPose = np.array((250, 275))
    # defaultPlayer.detectedPose = np.array((-1, -1))
    # defaultPlayer.detectedCameraPose = np.array((-1, -1))
    # defaultPlayer.trackedPose = np.array((-1, -1))
    # defaultPlayer.detectedID = -1
    # defaultPlayer.detectedBox = [(-1, -1), (1, -1)]
    # defaultPlayer.boundingBox = [-1, -1, -1, -1]
    # ofp.player = defaultPlayer
    # ofsPoseList.append(ofp)
    
    # # 二壘跑者
    # ofp = OffensivePosition(2)
    # ofp.playerID = -1
    # defaultPlayer = Player()
    # defaultPlayer.defaultPose = np.array((200, 245))
    # defaultPlayer.detectedPose = np.array((-1, -1))
    # defaultPlayer.detectedCameraPose = np.array((-1, -1))
    # defaultPlayer.trackedPose = np.array((-1, -1))
    # defaultPlayer.detectedID = -1
    # defaultPlayer.detectedBox = [(-1, -1), (1, -1)]
    # defaultPlayer.boundingBox = [-1, -1, -1, -1]
    # ofp.player = defaultPlayer
    # ofsPoseList.append(ofp)
    
    # # 三壘跑者
    # ofp = OffensivePosition(3)
    # ofp.playerID = -1
    # defaultPlayer = Player()
    # defaultPlayer.defaultPose = np.array((165, 290))
    # defaultPlayer.detectedPose = np.array((-1, -1))
    # defaultPlayer.detectedCameraPose = np.array((-1, -1))
    # defaultPlayer.trackedPose = np.array((-1, -1))
    # defaultPlayer.detectedID = -1
    # defaultPlayer.detectedBox = [(-1, -1), (1, -1)]
    # defaultPlayer.boundingBox = [-1, -1, -1, -1]
    # ofp.player = defaultPlayer
    # ofsPoseList.append(ofp)

    return ofsPoseList

def getTxtFilePath(filePath):
    """Change .jpg file to .txt file.
    Args:
        filePath:       aaa/xxxx.jpg
    Returns:
        txtFilePath:    aaa/xxxx.txt
    """
    txtFilePath = filePath.replace(".jpg", ".txt")
    return txtFilePath

def findPlayerPoseFromTxt(NAS_config, filePath, isInFieldROIImage):
    """Read player position from txt file.
    
    Args:
        filePath:           aaa/xxxx.txt
    Returns:
        tmpPlayerPostList:  [(x,y)]
    """
    tmpPlayerPostList = []
    txtFilePath = getTxtFilePath(filePath)

    with open(txtFilePath, 'r') as reader:
        for line in reader.readlines():
            tmpX, tmpY, leftTopX, leftTopY, boxW, boxH, player0umpire1 = line.strip().split(" ")
            # TODO: Modified this for offender and defender
            if is_in_field_ROI(isInFieldROIImage, (int(tmpX), int(leftTopY) + int(boxH) - 1), defender0Offender1=1):
                tmpPlayerPostList.append({'detectedPt': (int(tmpX), int(tmpY)), 
                                          'detectedBoxLeftTopPt': (int(leftTopX), int(leftTopY)), 
                                          'detectedBoxWH': (int(boxW), int(boxH)),
                                          'player0umpire1':int(player0umpire1)})

    return tmpPlayerPostList

def compute_player_info(NAS_config, yoloResults, isInFieldROIImage):
    """Compute player info from Lily and Nicole's yolo v3.
    
    Args:
        yoloResults:        ['c_x c_y x_min y_min w h class', ...]
    Returns:
        playerInfoList:     [{'detectedPt':(x, y), 'detectedBoxLeftTopPt':(x, y), 'detectedBoxWH':(w, h), 'player0umpire1':player0umpire1}]
    """
    
    playerInfoList = []
    for resultLine in yoloResults:
            tmpX, tmpY, leftTopX, leftTopY, boxW, boxH, player0umpire1 = resultLine.strip().split(" ")
            if is_in_field_ROI(isInFieldROIImage, (int(tmpX), int(leftTopY) + int(boxH) - 1), defender0Offender1=1):
                playerInfoList.append({'detectedPt': (int(tmpX), int(tmpY)), 
                                       'detectedBoxLeftTopPt': (int(leftTopX), int(leftTopY)), 
                                       'detectedBoxWH': (int(boxW), int(boxH)),
                                       'player0umpire1': int(player0umpire1)})
    return playerInfoList

def adjustPlayerPointForHomography(playerInfoList):
    """Adjust the player points with PLAYER_HEIGHT for apply homography.
    
    Args:
        playerInfoList:     [{'detectedPt', 'detectedBoxLeftTopPt', 'detectedBoxWH'}, ...]
    Returns:
        adjustedPlayerPt:   [(x,y)]
    """
    adjustedPlayerPt = [(playerInfo['detectedPt'][0], playerInfo['detectedBoxLeftTopPt'][1] + playerInfo['detectedBoxWH'][1] -1) for playerInfo in playerInfoList]
    return adjustedPlayerPt

def isInField(NAS_config, position, defender0Offender1=1):
    """Check if player's position is in the field.
    Args:
        position:           (x, y)
        defender0Offender1: 
    Returns:
        bool
    """
    if defender0Offender1 == 0:
        base        = NAS_config['DEF_IN_FIELD_BASE']
        leftBorder  = NAS_config['DEF_IN_FIELD_LEFTBORDER']
        rightBorder = NAS_config['DEF_IN_FIELD_RIGHTBORDER']
        base_circle = NAS_config['DEF_BASE_CIRCLE']
    else:
        base        = NAS_config['OFS_IN_FIELD_BASE']
        leftBorder  = NAS_config['OFS_IN_FIELD_LEFTBORDER']
        rightBorder = NAS_config['OFS_IN_FIELD_RIGHTBORDER']
        base_circle = NAS_config['OFS_BASE_CIRCLE']

    leftSlope = -(leftBorder[1] - base[1]) / (leftBorder[0] - base[0])
    rightSlope = -(rightBorder[1] - base[1]) / (rightBorder[0] - base[0])
    if (position[0] - base[0]) == 0:
        tmpSlope = -(position[1] - base[1]) / (0.00000001)
    else:
        tmpSlope = -(position[1] - base[1]) / (position[0] - base[0])

    # 在右邊線內
    if tmpSlope > rightSlope:
        return True
    # 在左邊線內
    elif tmpSlope < 0 and math.fabs(tmpSlope) > math.fabs(leftSlope):
        return True
    # 在打擊圈內
    else:
        if cv2.norm(np.array(position) - np.array(base)) < base_circle:
            return True
        else:
            return False
        
def draw_is_in_field_ROI(NAS_config, defender0Offender1=1):
    """Draw the ROI image for cheking is player's position in the field.
    Args:
        defender0Offender1: 
    Returns:
    """
    
    if defender0Offender1 == 0:
        BASE        = NAS_config['DEF_IN_FIELD_BASE']
        LEFTBORDER  = NAS_config['DEF_IN_FIELD_LEFTBORDER']
        RIGHTBORDER = NAS_config['DEF_IN_FIELD_RIGHTBORDER']
        BASE_CIRCLE = NAS_config['DEF_BASE_CIRCLE']
    else:
        BASE        = NAS_config['OFS_IN_FIELD_BASE']
        LEFTBORDER  = NAS_config['OFS_IN_FIELD_LEFTBORDER']
        RIGHTBORDER = NAS_config['OFS_IN_FIELD_RIGHTBORDER']
        BASE_CIRCLE = NAS_config['OFS_BASE_CIRCLE']

    cameraImage = cv2.imread("{}camera_NAS_{}.jpg".format(PATH_OF_REFERENCE, NAS_config['NAS_INDEX']))

    downBorder = cameraImage.shape[0] - 1
    rightBorder = cameraImage.shape[1] - 1
    
    lineLeft = line(BASE, LEFTBORDER)
    lineLeftTests = decide_intersection_boarder(BASE, LEFTBORDER, cameraImage)
    intersectionPointLeftTemp = intersection(lineLeft, lineLeftTests[1])
    if (intersectionPointLeftTemp[0] < 0 or
        intersectionPointLeftTemp[0] > rightBorder or
        intersectionPointLeftTemp[1] < 0 or
        intersectionPointLeftTemp[1] > downBorder):
        intersectionPointLeft = intersection(lineLeft, lineLeftTests[0])
    else:
        intersectionPointLeft = intersection(lineLeft, lineLeftTests[1])

    lineRight = line(BASE, RIGHTBORDER)
    lineRightTests = decide_intersection_boarder(BASE, RIGHTBORDER, cameraImage)
    intersectionPointRightTemp = intersection(lineRight, lineRightTests[1])
    if (intersectionPointRightTemp[0] < 0 or
        intersectionPointRightTemp[0] > rightBorder or
        intersectionPointRightTemp[1] < 0 or
        intersectionPointRightTemp[1] > downBorder):
        intersectionPointRight = intersection(lineRight, lineRightTests[0])
    else:
        intersectionPointRight = intersection(lineRight, lineRightTests[1])

    output = cameraImage.copy()
    output.fill(255)
    
    points = [BASE, intersectionPointRight]
    if (intersectionPointRight[0] == rightBorder):
        if (intersectionPointLeft[0] == rightBorder):
            points.append(intersectionPointLeft)
        elif (intersectionPointLeft[1] == 0):
            points.append([rightBorder, 0])
            points.append(intersectionPointLeft)
        elif (intersectionPointLeft[0] == 0):
            points.append([rightBorder, 0])
            points.append([0, 0])
            points.append(intersectionPointLeft)
        elif (intersectionPointLeft[1] == downBorder):
            points.append([rightBorder, 0])
            points.append([0, 0])
            points.append([0, downBorder])
        
    elif (intersectionPointRight[1] == 0):
        if (intersectionPointLeft[1] == 0):
            points.append(intersectionPointLeft)
        elif (intersectionPointLeft[0] == 0):
            points.append([0, 0])
            points.append(intersectionPointLeft)
        elif (intersectionPointLeft[1] == downBorder):
            points.append([0, 0])
            points.append([0, downBorder])
            points.append(intersectionPointLeft)
        elif (intersectionPointLeft[0] == rightBorder):
            points.append([0, 0])
            points.append([0, downBorder])
            points.append([rightBorder, downBorder])
            
    elif (intersectionPointRight[0] == 0):
        if (intersectionPointLeft[0] == 0):
            points.append(intersectionPointLeft)
        elif (intersectionPointLeft[1] == downBorder):
            points.append([0, downBorder])
            points.append(intersectionPointLeft)
        elif (intersectionPointLeft[0] == rightBorder):
            points.append([0, downBorder])
            points.append([rightBorder, downBorder])
            points.append(intersectionPointLeft)
        elif (intersectionPointLeft[1] == 0):
            points.append([0, downBorder])
            points.append([rightBorder, downBorder])
            points.append([rightBorder, 0])
            
    elif (intersectionPointRight[1] == downBorder):
        if (intersectionPointLeft[1] == downBorder):
            points.append(intersectionPointLeft)
        elif (intersectionPointLeft[0] == rightBorder):
            points.append([rightBorder, downBorder])
            points.append(intersectionPointLeft)
        elif (intersectionPointLeft[1] == 0):
            points.append([rightBorder, downBorder])
            points.append([rightBorder, 0])
            points.append(intersectionPointLeft)
        elif (intersectionPointLeft[0] == 0):
            points.append([rightBorder, downBorder])
            points.append([rightBorder, 0])
            points.append([0, 0])
        
    points.append(intersectionPointLeft)
    points = np.array(points)
    cv2.fillPoly(output, np.int32([points]), (0, 0, 0))
    cv2.circle(output,point2fToIntTuple(BASE), BASE_CIRCLE, (0, 0, 0), -1)
    
    cv2.imwrite(PATH_OF_REFERENCE + "camera_image_is_in_field_ROI_NAS_{}.jpg".format(NAS_config['NAS_INDEX']), output)

    return output

def load_isInFieldROIImage(NAS_config):
    """Load isInFieldROIImage
    Returns:
        isInFieldROIImage
    """
    
    isInFieldROIImage = cv2.imread(PATH_OF_REFERENCE + "camera_image_is_in_field_ROI_NAS_{}.jpg".format(NAS_config['NAS_INDEX']), cv2.IMREAD_GRAYSCALE)
    return isInFieldROIImage

def is_in_field_ROI(isInFieldROIImage, position, defender0Offender1=1):
    """Check if player's position is in the field.
    Args:
        isInFieldROIImage
        position:           (x, y)
        defender0Offender1: 
    Returns:
        bool
    """
    
    if defender0Offender1 == 0:
        checkImage = isInFieldROIImage
    else:
        checkImage = isInFieldROIImage

    if checkImage[position[1], position[0]] > 0:
        return False
    else:
        return True

def retrieveImagesFromFolder(path):
    """Get all images file name from the path."
    Args:
        path:           string
    Returns:
        filesInPath:    [string]
    """
    filesInPath = gb.glob(path)
    filesInPath.sort()
    return filesInPath

def drawDetectedPlayer(cameraImage, playerPoseList, color=(0, 0, 255), size=10, adjustedPlayerPoseList=[], playerIdxList=[]):
    """FOR TESTING: Draw the detected players on the camera image.
    Args:
        cameraImage:          
        playerPoseList: [position]
        color:
        size:
        adjustedPlayerPoseList:
        playerIdxList:
    Returns:
        output:  
    """
    output = cameraImage.copy()
    
    # If there's adjusted position for homography
    if len(adjustedPlayerPoseList) == len(playerPoseList) == len(playerIdxList):
        for i in range(len(playerPoseList)):
            cv2.line(output, point2fToIntTuple(playerPoseList[i]), point2fToIntTuple(adjustedPlayerPoseList[i]), (0, 200, 200), size - 10)
            cv2.circle(output, point2fToIntTuple(adjustedPlayerPoseList[i]), size - 5, (0, 200, 200), 3)
            cv2.putText(output, str("  %s" % playerIdxList[i]), point2fToIntTuple(adjustedPlayerPoseList[i]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 200), 4)
    else:
        i = 0
        for position in playerPoseList:
            cv2.putText(output, str("  %s" % i), point2fToIntTuple(position), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 4)
            i+=1
    
    for position in playerPoseList:
        cv2.circle(output, point2fToIntTuple(position), size, color, -1)
    
    
    return output

def drawDefaultDefensivePose(fieldImage, defPoseList):
    """FOR TESTING: Draw the default position of defensive players on the field image.
    Args:
        fieldImage:          np.array
        defPoseList:    [position]
    Returns:            np.array
        output:  
    """
    output = fieldImage.copy()
    for i in range(len(defPoseList)):
        if defPoseList[i].playerID == -1:
            cv2.circle(output, point2fToIntTuple(defPoseList[i].player.defaultPose), 5, (255, 0 ,0), -1)
        else:
            cv2.circle(output, point2fToIntTuple(defPoseList[i].player.defaultPose), 5, (0, 0 ,255), -1)
    return output

def drawDefaultOffensivePose(fieldImage, ofsPoseList):
    """FOR TESTING: Draw the default position of offensive players on the field image.
    Args:
        fieldImage:          np.array
        ofsPoseList:    [position]
    Returns:
        output:          np.array
    """
    output = fieldImage.copy()
    for i in range(len(ofsPoseList)):
        if ofsPoseList[i].playerID == -1:
            cv2.circle(output, point2fToIntTuple(ofsPoseList[i].player.defaultPose), 5, (0, 255 ,0), -1)
        else:
            cv2.circle(output, point2fToIntTuple(ofsPoseList[i].player.defaultPose), 5, (0, 255 ,255), -1)
    return output

def checkIfPlayersDuplicated(NAS_config, positionID, playerPt, projectedPt, referenceProjectedPose, defensiveDist, referenceCameraPt=[]):
    """Check if players are duplicated in field image. If gives referenceCameraPt, then check camera image.
    
    Args:
        positionID:                     int
        playerPt:                       [np.array((x, y))]
        projectedPt:                    [np.array((x, y))]
        referenceProjectedPose:         np.array((x, y))
        defensiveDist:                  float
        referenceCameraPt:              np.array((x, y))
    Returns:
        potentialIdx:       [int]
        potentialPlayer:    [np.array((x, y))]
        potentialProjected: [np.array((x, y))]
    """
    # TODO: Different DUPLICATED_CAMERA_THRESHOLD for different positions
    FIELD_COEFFICIENT = NAS_config['DUPLICATED_FIELD_COEFFICIENT']
    CAMERA_THRESHOLD = NAS_config['DUPLICATED_CAMERA_THRESHOLD']
    potentialIdx = []
    potentialPlayer = []
    potentialProjected = []
    if len(referenceCameraPt) != 0 and referenceCameraPt[0] > -1 and positionID <= 4:
        for i in range(len(projectedPt)):
            distField = cv2.norm(np.array(projectedPt[i]) - referenceProjectedPose)
            distCamera = cv2.norm(np.array(playerPt[i]) - referenceCameraPt)
            if distField < defensiveDist * FIELD_COEFFICIENT or distCamera < CAMERA_THRESHOLD:
                potentialIdx.append(i)
                potentialPlayer.append(playerPt[i])
                potentialProjected.append(projectedPt[i])
    else:
        for i in range(len(projectedPt)):
            # if positionID == 0:
            #     dist = cv2.norm(np.array(projectedPt[i]) - base)
            # else:
            #     dist = cv2.norm(np.array(projectedPt[i]) - referenceProjectedPose)
            dist = cv2.norm(np.array(projectedPt[i]) - referenceProjectedPose)
            if dist < defensiveDist * FIELD_COEFFICIENT:
                potentialIdx.append(i)
                potentialPlayer.append(playerPt[i])
                potentialProjected.append(projectedPt[i])

    return potentialIdx, potentialPlayer, potentialProjected

def findDefenders(NAS_config, cameraImage, defPoseList, playerPt, projectedPt, currentDefendersOnField, detectedBox=[], lastPitcherROI=None, player0umpire1=[]):
    """Main function for finding defenders.
    Args:
        cameraImage:                np.array
        defPoseList:                [position]
        playerPt:                   
        projectedPt:                
        currentDefendersOnField:
        detectedBox:                [ [(topLeftX, topLeftY), (W, H)] , ... ]   
        lastPitcherROI:             np.array 
    Returns:
        defPoseList:
        defenderTeamList:           [boolean]
        hueCompareResultList:       [boolean]
    """
    print("Defender Size: ", len(currentDefendersOnField))

    DEF_HUE_THRESHOLD = NAS_config['DEF_HUE_THRESHOLD']
    
    # Check if has pitcher info for team classifier
    hueCompareResultList = [True] * len(playerPt)
    if lastPitcherROI is not None:
        # Here hueCompareResultList is come from hue compare results
        hueCompareResultList = get_defender_list_with_hue_compare(cameraImage, lastPitcherROI, detectedBox, threshold=DEF_HUE_THRESHOLD)

    # For 9 position
    for i in range(len(defPoseList)):
        tmpDef = defPoseList[i]
        # The positions not filled yet
        if tmpDef.playerID == -1:
            min_idx = -1
            min_dist = 100000
            # Filled with YOLO detected position
            for j in range(len(projectedPt)):
                tmpDist = cv2.norm(tmpDef.player.defaultPose - np.array(projectedPt[j]))
                # Find closest player
                if tmpDist < min_dist:
                    # 外野不管
                    if (tmpDef.positionID == 0 or 
                        tmpDef.positionID == 1 or 
                        tmpDef.positionID == 2 or 
                        tmpDef.positionID == 3 or 
                        tmpDef.positionID == 4 or 
                        tmpDef.positionID == 8):
                        if hueCompareResultList[j] and player0umpire1[j] == 0:
                            min_idx = j
                            min_dist = tmpDist
                    # For outfielders no need to consider team_classifier
                    else:
                        min_idx = j
                        min_dist = tmpDist

            # Make sure not mismatched defensive position
            defDuplicatedIdx, defDuplicatedPt, defDuplicatedProjectedPt = checkIfPlayersDuplicated(NAS_config, tmpDef.positionID, playerPt, projectedPt, tmpDef.player.defaultPose, float(min_dist), tmpDef.player.defaultCameraPose)
            # Filter out umpires
            defDuplicatedPt = [defDuplicatedPt[idx] for x, idx in zip(defDuplicatedIdx, range(len(defDuplicatedIdx))) if player0umpire1[x] == 0]
            defDuplicatedProjectedPt = [defDuplicatedProjectedPt[idx] for x, idx in zip(defDuplicatedIdx, range(len(defDuplicatedIdx))) if player0umpire1[x] == 0]
            defDuplicatedIdx = [x for x in defDuplicatedIdx if player0umpire1[x] == 0]

            # TODO: Solve other cases which are not happening in base area)
            if len(defDuplicatedPt) == 1:
                # New a player to defend the position
                newPlayer = Player()
                # TODO: (not hurry) 改成球員背號
                newPlayer.id = len(currentDefendersOnField)
                # newPlayer.tracker = cv2.TrackerKCF_create()
                newPlayer.detectedID = defDuplicatedIdx[0]
                newPlayer.detectedBox = detectedBox[defDuplicatedIdx[0]]
                newPlayer.boundingBox = (detectedBox[defDuplicatedIdx[0]][0][0], 
                                            detectedBox[defDuplicatedIdx[0]][0][1], 
                                            detectedBox[defDuplicatedIdx[0]][1][0], 
                                            detectedBox[defDuplicatedIdx[0]][1][1])
                # newPlayer.tracker.init(cameraImage, newPlayer.boundingBox)
                newPlayer.defaultPose = tmpDef.player.defaultPose
                newPlayer.detectedPose = np.array(defDuplicatedProjectedPt[0])
                newPlayer.detectedCameraPose = np.array(defDuplicatedPt[0])
                # Add player to defender list
                currentDefendersOnField.append(newPlayer)
                # Update defensive position info
                tmpDef.playerID = newPlayer.id
                tmpDef.player = newPlayer

                defPoseList[i] = tmpDef
            else:
                # 打擊區內有三個人 裁判 捕手 打擊者 裁判已經濾掉
                if tmpDef.positionID == 0:
                    if len(defDuplicatedPt) == 2:
                        # 兩個人的話，離投手較遠的是補手
                        pitcherIdx = NAS_config['DEFAULT_DEF_INDEX'].index(8)
                        pitcherPt = defPoseList[pitcherIdx].player.defaultPose

                        aDist = cv2.norm(np.array(defDuplicatedPt[0]) - pitcherPt)
                        bDist = cv2.norm(np.array(defDuplicatedPt[1]) - pitcherPt)
                        if aDist < bDist:
                            idx = 1
                        else:
                            idx = 0

                        newPlayer = Player()
                        # TODO: (not hurry) 改成球員背號
                        newPlayer.id = len(currentDefendersOnField)
                        # newPlayer.tracker = cv2.TrackerKCF_create()
                        newPlayer.detectedID = defDuplicatedIdx[idx]
                        newPlayer.detectedBox = detectedBox[defDuplicatedIdx[idx]]
                        newPlayer.boundingBox = (detectedBox[defDuplicatedIdx[idx]][0][0], 
                                                 detectedBox[defDuplicatedIdx[idx]][0][1], 
                                                 detectedBox[defDuplicatedIdx[idx]][1][0], 
                                                 detectedBox[defDuplicatedIdx[idx]][1][1])
                        # newPlayer.tracker.init(cameraImage, newPlayer.boundingBox)
                        newPlayer.defaultPose = tmpDef.player.defaultPose
                        newPlayer.detectedPose = np.array(defDuplicatedProjectedPt[idx])
                        newPlayer.detectedCameraPose = np.array(defDuplicatedPt[idx])
                        # Add player to defender list
                        currentDefendersOnField.append(newPlayer)
                        # Update defensive position info
                        tmpDef.playerID = newPlayer.id
                        tmpDef.player = newPlayer

                        defPoseList[i] = tmpDef

                    elif len(defDuplicatedPt) == 3:
                        # 三個人的話，中間是捕手，代表裁判沒被分類出來，才有三個球員
                        # TODO: NEED TO BE CHECKED
                        pitcherIdx = NAS_config['DEFAULT_DEF_INDEX'].index(8)
                        pitcherPt = defPoseList[pitcherIdx].player.defaultPose

                        aDist = cv2.norm(np.array(defDuplicatedProjectedPt[0]) - pitcherPt)
                        bDist = cv2.norm(np.array(defDuplicatedProjectedPt[1]) - pitcherPt)
                        cDist = cv2.norm(np.array(defDuplicatedProjectedPt[2]) - pitcherPt)
                        if ((aDist < bDist and bDist < cDist) or 
                            (cDist < bDist and bDist < aDist)):
                            idx = 1
                        elif ((bDist < cDist and cDist < aDist) or 
                              (aDist < cDist and cDist < bDist)):
                            idx = 2
                        elif ((cDist < aDist and aDist < bDist) or 
                              (bDist < aDist and aDist < cDist)):
                            idx = 0
                            

                        newPlayer = Player()
                        # TODO: (not hurry) 改成球員背號
                        newPlayer.id = len(currentDefendersOnField)
                        # newPlayer.tracker = cv2.TrackerKCF_create()
                        newPlayer.detectedID = defDuplicatedIdx[idx]
                        newPlayer.detectedBox = detectedBox[defDuplicatedIdx[idx]]
                        newPlayer.boundingBox = (detectedBox[defDuplicatedIdx[idx]][0][0], 
                                                 detectedBox[defDuplicatedIdx[idx]][0][1], 
                                                 detectedBox[defDuplicatedIdx[idx]][1][0], 
                                                 detectedBox[defDuplicatedIdx[idx]][1][1])
                        # newPlayer.tracker.init(cameraImage, newPlayer.boundingBox)
                        newPlayer.defaultPose = tmpDef.player.defaultPose
                        newPlayer.detectedPose = np.array(defDuplicatedProjectedPt[idx])
                        newPlayer.detectedCameraPose = np.array(defDuplicatedPt[idx])
                        # Add player to defender list
                        currentDefendersOnField.append(newPlayer)
                        # Update defensive position info
                        tmpDef.playerID = newPlayer.id
                        tmpDef.player = newPlayer

                        defPoseList[i] = tmpDef
                    
                    else:
                        tmpDef.player.detectedID = -1
                        defPoseList[i] = tmpDef
                        print("Unexpected!!!")
                # 一二三壘＆游擊手
                elif (tmpDef.positionID == 1 or 
                      tmpDef.positionID == 2 or 
                      tmpDef.positionID == 3 or 
                      tmpDef.positionID == 4):
                    if len(defDuplicatedPt) == 2:
                        # 兩個人的話，靠近預設點的是防守方
                        defaultPose = defPoseList[i].player.defaultPose

                        aDist = cv2.norm(np.array(defDuplicatedPt[0]) - defaultPose)
                        bDist = cv2.norm(np.array(defDuplicatedPt[1]) - defaultPose)
                        idx = -1
                        if aDist < bDist and aDist < 60:
                            idx = 1
                        elif bDist < aDist and bDist < 60:
                            idx = 0
                        if idx > -1:
                            newPlayer = Player()
                            # TODO: (not hurry) 改成球員背號
                            newPlayer.id = len(currentDefendersOnField)
                            # newPlayer.tracker = cv2.TrackerKCF_create()
                            newPlayer.detectedID = defDuplicatedIdx[idx]
                            newPlayer.detectedBox = detectedBox[defDuplicatedIdx[idx]]
                            newPlayer.boundingBox = (detectedBox[defDuplicatedIdx[idx]][0][0], 
                                                    detectedBox[defDuplicatedIdx[idx]][0][1], 
                                                    detectedBox[defDuplicatedIdx[idx]][1][0], 
                                                    detectedBox[defDuplicatedIdx[idx]][1][1])
                            # newPlayer.tracker.init(cameraImage, newPlayer.boundingBox)
                            newPlayer.defaultPose = tmpDef.player.defaultPose
                            newPlayer.detectedPose = np.array(defDuplicatedProjectedPt[idx])
                            newPlayer.detectedCameraPose = np.array(defDuplicatedPt[idx])
                            # Add player to defender list
                            currentDefendersOnField.append(newPlayer)
                            # Update defensive position info
                            tmpDef.playerID = newPlayer.id
                            tmpDef.player = newPlayer
                            defPoseList[i] = tmpDef
                        else:
                            tmpDef.player.detectedID = -1
                            defPoseList[i] = tmpDef
                            
                    # 如果內野剩一個沒配對到，直接找最近的
                    unMatchedCount = 0
                    for idx in [0, 1, 2, 3, 4]:
                        tmpIdx = NAS_config['DEFAULT_DEF_INDEX'].index(idx)
                        if defPoseList[tmpIdx].playerID == -1:
                            unMatchedCount += 1
                    if unMatchedCount == 1:
                        # New a player to defend the position
                        newPlayer = Player()
                        # TODO: (not hurry) 改成球員背號
                        newPlayer.id = len(currentDefendersOnField)
                        # newPlayer.tracker = cv2.TrackerKCF_create()
                        newPlayer.detectedID = min_idx
                        newPlayer.detectedBox = detectedBox[min_idx]
                        newPlayer.boundingBox = (detectedBox[min_idx][0][0], 
                                                detectedBox[min_idx][0][1], 
                                                detectedBox[min_idx][1][0], 
                                                detectedBox[min_idx][1][1])
                        # newPlayer.tracker.init(cameraImage, newPlayer.boundingBox)
                        newPlayer.defaultPose = tmpDef.player.defaultPose
                        newPlayer.detectedPose = np.array(projectedPt[min_idx])
                        newPlayer.detectedCameraPose = np.array(playerPt[min_idx])
                        # Add player to defender list
                        currentDefendersOnField.append(newPlayer)
                        # Update defensive position info
                        tmpDef.playerID = newPlayer.id
                        tmpDef.player = newPlayer
                        defPoseList[i] = tmpDef

        # playerID ~= -1
        else:
            # Last detected object
            min_idx = -1
            min_dist = 100000
            
            for j in range(len(projectedPt)):
                tmpDist = cv2.norm(tmpDef.player.detectedPose - np.array(projectedPt[j]))

                # Find closest player
                if tmpDist < min_dist:
                    if (tmpDef.positionID == 1 or 
                        tmpDef.positionID == 2 or 
                        tmpDef.positionID == 3 or 
                        tmpDef.positionID == 4 or 
                        tmpDef.positionID == 8):
                        if hueCompareResultList[j] and player0umpire1[j] == 0:
                            min_idx = j
                            min_dist = tmpDist
                    # For outfielders no need to consider team_classifier
                    else:
                        min_idx = j
                        min_dist = tmpDist
            
            # Check if duplicated
            if tmpDef.positionID == 0:
                defDuplicatedIdx, defDuplicatedPt, defDuplicatedProjectedPt = checkIfPlayersDuplicated(NAS_config, tmpDef.positionID, playerPt, projectedPt, tmpDef.player.defaultPose, float(min_dist), tmpDef.player.detectedCameraPose)
            else:
                defDuplicatedIdx, defDuplicatedPt, defDuplicatedProjectedPt = checkIfPlayersDuplicated(NAS_config, tmpDef.positionID, playerPt, projectedPt, tmpDef.player.detectedPose, float(min_dist), tmpDef.player.detectedCameraPose)
            # Filter out umpires
            defDuplicatedPt = [defDuplicatedPt[idx] for x, idx in zip(defDuplicatedIdx, range(len(defDuplicatedIdx))) if player0umpire1[x] == 0]
            defDuplicatedProjectedPt = [defDuplicatedProjectedPt[idx] for x, idx in zip(defDuplicatedIdx, range(len(defDuplicatedIdx))) if player0umpire1[x] == 0]
            defDuplicatedIdx = [x for x in defDuplicatedIdx if player0umpire1[x] == 0]
            if len(defDuplicatedIdx) > 1:
                print("防守重疊超過一人的情況 - 位置： ", tmpDef.positionID, " Potential Players Count: ", len(defDuplicatedIdx))
                for dupIdx in defDuplicatedIdx:
                    print("     重疊編號： ", dupIdx)
            
            if len(defDuplicatedPt) == 0:
                tmpDef.player.detectedID = -1
                defPoseList[i] = tmpDef
            
            elif len(defDuplicatedPt) == 1:
                isMovePossible = checkIsPlayerMomentPossible(NAS_config, tmpDef.player.detectedPose, np.array(defDuplicatedProjectedPt[0]))

                if isMovePossible:
                    # Update defensive position info
                    tmpDef.player.detectedPose = np.array(defDuplicatedProjectedPt[0])
                    tmpDef.player.detectedCameraPose = np.array(defDuplicatedPt[0])
                    tmpDef.player.detectedID = defDuplicatedIdx[0]
                    tmpDef.player.detectedBox = detectedBox[defDuplicatedIdx[0]]
                else:
                    tmpDef.player.detectedID = -1
                
                # # Opencv Tracker
                # isTrackerOk, tmpDef.player.boundingBox = tmpDef.player.tracker.update(cameraImage)
                # if isTrackerOk:
                #     tmpDef.player.trackedPose = np.array([tmpDef.player.boundingBox[0] + tmpDef.player.boundingBox[2] / 2, 
                #                                         tmpDef.player.boundingBox[1] + tmpDef.player.boundingBox[3] / 2])
                # else:
                #     # TODO:
                #     tmpDef.player.trackedPose = np.zeros(2)
                defPoseList[i] = tmpDef
            
            elif len(defDuplicatedProjectedPt) == 2:
                if tmpDef.positionID == 0:
                    # 兩個人的話，靠近投手的是打擊者
                    pitcherIdx = NAS_config['DEFAULT_DEF_INDEX'].index(8)
                    pitcherPt = defPoseList[pitcherIdx].player.defaultPose

                    aDist = cv2.norm(np.array(defDuplicatedProjectedPt[0]) - pitcherPt)
                    bDist = cv2.norm(np.array(defDuplicatedProjectedPt[1]) - pitcherPt)
                    if aDist < bDist:
                        idx = 1
                    else:
                        idx = 0

                    newPlayer = Player()
                    # TODO: (not hurry) 改成球員背號
                    newPlayer.id = len(currentDefendersOnField)
                    # newPlayer.tracker = cv2.TrackerKCF_create()
                    newPlayer.detectedID = defDuplicatedIdx[idx]
                    newPlayer.detectedBox = detectedBox[defDuplicatedIdx[idx]]
                    newPlayer.boundingBox = (detectedBox[defDuplicatedIdx[idx]][0][0], 
                                             detectedBox[defDuplicatedIdx[idx]][0][1], 
                                             detectedBox[defDuplicatedIdx[idx]][1][0], 
                                             detectedBox[defDuplicatedIdx[idx]][1][1])
                    # newPlayer.tracker.init(cameraImage, newPlayer.boundingBox)
                    newPlayer.defaultPose = tmpDef.player.defaultPose
                    newPlayer.detectedPose = np.array(defDuplicatedProjectedPt[idx])
                    newPlayer.detectedCameraPose = np.array(defDuplicatedPt[idx])
                    # Add player to defender list
                    currentDefendersOnField.append(newPlayer)
                    # Update defensive position info
                    tmpDef.playerID = newPlayer.id
                    tmpDef.player = newPlayer

                    defPoseList[i] = tmpDef
                
                else:
                    tmpDef.player.detectedID = -1
                    defPoseList[i] = tmpDef
            
            elif len(defDuplicatedProjectedPt) == 3:
                if tmpDef.positionID == 0:
                    # 三個人的話，中間是捕手，代表裁判沒被分類出來，或是多框，才有三個球員
                    # TODO: NEED TO BE CHECKED
                    pitcherIdx = NAS_config['DEFAULT_DEF_INDEX'].index(8)
                    pitcherPt = defPoseList[pitcherIdx].player.defaultPose

                    aDist = cv2.norm(np.array(defDuplicatedProjectedPt[0]) - pitcherPt)
                    bDist = cv2.norm(np.array(defDuplicatedProjectedPt[1]) - pitcherPt)
                    cDist = cv2.norm(np.array(defDuplicatedProjectedPt[2]) - pitcherPt)
                    if (aDist < bDist and bDist < cDist) or (cDist < bDist and bDist < aDist):
                        idx = 1
                    elif (bDist < cDist and cDist < aDist) or (aDist < cDist and cDist < bDist):
                        idx = 2
                    elif (cDist < aDist and aDist < bDist) or (bDist < aDist and aDist < cDist):
                        idx = 0
                        
                    newPlayer = Player()
                    # TODO: (not hurry) 改成球員背號
                    newPlayer.id = len(currentDefendersOnField)
                    # newPlayer.tracker = cv2.TrackerKCF_create()
                    newPlayer.detectedID = defDuplicatedIdx[idx]
                    newPlayer.detectedBox = detectedBox[defDuplicatedIdx[idx]]
                    newPlayer.boundingBox = (detectedBox[defDuplicatedIdx[idx]][0][0], 
                                                detectedBox[defDuplicatedIdx[idx]][0][1], 
                                                detectedBox[defDuplicatedIdx[idx]][1][0], 
                                                detectedBox[defDuplicatedIdx[idx]][1][1])
                    # newPlayer.tracker.init(cameraImage, newPlayer.boundingBox)
                    newPlayer.defaultPose = tmpDef.player.defaultPose
                    newPlayer.detectedPose = np.array(defDuplicatedProjectedPt[idx])
                    newPlayer.detectedCameraPose = np.array(defDuplicatedPt[idx])
                    # Add player to defender list
                    currentDefendersOnField.append(newPlayer)
                    # Update defensive position info
                    tmpDef.playerID = newPlayer.id
                    tmpDef.player = newPlayer

                    defPoseList[i] = tmpDef
                    
                else:
                    tmpDef.player.detectedID = -1
                    defPoseList[i] = tmpDef
            else:
                tmpDef.player.detectedID = -1
                defPoseList[i] = tmpDef
                pass

                    
    # 針對二壘手與游擊手做校正
    def2BIdx = NAS_config['DEFAULT_DEF_INDEX'].index(2)
    defSSIdx = NAS_config['DEFAULT_DEF_INDEX'].index(3)
    DISTANCE_CAMERA_THRESHOLD = NAS_config['DISTANCE_CAMERA_THRESHOLD']
    if (defPoseList[def2BIdx].player.detectedID == defPoseList[defSSIdx].player.detectedID and 
        defPoseList[def2BIdx].player.detectedID != -1):
        min_idx_a = -1
        min_idx_b = -1
        min_dist = 100000
        min_dist_a_to_2B = 100000
        min_dist_b_to_2B = 100000
        min_dist_a_to_SS = 100000
        min_dist_b_to_SS = 100000
        # 九個位置中已經偵測到的框
        positionsWithDetectedID = [defPose.player.detectedID for defPose in defPoseList]
        # 找最靠近兩個防守位置的框
        for j in range(len(projectedPt)):
            # 如果框是已經框了二壘手或游擊手或還沒指派，那就加入考慮
            if j in positionsWithDetectedID:
                if (positionsWithDetectedID.index(j) == def2BIdx or 
                    positionsWithDetectedID.index(j) == defSSIdx):
                    tmpDist_to_2B = cv2.norm(defPoseList[def2BIdx].player.defaultPose - np.array(projectedPt[j]))
                    tmpDist_to_SS = cv2.norm(defPoseList[defSSIdx].player.defaultPose - np.array(projectedPt[j]))
                    tmpDist = tmpDist_to_2B + tmpDist_to_SS
                    # Find closest player
                    if (tmpDist < min_dist and 
                        hueCompareResultList[j] and 
                        tmpDist_to_2B < DISTANCE_CAMERA_THRESHOLD and 
                        tmpDist_to_SS < DISTANCE_CAMERA_THRESHOLD):
                        min_idx_a = j
                        min_dist_a_to_2B = tmpDist_to_2B
                        min_dist_a_to_SS = tmpDist_to_SS
            else:
                tmpDist_to_2B = cv2.norm(defPoseList[def2BIdx].player.defaultPose - np.array(projectedPt[j]))
                tmpDist_to_SS = cv2.norm(defPoseList[defSSIdx].player.defaultPose - np.array(projectedPt[j]))
                tmpDist = tmpDist_to_2B + tmpDist_to_SS
                # Find closest player
                if (tmpDist < min_dist and 
                    hueCompareResultList[j] and 
                    tmpDist_to_2B < DISTANCE_CAMERA_THRESHOLD and 
                    tmpDist_to_SS < DISTANCE_CAMERA_THRESHOLD):
                    min_idx_a = j
                    min_dist_a_to_2B = tmpDist_to_2B
                    min_dist_a_to_SS = tmpDist_to_SS
        # 找第二靠近兩個防守位置的框
        min_dist = 100000 
        for j in range(len(projectedPt)):
            # 找第二近的
            if j != min_idx_a:
                # 如果框是已經框了二壘手或游擊手或還沒指派，那就加入考慮
                if j in positionsWithDetectedID:
                    if (positionsWithDetectedID.index(j) == def2BIdx or 
                        positionsWithDetectedID.index(j) == defSSIdx):
                        tmpDist_to_2B = cv2.norm(defPoseList[def2BIdx].player.defaultPose - np.array(projectedPt[j]))
                        tmpDist_to_SS = cv2.norm(defPoseList[defSSIdx].player.defaultPose - np.array(projectedPt[j]))
                        tmpDist = tmpDist_to_2B + tmpDist_to_SS
                        # Find closest player
                        if (tmpDist < min_dist and 
                            hueCompareResultList[j] and 
                            tmpDist_to_2B < DISTANCE_CAMERA_THRESHOLD and 
                            tmpDist_to_SS < DISTANCE_CAMERA_THRESHOLD):
                            min_idx_b = j
                            min_dist_b_to_2B = tmpDist_to_2B
                            min_dist_b_to_SS = tmpDist_to_SS
                else:
                    tmpDist_to_2B = cv2.norm(defPoseList[def2BIdx].player.defaultPose - np.array(projectedPt[j]))
                    tmpDist_to_SS = cv2.norm(defPoseList[3].player.defaultPose - np.array(projectedPt[j]))
                    tmpDist = tmpDist_to_2B + tmpDist_to_SS
                    # Find closest player
                    if (tmpDist < min_dist and 
                        hueCompareResultList[j] and 
                        tmpDist_to_2B < DISTANCE_CAMERA_THRESHOLD and 
                        tmpDist_to_SS < DISTANCE_CAMERA_THRESHOLD):
                        min_idx_b = j
                        min_dist_b_to_2B = tmpDist_to_2B
                        min_dist_b_to_SS = tmpDist_to_SS
        
        # 如果真的有兩個可以考慮的框
        if min_idx_a != -1 and min_idx_b != -1:
            print("min_idx_a:", min_idx_a, " min_idx_b:", min_idx_a)
            unSetPositions = [2, 3]
            unSetPlayers = [{"player":min_idx_a, "distances":[min_dist_a_to_2B, min_dist_a_to_SS]}, 
                            {"player":min_idx_b, "distances":[min_dist_b_to_2B, min_dist_b_to_SS]}]
            matchedPlayerForPositions = PPMatch.calc_player_position_match(unSetPositions, unSetPlayers)
            # 校正
            for matchedPlayerForPosition in matchedPlayerForPositions:
                positionIdx = NAS_config['DEFAULT_DEF_INDEX'].index(matchedPlayerForPosition["position"])
                tmpDef = defPoseList[positionIdx]
                # New a player to defend the position
                matchedPlayerIdx = matchedPlayerForPosition["player"]
                newPlayer = Player()
                # TODO: (not hurry) 改成球員背號
                newPlayer.id = len(currentDefendersOnField)
                # newPlayer.tracker = cv2.TrackerKCF_create()
                newPlayer.detectedID = matchedPlayerIdx
                newPlayer.detectedBox = detectedBox[matchedPlayerIdx]
                newPlayer.boundingBox = (detectedBox[matchedPlayerIdx][0][0], 
                                         detectedBox[matchedPlayerIdx][0][1], 
                                         detectedBox[matchedPlayerIdx][1][0], 
                                         detectedBox[matchedPlayerIdx][1][1])
                # newPlayer.tracker.init(cameraImage, newPlayer.boundingBox)
                newPlayer.defaultPose = tmpDef.player.defaultPose
                newPlayer.detectedPose = np.array(projectedPt[matchedPlayerIdx])
                newPlayer.detectedCameraPose = np.array(playerPt[matchedPlayerIdx])
                # Add player to defender list
                currentDefendersOnField.append(newPlayer)
                # Update defensive position info
                tmpDef.playerID = newPlayer.id
                tmpDef.player = newPlayer

                defPoseList[positionIdx] = tmpDef
            
    
    # TODO: 試著自動校正同一人有兩個守備位置
    # List all defenders' detected box ID
    positionsWithDetectedID = []
    for tmpDef in defPoseList:
        positionsWithDetectedID.append(tmpDef.player.detectedID)
    # Check if one player has two positions, and reset farther ones
    occurrencesOfID = Counter(positionsWithDetectedID)
    for ID, count in occurrencesOfID.items():
        if count > 1 and ID > -1:
            min_idx = -1
            min_dist = 100000
            # Find closest default defender for detected box
            for positionIdx in range(len(positionsWithDetectedID)):
                if positionsWithDetectedID[positionIdx] == ID:
                    tmpDist = cv2.norm(defPoseList[positionIdx].player.defaultPose - projectedPt[ID])
                    if tmpDist < min_dist:
                        min_idx = positionIdx
                        min_dist = tmpDist
            # Reset others
            for positionIdx in range(len(positionsWithDetectedID)):
                if positionsWithDetectedID[positionIdx] == ID and positionIdx != min_idx:
                    defPoseList[positionIdx].playerID = -1
        
        
    # Make defenderTeamList to final results
    defenderTeamList = []
    for defenderIdx in range(len(hueCompareResultList)):
        defenderTeamList.append(False)
    for defender in defPoseList:
        #!FIXME: Some problem here!!!
        if defender.player.detectedID > -1:
            try:
                defenderTeamList[defender.player.detectedID] = True
            except Exception as error:
                # print("Unexpected error:", sys.exc_info()[0])
                print("defenderTeamList[defender.player.detectedID] error!!!! defenderTeamList len: ", len(defenderTeamList), " detectedID: ", defender.player.detectedID)
                print("Error: ", error)
                
            
            
    return defPoseList, defenderTeamList, hueCompareResultList

def findOffenders(NAS_config, cameraImage, isInFieldROIImage, ofsPoseList, playerPt, projectedPt, currentOffendersOnField, detectedBox=[], lastPitcherROI=None, player0umpire1=[], defenderTeamList=[], isResetPressed=False):
    """Main function for finding offenders.

    Args:
        cameraImage:                np.array
        ofsPoseList:                [position]
        playerPt:                   
        projectedPt:                
        currentOffendersOnField:    
        detectedBox:
        lastPitcherROI:
        defenderTeamList:
    Returns:
        ofsPoseList:
        offenderTeamList
    """
    print("Offender Size: ", len(currentOffendersOnField))
    DEFAULT_OFS_POSE = NAS_config['DEFAULT_OFS_POSE']
    MOVEMENT_THRESHOLD = NAS_config['MOVEMENT_THRESHOLD']
    # If pressed reset, check three bases first
    if isResetPressed:
        ofsIdx = 0
        ofsBaseDefaultPose = []
        # 三壘跑者
        ofsBaseDefaultPose.append(np.array(DEFAULT_OFS_POSE[3]))
        # 二壘跑者
        ofsBaseDefaultPose.append(np.array(DEFAULT_OFS_POSE[2]))
        # 一壘跑者
        ofsBaseDefaultPose.append(np.array(DEFAULT_OFS_POSE[1]))
        for i in range(len(ofsBaseDefaultPose)):
            min_idx = -1
            min_dist = 100000
            for j in range(len(projectedPt)):
                tmpDist = cv2.norm(ofsBaseDefaultPose[i] - np.array(projectedPt[j]))
                # Find closest player
                if tmpDist < min_dist:
                    if defenderTeamList[j] == False and player0umpire1[j] == 0:
                        min_idx = j
                        min_dist = tmpDist
            # 確保進攻位置不會誤判
            ofsDuplicatedIdx, ofsDuplicatedPt, ofsDuplicatedProjectedPt = checkIfPlayersDuplicated(NAS_config, i, playerPt, projectedPt, ofsBaseDefaultPose[i], min_dist)
            # Filter out defenders & umpires
            ofsDuplicatedPt = [ofsDuplicatedPt[idx] for x, idx in zip(ofsDuplicatedIdx, range(len(ofsDuplicatedIdx))) if defenderTeamList[x] == False and player0umpire1[x] == 0]
            ofsDuplicatedProjectedPt = [ofsDuplicatedProjectedPt[idx] for x, idx in zip(ofsDuplicatedIdx, range(len(ofsDuplicatedIdx))) if defenderTeamList[x] == False and player0umpire1[x] == 0]
            ofsDuplicatedIdx = [x for x in ofsDuplicatedIdx if defenderTeamList[x] == False and player0umpire1[x] == 0]
            
            if len(ofsDuplicatedPt) == 1:
                dist = cv2.norm(ofsBaseDefaultPose[i] - np.array(projectedPt[ofsDuplicatedIdx[0]]))
                if dist < MOVEMENT_THRESHOLD:
                    # New a player to the position
                    newPlayer = Player()
                    # TODO: (not hurry) 改成球員背號
                    newPlayer.id = len(currentOffendersOnField)
                    # newPlayer.tracker = cv2.TrackerKCF_create()
                    newPlayer.detectedID = ofsDuplicatedIdx[0]
                    newPlayer.detectedBox = detectedBox[ofsDuplicatedIdx[0]]
                    newPlayer.boundingBox = (detectedBox[ofsDuplicatedIdx[0]][0][0], 
                                                detectedBox[ofsDuplicatedIdx[0]][0][1], 
                                                detectedBox[ofsDuplicatedIdx[0]][1][0], 
                                                detectedBox[ofsDuplicatedIdx[0]][1][1])
                    # newPlayer.tracker.init(cameraImage, newPlayer.boundingBox)
                    newPlayer.defaultPose = ofsPoseList[ofsIdx].player.defaultPose
                    newPlayer.detectedPose = np.array(ofsDuplicatedProjectedPt[0])
                    newPlayer.detectedCameraPose = np.array(ofsDuplicatedPt[0])
                    # Add player to offender list
                    currentOffendersOnField.append(newPlayer)
                    # Update offensive position info
                    ofsPoseList[ofsIdx].playerID = newPlayer.id
                    ofsPoseList[ofsIdx].player = newPlayer
                    
                    ofsIdx += 1

            elif len(ofsDuplicatedPt) == 2:
                print("Unexpected!!!")
            elif len(ofsDuplicatedPt) == 3:
                print("Unexpected!!!")
        # 目前是如果按下reset，只會下一次偵測進攻方一次，不管有沒有找到都會取消
    isResetPressed = False
            

    if len(player0umpire1) == 0:
        player0umpire1 = [0] * len(projectedPt)
    for i in range(len(ofsPoseList)):
        tmpOfs = ofsPoseList[i]

        # if i > 0:
        #     lastOfs = ofsPoseList[i - 1]
        # if i == 0:
        #     lastOfs = ofsPoseList[len(ofsPoseList) - 1]

        # The position is not filled yet
        if tmpOfs.playerID == -1:
            # If this player is too close to last player, then skip
            # if len(currentOffendersOnField) > 0 and lastOfs.playerID > -1 and cv2.norm(tmpOfs.player.defaultPose - lastOfs.player.detectedPose) < 50:
            #     continue
            
            # List already detected poses, no more than 4 offenders
            ofsDetectedPoseList = [ofsPlayer.player.detectedPose for ofsPlayer in ofsPoseList if ofsPlayer.playerID > -1]
            if len(ofsDetectedPoseList) > 3:
                continue
            # If already offender in base area
            isAlreadyBatter = False
            for ofsDetectedPose in ofsDetectedPoseList:
                if cv2.norm(ofsDetectedPose - tmpOfs.player.defaultPose) < MOVEMENT_THRESHOLD:
                    isAlreadyBatter = True
            if isAlreadyBatter:
                continue
                    
            
            # 確認打擊區域有多少人，若球員皆站定位，最靠近投手的一定是打擊者，最靠近畫面左下角的是裁判
            min_idx = -1
            min_dist = 100000
            # Filled the position with YOLO detected
            for j in range(len(projectedPt)):
                tmpDist = cv2.norm(tmpOfs.player.defaultPose - np.array(projectedPt[j]))
                # Find closest player
                if tmpDist < min_dist:
                    if defenderTeamList[j] == False and player0umpire1[j] == 0:
                        min_idx = j
                        min_dist = tmpDist

            # 確保進攻位置不會誤判
            ofsDuplicatedIdx, ofsDuplicatedPt, ofsDuplicatedProjectedPt = checkIfPlayersDuplicated(NAS_config, i, playerPt, projectedPt, tmpOfs.player.defaultPose, min_dist)
            
            if len(ofsDuplicatedPt) == 1:
                dist = cv2.norm(tmpOfs.player.defaultPose - np.array(projectedPt[ofsDuplicatedIdx[0]]))
                if dist < MOVEMENT_THRESHOLD:
                    newPlayer = Player()
                    # TODO: (not hurry) 改成球員背號
                    newPlayer.id = len(currentOffendersOnField)
                    newPlayer.tracker = cv2.TrackerKCF_create()
                    newPlayer.detectedID = min_idx
                    newPlayer.detectedBox = detectedBox[min_idx]
                    newPlayer.boundingBox = (detectedBox[min_idx][0][0], 
                                            detectedBox[min_idx][0][1], 
                                            detectedBox[min_idx][1][0], 
                                            detectedBox[min_idx][1][1])
                    newPlayer.tracker.init(cameraImage, newPlayer.boundingBox)
                    newPlayer.defaultPose = tmpOfs.player.defaultPose
                    newPlayer.detectedPose = np.array(projectedPt[min_idx])
                    newPlayer.detectedCameraPose = np.array(playerPt[min_idx])
                    # Add player to offender list
                    currentOffendersOnField.append(newPlayer)
                    # Update offensive position info
                    tmpOfs.playerID = newPlayer.id
                    tmpOfs.player = newPlayer

                    ofsPoseList[i] = tmpOfs

            elif len(ofsDuplicatedPt) == 2:
                print("Unexpected!!!")
            elif len(ofsDuplicatedPt) == 3:
                print("Unexpected!!!")
                # leftBottom = np.array((15, 1725))
                # aDist = cv2.norm(np.array(ofsDuplicatedPt[0]) - leftBottom)
                # bDist = cv2.norm(np.array(ofsDuplicatedPt[1]) - leftBottom)
                # cDist = cv2.norm(np.array(ofsDuplicatedPt[2]) - leftBottom)
                # if (aDist < bDist and bDist < cDist) or (cDist < bDist and bDist < aDist):
                #     idx = 0
                # elif (bDist < aDist and aDist < cDist) or (cDist < aDist and aDist < bDist):
                #     idx = 1
                # elif (aDist < cDist and cDist < bDist) or (bDist < cDist and cDist < aDist):
                #     idx = 2
                    
                # newPlayer = Player()
                # # TODO: (not hurry) 改成球員背號
                # newPlayer.id = len(currentOffendersOnField)
                # # newPlayer.tracker = cv2.TrackerKCF_create()
                # newPlayer.detectedID = ofsDuplicatedIdx[idx]
                # newPlayer.detectedBox = detectedBox[ofsDuplicatedIdx[idx]]
                # newPlayer.boundingBox = (detectedBox[ofsDuplicatedIdx[idx]][0][0], 
                #                             detectedBox[ofsDuplicatedIdx[idx]][0][1], 
                #                             detectedBox[ofsDuplicatedIdx[idx]][1][0], 
                #                             detectedBox[ofsDuplicatedIdx[idx]][1][1])
                # # newPlayer.tracker.init(cameraImage, newPlayer.boundingBox)
                # newPlayer.defaultPose = tmpOfs.player.defaultPose
                # newPlayer.detectedPose = np.array(ofsDuplicatedProjectedPt[idx])
                # newPlayer.detectedCameraPose = np.array(ofsDuplicatedPt[idx])
                # # Add player to offender list
                # currentOffendersOnField.append(newPlayer)
                # # Update defensive position info
                # tmpOfs.playerID = newPlayer.id
                # tmpOfs.player = newPlayer

                # ofsPoseList[i] = tmpOfs

        else:
            # Last detected object
            min_idx = -1
            min_dist = 100000
            for j in range(len(projectedPt)):
                tmpDist = cv2.norm(tmpOfs.player.detectedPose - np.array(projectedPt[j]))

                # Find closest player
                if tmpDist < min_dist:
                    if (player0umpire1[j] == 0 and 
                        defenderTeamList[j] == False):
                        min_idx = j
                        min_dist = tmpDist
            
            isMovePossible = checkIsPlayerMomentPossible(NAS_config, tmpOfs.player.detectedPose, np.array(projectedPt[min_idx]))
            if isMovePossible:
                # Update offensive position info
                tmpOfs.player.detectedPose = np.array(projectedPt[min_idx])
                tmpOfs.player.detectedCameraPose = np.array(playerPt[min_idx])
                tmpOfs.player.detectedID = min_idx
                tmpOfs.player.detectedBox = detectedBox[min_idx]
            else:
                tmpOfs.player.detectedID = -1
            
            # # Opencv Tracker
            # isTrackerOk, tmpOfs.player.boundingBox = tmpOfs.player.tracker.update(cameraImage)
            # if isTrackerOk:
            #     tmpOfs.player.trackedPose = np.array([tmpOfs.player.boundingBox[0] + tmpOfs.player.boundingBox[2] / 2, 
            #                                           tmpOfs.player.boundingBox[1] + tmpOfs.player.boundingBox[3] / 2])
            # else:
            #     # TODO:
            #     tmpOfs.player.trackedPose = np.zeros(2)
            
            ofsPoseList[i] = tmpOfs
            
            if checkIsPlayerOut(NAS_config, tmpOfs.player.detectedCameraPose, isInFieldROIImage):
                ofp = OffensivePosition(i)
                ofp.playerID = -1
                defaultPlayer = Player()
                defaultPlayer.defaultPose = np.array(DEFAULT_OFS_POSE[0])
                defaultPlayer.detectedPose = np.array((-1, -1))
                defaultPlayer.detectedCameraPose = np.array((-1, -1))
                defaultPlayer.trackedPose = np.array((-1, -1))
                defaultPlayer.detectedID = -1
                defaultPlayer.detectedBox = [(-1, -1), (1, -1)]
                defaultPlayer.boundingBox = [-1, -1, -1, -1]
                ofp.player = defaultPlayer
                ofsPoseList[i] = ofp

    return ofsPoseList, isResetPressed

def modify_Defenders_And_Offenders(cameraImage, defPoseList, ofsPoseList, playerPt, projectedPt, currentOffendersOnField, detectedBox=[], lastPitcherROI=None, player0umpire1=[], defenderTeamList=[]):
    """Main function for modify defenders and offenders.

    Args:
        cameraImage:                np.array
        defPoseList:                np.array
        ofsPoseList:                [position]
        playerPt:                   
        projectedPt:                
        currentOffendersOnField:    
        detectedBox:
        lastPitcherROI:
        defenderTeamList:
    Returns:
        defPoseList:
        ofsPoseList:
    """

    offenderCount = 0
    for offender in ofsPoseList:
        if offender.playerID > -1:
            offenderCount +=1
    if offenderCount == 0:
        return defPoseList, ofsPoseList
    
    
    if len(player0umpire1) == 0:
        player0umpire1 = [0] * len(projectedPt)
    
    for i in range(len(defPoseList)):
        tmpDef = defPoseList[i]
        
        if tmpDef.playerID == 1 or tmpDef.playerID == 2 or tmpDef.playerID == 3:
            
            for j in range(len(ofsPoseList)):
                tmpOfs = ofsPoseList[j]
                if tmpOfs.playerID > -1:
                    if cv2.norm(tmpDef.player.detectedPose, tmpOfs.player.detectedPose) < 10:
                        pass

    return defPoseList, ofsPoseList

def checkIsPlayerMomentPossible(NAS_config, prePose, curPose):
    """Check if player movement is possible.
    
    Args:
        prePose:    [np.array]
        curPose:    [np.array]
    Returns:
        isMovePossible: bool
    """
    
    MOVEMENT_THRESHOLD = NAS_config['MOVEMENT_THRESHOLD']
    movement = cv2.norm(prePose - curPose)
    if movement > MOVEMENT_THRESHOLD:
        return False
    else:
        return True
    
def checkIsPlayerOut(NAS_config, curPose, isInFieldROIImage):
    """Check if player is out.
    
    Args:
        curPose:    [np.array]
    Returns:
                    boolean
    """
    
    OFS_IN_FIELD_BASE = NAS_config['OFS_IN_FIELD_BASE']
    OFS_IN_FIELD_LEFTBORDER = NAS_config['OFS_IN_FIELD_LEFTBORDER']
    OFS_IN_FIELD_RIGHTBORDER = NAS_config['OFS_IN_FIELD_RIGHTBORDER']
    OFS_BASE_CIRCLE = NAS_config['OFS_BASE_CIRCLE']
    OFS_RIGHT_OUT_DISTANCE = NAS_config['OFS_RIGHT_OUT_DISTANCE']
    OFS_LEFT_OUT_DISTANCE = NAS_config['OFS_LEFT_OUT_DISTANCE']
    OFS_BASE_OUT_DISTANCE = NAS_config['OFS_BASE_OUT_DISTANCE']
    base        = OFS_IN_FIELD_BASE
    leftBorder  = OFS_IN_FIELD_LEFTBORDER
    rightBorder = OFS_IN_FIELD_RIGHTBORDER
    base_circle = OFS_BASE_CIRCLE
    
    # right border
    array_longi  = np.array([rightBorder[0] - base[0], rightBorder[1] - base[1]])
    array_trans = np.array([rightBorder[0] - curPose[0], rightBorder[1] - curPose[1]])
    # 用向量計算點到直線距離
    array_temp = (float(array_trans.dot(array_longi)) / array_longi.dot(array_longi))   # 注意转成浮点数运算
    array_temp = array_longi.dot(array_temp)
    rightDistance   = np.sqrt((array_trans - array_temp).dot(array_trans - array_temp))
    
    # left border
    array_longi  = np.array([leftBorder[0] - base[0], leftBorder[1] - base[1]])
    array_trans = np.array([leftBorder[0] - curPose[0], leftBorder[1] - curPose[1]])
    # 用向量計算點到直線距離
    array_temp = (float(array_trans.dot(array_longi)) / array_longi.dot(array_longi))   # 注意转成浮点数运算
    array_temp = array_longi.dot(array_temp)
    leftDistance   = np.sqrt((array_trans - array_temp).dot(array_trans - array_temp))

    if (rightDistance < OFS_RIGHT_OUT_DISTANCE or 
        leftDistance < OFS_LEFT_OUT_DISTANCE and 
        cv2.norm(curPose - np.array(base)) > OFS_BASE_OUT_DISTANCE):
        return True
    elif is_in_field_ROI(isInFieldROIImage, (curPose[0], curPose[1]), defender0Offender1=1) == False:
        return True
    else:
        return False
    
def checkIsPlayerNotOccupied(currentID, targetID):
    """Check if player is occupied by other position.
    
    Args:
        currentID:      [np.array]
        targetID:       [np.array]
    Returns:
        isNotOccupied:  bool
    """
    # movement = cv2.norm(prePose - curPose)
    # if movement > MOVEMENT_THRESHOLD:
    #     return False
    # else:
    #     return True

def drawPlayersOnFieldImage(fieldImage, playerPoseList, color=(255, 0, 0)):
    """Check if player movement is possible.
    
    Args:
        fieldImage:     np.array
        playerPoseList: np.array
    Returns:
        trackingImage:  np.array
    """
    
    trackingImage = fieldImage.copy()
    idx = 0
    for playerPose in playerPoseList:
        if playerPose.playerID > -1:
            cv2.circle(trackingImage, point2fToIntTuple(playerPose.player.detectedPose), 5, color, -1)
            cv2.circle(trackingImage, point2fToIntTuple(playerPose.player.defaultPose), 5, (0, 0 ,0), -1)
            cv2.line(trackingImage, point2fToIntTuple(playerPose.player.detectedPose), point2fToIntTuple(playerPose.player.defaultPose), (0, 255, 0))
            cv2.putText(trackingImage, str(idx), point2fToIntTuple(playerPose.player.detectedPose), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
        else:
            # Not find player yet
            cv2.circle(trackingImage, point2fToIntTuple(playerPose.player.defaultPose), 5, (0, 0 ,0), -1)
        idx +=1
    return trackingImage

def drawBoxOnCameraImage(cameraImage, boxList, color=(0, 0, 255), thickness=3):
    """Check if player movement is possible.
    
    Args:
        cameraImage:    np.array
        boxList:        [(x, y), (x + w, y + h)]
    Returns:
        trackingImage:  np.array
    """
    
    trackingImage = cameraImage.copy()
    # for playerPose in playerPoseList:
    #     if playerPose.player.trackedPose[0] > 0 and playerPose.player.trackedPose[1] > 0:
    #         (x, y, w, h) = [int(v) for v in playerPose.player.boundingBox]
    #         cv2.rectangle(trackingImage, (x, y), (x + w, y + h), (0, 0, 255), 3)
    
    for box in boxList:
        cv2.rectangle(trackingImage, 
                      (int(box[0]), int(box[1])), 
                      (int(box[0] + box[2]), int(box[1] + box[3])), 
                      color, 
                      thickness)
            
    return trackingImage

def get_defender_list_with_hue_compare(cameraImage, lastPitcherROI, detectedBox, threshold=0.2):
    """Get the defender list by compare the pitcher with every detected bounding box in the camera image.
    Return a list with boolean values of all detected boxes.
    
    Args:
        cameraImage:        np.array
        lastPitcherROI:     np.array
        detectedBox:        []
        threshold:          float
    Returns:
        defenderTeamList:   [boolean]
    """
    # Get ROI images
    detectedBoxLeftTopRightBottom = [[ROI[0][0], ROI[0][1], ROI[0][0] + ROI[1][0], ROI[0][1] + ROI[1][1]] for ROI in detectedBox]
    ROIImageList = []
    for ROI in detectedBoxLeftTopRightBottom:
        tmp = cameraImage[ROI[1] : ROI[3], ROI[0] : ROI[2]]
        ROIImageList.append(tmp)

    defenderTeamList = TC.calc_defenders_possibilities(ROIImageList, lastPitcherROI, threshold)
    return defenderTeamList
    
def get_pitcher_ROI_image(cameraImage, defenderList):
    """Get the last image of the pitcher.
    
    Args:
        cameraImage:        np.array
        defenderList:       []
    Returns:
        pitcherImage:       np.array
    """
    if defenderList[8].playerID > -1:
        pitcherLTRB = [defenderList[8].player.detectedBox[0][0],
                    defenderList[8].player.detectedBox[0][1],
                    defenderList[8].player.detectedBox[0][0] + defenderList[8].player.detectedBox[1][0],
                    defenderList[8].player.detectedBox[0][1] + defenderList[8].player.detectedBox[1][1]]
        pitcherImage = cameraImage[pitcherLTRB[1] : pitcherLTRB[3], pitcherLTRB[0] : pitcherLTRB[2]]
        return pitcherImage
    else:
        return None

def compute_result(defPoseList=[], ofsPoseList=[], NAS_config={}):
    """Compute the results for Wade Du.
    
    Args:
        defPoseList:        []
        ofsPoseList:        []
    Returns:
        results:            [[x, y], trackerID, "", "defend0Attack1Judge2"]
    """
    if not NAS_config:
        DEF_SHIFT_BIAS = [[0, 0],
                          [0, 0],
                          [0, 0],
                          [0, 0],
                          [0, 0],
                          [0, 0],
                          [0, 0],
                          [0, 0],
                          [0, 0]]
    else:
        DEF_SHIFT_BIAS = NAS_config['DEF_SHIFT_BIAS']
        
    results = []
    for tmpPlayer in defPoseList:
        if tmpPlayer.playerID > -1:
            result = [point2fToIntTuple(tmpPlayer.player.detectedPose + DEF_SHIFT_BIAS[tmpPlayer.positionID]), 
                    tmpPlayer.positionID, 
                    "", 
                    "0"]
            results.append(result)
        else:
            result = [point2fToIntTuple(tmpPlayer.player.defaultPose), 
                    tmpPlayer.positionID, 
                    "", 
                    "0"]
            results.append(result)
        
    for tmpPlayer in ofsPoseList:
        if tmpPlayer.playerID > -1:
            result = [point2fToIntTuple(tmpPlayer.player.detectedPose + DEF_SHIFT_BIAS[tmpPlayer.positionID]), 
                    tmpPlayer.positionID, 
                    "", 
                    "1"]
            results.append(result)
    return results

def compute_result_pickle(filename, defPoseList=[], ofsPoseList=[]):
    """Compute the pickle results for Ray Liu.
    
    Args:
        defPoseList:        []
        ofsPoseList:        []
    Returns:
    """
    
    results = []
    for tmpPlayer in defPoseList:
        if tmpPlayer.playerID > -1:
            result = [point2fToIntTuple(tmpPlayer.player.detectedPose), 
                    tmpPlayer.positionID, 
                    tmpPlayer.playerID, 
                    "0"]
            results.append(result)
        else:
            result = [point2fToIntTuple(tmpPlayer.player.defaultPose), 
                    tmpPlayer.positionID, 
                    tmpPlayer.playerID, 
                    "0"]
            results.append(result)
        
        
    for tmpPlayer in ofsPoseList:
        if tmpPlayer.playerID > -1:
            result = [point2fToIntTuple(tmpPlayer.player.detectedPose), 
                    tmpPlayer.positionID, 
                    tmpPlayer.playerID, 
                    "1"]
            results.append(result)

    with open(filename,"ab") as handle:
        pickle.dump(results, handle)
    
def point2fToIntTuple(point2f):
    """Convert point2f (np.array) to tuple.

    Args:
        point2f:    np.array((x, y))
    Returns:
        tuple
    """
    
    return (int(point2f[0]), int(point2f[1]))

# For draw isInFieldROIImage:
def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return [A, B, -C]

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return (int(math.fabs(x)), int(y))
    elif D == 0:
        x = Dx / 0.00000001
        y = Dy / 0.00000001
        return (int(math.fabs(x)), int(y))
    
def decide_intersection_boarder(base, border, cameraImage):
    downBorder = cameraImage.shape[0] - 1
    rightBorder = cameraImage.shape[1] - 1
    
    dx = border[0] - base[0]
    dy = border[1] - base[1]
    
    if dy > 0:
        lineH = line([0, 0], [rightBorder, 0])
    else:
        lineH = line([0, downBorder], [rightBorder, downBorder])
    if dx > 0:
        lineV = line([rightBorder, 0], [rightBorder, downBorder])
    else:
        lineV = line([0, 0], [0, downBorder])
    return lineH, lineV


def test_run():
    # fileName = "nas_5_20-30/20190825_20-30-00.033215.jpg"
    # camera_img = cv2.imread(fileName)
    # field_img = cv2.imread("./baseball_package/field.png")

    # #-----findHomographyBetweenCameraAndField-----
    # findHomographyBetweenCameraAndField()

    # #-----loadHomography-----
    # H = loadHomography()
    
    # #-----setupDefensivePose-----
    # defPoseList = setupDefensivePose()
    # print(defPoseList[0].positionID)
    # print(len(defPoseList))

    # #-----setupDefensivePose-----
    # ofsPoseList = setupOffensivePose()

    # #-----getTxtFilePath-----
    # print(getTxtFilePath("nas_5_20-30/20190825_20-30-00.033215.jpg"))

    # # -----findPlayerPoseFromTxt-----
    # print(findPlayerPoseFromTxt("nas_5_20-30/20190825_20-30-00.033215.jpg"))

    # #-----isInField-----drawDetectedPlayer-----
    # playerPoseList = findPlayerPoseFromTxt(fileName)
    # img_detected = drawDetectedPlayer(img, playerPoseList)
    # cv2.namedWindow("img_detected", cv2.WINDOW_NORMAL)
    # cv2.imshow("img_detected", img_detected)

    # #-----retrieveImagesFromFolder-----
    # print(retrieveImagesFromFolder("nas_5_20-30/*.jpg"))

    # #-----drawDefaultDefensivePose-----
    # field = cv2.imread(FIELD_NAME)
    # defPoseList = setupDefensivePose()
    # img_defaultDef = drawDefaultDefensivePose(field, defPoseList)
    # cv2.namedWindow("img_defaultDef", cv2.WINDOW_NORMAL)
    # cv2.imshow("img_defaultDef", img_defaultDef)
    # cv2.waitKey(0)

    # #-----drawDefaultOffensivePose-----
    # ofsPoseList = setupOffensivePose()
    # img_defaultOfs = drawDefaultOffensivePose(field_img, ofsPoseList)

    # cv2.namedWindow("camera_img", cv2.WINDOW_NORMAL)
    # cv2.imshow("camera_img", camera_img)
    # cv2.waitKey(0)
    
    # ========== Main ==========

    # Load NAS_config
    with open('./osense_baseball_player_tracking/reference/camera_config.json') as config_json_file:
        configs = json.load(config_json_file)
        for param in configs:
            if param['NAS_INDEX'] == NAS_INDEX:
                NAS_config = param

    currentDefendersOnField = []
    currentOffendersOnField = []
    currentReferersOnField = []
    isResetPressed = False

    field = cv2.imread(FIELD_NAME)
    
    # === Find and save homography ===
    # findHomographyBetweenCameraAndField(NAS_config)
    
    # === Load homography ===
    H = loadHomography(NAS_config)

    # === Find and save isInFieldROIImage ===
    # isInFieldROIImage = draw_is_in_field_ROI(NAS_config, defender0Offender1=1)
    # cv2.namedWindow("isInFieldROIImage", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('isInFieldROIImage', 1260,680)
    # cv2.imshow('isInFieldROIImage', isInFieldROIImage)
    # cv2.waitKey(0)
    # === Load isInFieldROIImage ===
    isInFieldROIImage = load_isInFieldROIImage(NAS_config)

    # === Setup defenders ===
    defPoseList = setupDefensivePose(NAS_config)
    
    # === Setup Offenders ===
    # TODO: call this function after got all defenders
    ofsPoseList = setupOffensivePose(NAS_config)

    # === lastPitcherImage ===
    lastPitcherImage = None
    
    # === Draw default defensive and offensive positions ===
    # img_defaultDef = drawDefaultDefensivePose(field, defPoseList)
    # img_default = drawDefaultOffensivePose(img_defaultDef, ofsPoseList)

    # === Start ===
    cv2.namedWindow("imageForShow", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('imageForShow', 1260,680)
    cv2.namedWindow("tmpField", cv2.WINDOW_NORMAL)
    cv2.waitKey(0)
    imageNames = retrieveImagesFromFolder(PATH_OF_IMAGE_FOLDER)
    for i in range(len(imageNames)):
        # if i < 600: 
        #     continue
        if i%3 != 0:
            continue
        # TODO: Add Lily & Nicole function
        image = cv2.imread(imageNames[i])

        if image.size == 0:
            continue

        # Get player position on camera
        playerInfoList = findPlayerPoseFromTxt(NAS_config, imageNames[i], isInFieldROIImage)
        playerPt = [pt['detectedPt'] for pt in playerInfoList]
        
        
        print("------------------------------------------")
        print("frame : ", i)
        start_time = time.time()
        
        # Adjust detected player points for homography
        adjustedPlayerPt = adjustPlayerPointForHomography(playerInfoList)
        
        # 座標投影至戰術圖
        projectedPt = cv2.perspectiveTransform(np.float32(adjustedPlayerPt).reshape(-1, 1, 2), H).reshape(-1, 2).tolist()[:]

        # Bounding box top left & width & height
        detectedBoxLeftTopWH = [[BB['detectedBoxLeftTopPt'], BB['detectedBoxWH']] for BB in playerInfoList]

        # Player : 0, umpire : 1
        player0umpire1 = [PU['player0umpire1'] for PU in playerInfoList]

        # 找防守者
        defPoseList, defenderTeamList, hueCompareResultList = findDefenders(NAS_config, image, defPoseList, playerPt, projectedPt, currentDefendersOnField, detectedBoxLeftTopWH, lastPitcherImage, player0umpire1)        

        defenderCount = 0
        for defender in defenderTeamList:
            if defender:
                defenderCount +=1
        if defenderCount >= NAS_config['LEAST_DEF_PLAYER']:
            ofsPoseList, isResetPressed = findOffenders(NAS_config, image, isInFieldROIImage, ofsPoseList, playerPt, projectedPt, currentOffendersOnField, detectedBoxLeftTopWH, lastPitcherImage, player0umpire1, defenderTeamList, isResetPressed)

        # defPoseList, ofsPoseList = modify_Defenders_And_Offenders(image, defPoseList, ofsPoseList, playerPt, projectedPt, currentOffendersOnField, detectedBoxLeftTopWH, lastPitcherImage, player0umpire1, defenderTeamList)
    

        # === Don't forget ===
        # TODO: 還沒考慮如果這一幀投手不夠像上一幀的結果
        if defPoseList[8].playerID > -1:
            lastPitcherImage = get_pitcher_ROI_image(image, defPoseList)
        
        end_time = time.time()
        print("----- %s seconds without detection -----" % (time.time() - start_time))
        
        # Draw defenders & offenders on field image
        tmpField = drawPlayersOnFieldImage(field, defPoseList)
        tmpField = drawPlayersOnFieldImage(tmpField, ofsPoseList, (0, 0, 255))

        # for j in range(len(currentOffendersOnField)):
        #     cv2.circle(tmpField, point2fToIntTuple(currentOffendersOnField[j].detectedPose), 5, (0, 255, 0), -1)

        
        imageForShow = image.copy()
        # Draw in field rules on camera image
        DEF_IN_FIELD_BASE = tuple(NAS_config['DEF_IN_FIELD_BASE'])
        DEF_IN_FIELD_LEFTBORDER = tuple(NAS_config['DEF_IN_FIELD_LEFTBORDER'])
        DEF_IN_FIELD_RIGHTBORDER = tuple(NAS_config['DEF_IN_FIELD_RIGHTBORDER'])
        DEF_BASE_CIRCLE = NAS_config['DEF_BASE_CIRCLE']
        OFS_IN_FIELD_BASE = tuple(NAS_config['OFS_IN_FIELD_BASE'])
        OFS_IN_FIELD_LEFTBORDER = tuple(NAS_config['OFS_IN_FIELD_LEFTBORDER'])
        OFS_IN_FIELD_RIGHTBORDER = tuple(NAS_config['OFS_IN_FIELD_RIGHTBORDER'])
        OFS_BASE_CIRCLE = NAS_config['OFS_BASE_CIRCLE']
        cv2.line(imageForShow, DEF_IN_FIELD_BASE, DEF_IN_FIELD_LEFTBORDER, (200, 0, 0), 3)
        cv2.line(imageForShow, DEF_IN_FIELD_BASE, DEF_IN_FIELD_RIGHTBORDER, (200, 0, 0), 3)
        cv2.circle(imageForShow, DEF_IN_FIELD_BASE, DEF_BASE_CIRCLE, (200, 0, 0), 3)
        cv2.line(imageForShow, OFS_IN_FIELD_BASE, OFS_IN_FIELD_LEFTBORDER, (0, 0, 200), 3)
        cv2.line(imageForShow, OFS_IN_FIELD_BASE, OFS_IN_FIELD_RIGHTBORDER, (0, 0, 200), 3)
        cv2.circle(imageForShow, OFS_IN_FIELD_BASE, OFS_BASE_CIRCLE, (0, 0, 200), 3)
        
        
        # Draw detectedPlayer on camera image
        imageForShow = drawDetectedPlayer(imageForShow, playerPt)
        
        #   defenders
        defDetectedCameraPoseList = [defPose.player.detectedCameraPose for defPose in defPoseList if defPose.playerID > -1]
        adjustedDefDetectedCameraPoseList = [(defPose.player.detectedCameraPose[0], defPose.player.detectedBox[0][1] + defPose.player.detectedBox[1][1]) for defPose in defPoseList if defPose.playerID > -1]
        defDetectedPlayerIdxList = [defPose.positionID for defPose in defPoseList if defPose.playerID > -1]
        imageForShow = drawDetectedPlayer(imageForShow, defDetectedCameraPoseList, (255, 0, 0), 15, adjustedDefDetectedCameraPoseList, defDetectedPlayerIdxList)
        #   offenders
        ofsDetectedCameraPoseList = [ofsPose.player.detectedCameraPose for ofsPose in ofsPoseList if ofsPose.playerID > -1]
        adjustedOfsDetectedCameraPoseList = [(ofsPose.player.detectedCameraPose[0], ofsPose.player.detectedBox[0][1] + ofsPose.player.detectedBox[1][1]) for ofsPose in ofsPoseList if ofsPose.playerID > -1]
        ofsDetectedPlayerIdxList = [ofsPose.positionID for ofsPose in ofsPoseList if ofsPose.playerID > -1]
        imageForShow = drawDetectedPlayer(imageForShow, ofsDetectedCameraPoseList, (0, 0, 255), 15, adjustedOfsDetectedCameraPoseList, ofsDetectedPlayerIdxList)
        
        # trackerBoxList = [trackerB.player.boundingBox for trackerB in defPoseList]
        # imageForShow = drawBoxOnCameraImage(imageForShow, trackerBoxList, (0, 255, 0))
        
        detectedBoxList = [[detectedB[0][0], detectedB[0][1], detectedB[1][0], detectedB[1][1]] for detectedB in detectedBoxLeftTopWH]
        imageForShow = drawBoxOnCameraImage(imageForShow, detectedBoxList, (0, 200, 200))
        
        
        # Draw not defenders with black boxes with hue
        detectedBoxList_np = np.array(detectedBoxList)
        hueCompareResultList_np = np.array(hueCompareResultList)
        hueCompareResultNOTDefender = detectedBoxList_np[~hueCompareResultList_np].tolist()
        imageForShow = drawBoxOnCameraImage(imageForShow, hueCompareResultNOTDefender, color=(255, 255, 255), thickness=5)

        # Draw not defenders with black boxes with YOLO
        umpireBox = []
        for j in range(len(player0umpire1)):
            if player0umpire1[j] == 1:
                umpireBox.append(detectedBoxList[j])
        imageForShow = drawBoxOnCameraImage(imageForShow, umpireBox, color=(0, 0, 0))
        
        # for tmpPlayer in defPoseList:
        #     print("Position :",tmpPlayer.positionID, " playerID :", tmpPlayer.playerID, " player.detectedID :", tmpPlayer.player.detectedID)
        print("Defender : Position : ",defPoseList[0].positionID, " playerID :", defPoseList[0].playerID, " player.detectedID :", defPoseList[0].player.detectedID)
        for tmpPlayer in ofsPoseList:
            if tmpPlayer.playerID > -1:
                print("Offender : Position : ",tmpPlayer.positionID, " playerID :", tmpPlayer.playerID, " player.detectedID :", tmpPlayer.player.detectedID)
        print("isResetPressed: ", isResetPressed)
        cv2.imshow("imageForShow", imageForShow)
        cv2.imshow("tmpField", tmpField)
        k = cv2.waitKey(1)
        if k == 27:    # Esc key to stop
            break
        elif k == ord('r'):  # normally -1 returned,so don't print it
            print("Reinitializing defenders!!!")
            defPoseList = setupDefensivePose(NAS_config)
            ofsPoseList = setupOffensivePose(NAS_config)
            isResetPressed = True
        elif k == ord('d'):  # normally -1 returned,so don't print it
            print("Reinitializing defenders!!!")
            defPoseList = setupDefensivePose(NAS_config)
        elif k == ord('o'):  # normally -1 returned,so don't print it
            print("Reinitializing defenders!!!")
            ofsPoseList = setupOffensivePose(NAS_config)

        # Results in pickle
        # compute_result_pickle("./osense_baseball_player_tracking/reference/camera_functions_results_NAS_0{}.pkl".format(NAS_INDEX), defPoseList, ofsPoseList)




    # cv2.namedWindow("img_default", cv2.WINDOW_NORMAL)
    # cv2.imshow("img_default", img_default)

    # cv2.namedWindow("field", cv2.WINDOW_NORMAL)
    # cv2.imshow("field", field)
    # cv2.waitKey(0)


if __name__ == "__main__":
    # print("Load pickle")
    # results = []
    # with open("./osense_baseball_player_tracking/reference/camera_functions_results.pkl", 'rb') as handle:
    #     while True:
    #         try:
    #             print(len(results))
    #             results.append(pickle.load(handle))
    #         except EOFError:
    #             break
        
    print("test_run")
    test_run()