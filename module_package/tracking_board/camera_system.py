#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
=========================================================
File Name   :   camera_system.py
Copyright   :   Chao-Chuan Lu
Create date :   Sep. 10, 2019
Description :   Main system for camera tracking.
TODO: 
=========================================================
'''
##                                                       &mmmm@@@%%%%%%%%8%%88888%%%%%%%%@@@@@@@@
##                                                       @@@@@%%%%%%8888888888888g&&&%@@@@@@@@@@@
##                                                       @@@%%%%%%88888gggg&8g&g@@M@@%%%%%*""""`?
##                          _ggg_    _ggg                @@@@%%%%%%%%%%%M@@@@@@@@@@@@@@Mg_       
##                         m@N@@@g _@@F@@M               %%%@@%%%%^"`` @@@@@@@@@@@@@@@@@@@@w     
##                        ]@M   7" M@K   "               8        _m@E@@@@@@@@@@@@@@@@@@@@@M-    
## qmmg                   @@K      @@C                             g@@@@@@@@@@@@@@@@@@@@@@@@K    
## "@@M                   M@L     0@M          _g_               ,@@@@@@@@@@@@@@@@@@@@@@@@@M     
##  M@K      pmg  q@@M ___@@L_____]@M____mmg  0@@@               m@@@@@@@@@@@@@@@@@@@@@@@@%g     
##  M@P      @@@   @@$4@@@@@@@W@@@@@@@@W@@@M   @@W                m@@@@@@@@@@@@@@@@@@@@@@%%%     
##  @@L      ]@@  0@@$    M@L      @M    @@M  ]@@M                 @@@@@@@@@@@@@@@@@@@W%8"`g_    
## ]@@L ____  @@C_@@@K    M@L      @@    "@@L_@@@M                 *@@@@@@@@@Km%@@@@%%8^<m@#C    
## ]@@@@@@@@P M@@@@@@@g   M@@_     @@M    @@@@@@@M                  "@@@@@@@M@@M%%%%8   8@@L     
##  %@W"`     *@@W  @@N   W@@^     @@W    "@@Y ]@M                   @@@@@@@@@@gg@@@&&ggg%@@     
##                                             ]@M                    M@@@@@@Mg7%@@@@@@@@@@W?,   
##                                             ]@M       >            "@@@@@@@@@M@@@@@@@@@@M"    
##                                             M@W       88>           M@@@@@@@@@@@@@@@@@@@M     
##                                             *W        %888>         M@@@@@@@@@@@@@@@@@@@L     
##                                                       %%%%888      q@@@@@@@@@@@@@@@@@@@K    <8
##                                                       %%%%%%%88> _m@@@@@@@@@@@@@@@@@@@MC  <888

import queue
import threading
import time
import cv2
import numpy as np
from . import camera_functions as CF
from .yolov3_package import yolov3_predict as YOLO
from PIL import Image
import json

FIELD_IMAGE_FILE        =   "./osense_baseball_player_tracking/src/field.png"
NAS_CONFIG_FILE         =   "./osense_baseball_player_tracking/reference/camera_config.json"
PATH_OF_REFERENCE       =   "./osense_baseball_player_tracking/reference/"

PATH_OF_IMAGE_FOLDER    =   "../洲際棒球場/python/out_640_0.5_nas_5_20-30_freeze1/*.jpg"

class Camera_System:
    def __init__(self, nas_id, gpu_id, queue):
        self.nas_id = nas_id
        self.gpu_id = gpu_id
        self.queue = queue
        self.cameraShow = np.array([])
        self.fieldShow = np.array([])
        # Initialization
        self.currentDefendersOnField = []
        self.currentOffendersOnField = []
        self.currentReferersOnField = []
        self.fieldIamge = cv2.imread(FIELD_IMAGE_FILE)
        self.NAS_config = self.load_NAS_config()
        # === Load homography ===
        self.homography = CF.loadHomography(self.NAS_config)
        # === Load isInFieldROIImage ===
        self.isInFieldROIImage = CF.load_isInFieldROIImage(self.NAS_config)
        # === Setup defenders ===
        self.defPoseList = CF.setupDefensivePose(self.NAS_config)
        # === Setup Offenders ===
        # TODO: call this function after got all defenders
        self.ofsPoseList = CF.setupOffensivePose(self.NAS_config)
        # === lastPitcherImage ===
        self.lastPitcherImage = None
        # === Initial YOLOv3 ===
        # self.yolov3 = YOLO.initial_model('yolov3')
        # cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
        self.isResetPressed = False
    
    def init_yolo(self):
        self.yolov3 = YOLO.initial_model('yolov3')

    def load_NAS_config(self):
        # Load NAS_config
        with open(NAS_CONFIG_FILE) as config_json_file:
            configs = json.load(config_json_file)
            for param in configs:
                if param['NAS_INDEX'] == self.nas_id:
                    NAS_config = param
                else:
                    print("Can's find NAS config information!!!")
        return NAS_config
    
    def execute(self, image, yoloResults):
        print("--------------------------------------------------------------------------")
        start_time = time.time()
        self.cameraImage = image
        # wade
        # read from queue
        # self.cameraImage = self.queue.get()
        
        # if self.cameraImage.size == 0:
        #     pass
        
        # TODO: Add Lily & Nicole function
        # yoloResults = YOLO.detect_img(self.yolov3, self.cameraImage)
        self.playerInfoList = CF.compute_player_info(self.NAS_config, yoloResults, self.isInFieldROIImage)
        
        playerPt = [pt['detectedPt'] for pt in self.playerInfoList]

        # Adjust detected player points for homography
        adjustedPlayerPt = CF.adjustPlayerPointForHomography(self.playerInfoList)
        
        # 座標投影至戰術圖
        projectedPt = cv2.perspectiveTransform(np.float32(adjustedPlayerPt).reshape(-1, 1, 2), self.homography).reshape(-1, 2).tolist()[:]

        # Bounding box top left & width & height
        detectedBoxLeftTopWH = [[BB['detectedBoxLeftTopPt'], BB['detectedBoxWH']] for BB in self.playerInfoList]
        # Player : 0, umpire : 1
        player0umpire1 = [PU['player0umpire1'] for PU in self.playerInfoList]
        # 找防守者
        #!FIXME: if change PIL to opencv, this should be changed too
        # self.cameraImage = cv2.cvtColor(np.array(self.cameraImage), cv2.COLOR_RGB2BGR)
        
        self.defPoseList, defenderTeamList, hueCompareResultList = CF.findDefenders(self.NAS_config, self.cameraImage, self.defPoseList, playerPt, projectedPt, self.currentDefendersOnField, detectedBoxLeftTopWH, self.lastPitcherImage, player0umpire1)

        defenderCount = 0
        for defender in self.defPoseList:
            if defender.playerID > -1:
                defenderCount +=1
        if defenderCount >= self.NAS_config['LEAST_DEF_PLAYER']:
            self.ofsPoseList, self.isResetPressed = CF.findOffenders(self.NAS_config, self.cameraImage, self.isInFieldROIImage, self.ofsPoseList, playerPt, projectedPt, self.currentOffendersOnField, detectedBoxLeftTopWH, self.lastPitcherImage, player0umpire1, defenderTeamList, self.isResetPressed)

        # self.defPoseList, self.ofsPoseList = CF.modify_Defenders_And_Offenders(self.cameraImage, self.defPoseList, self.ofsPoseList, playerPt, projectedPt, self.currentOffendersOnField, detectedBoxLeftTopWH, self.lastPitcherImage, player0umpire1, defenderTeamList)

        # === Don't forget ===
        # TODO: 還沒考慮如果這一幀投手不夠像上一幀的結果
        if self.defPoseList[8].playerID > -1:
            self.lastPitcherImage = CF.get_pitcher_ROI_image(self.cameraImage, self.defPoseList)
        
        # Draw defenders on field image
        tmpField = CF.drawPlayersOnFieldImage(self.fieldIamge, self.defPoseList)
        tmpField = CF.drawPlayersOnFieldImage(tmpField, self.ofsPoseList, (0, 0, 255))

        
        imageForShow = self.cameraImage.copy()
        # Draw in field rules on camera image
        DEF_IN_FIELD_BASE = tuple(self.NAS_config['DEF_IN_FIELD_BASE'])
        DEF_IN_FIELD_LEFTBORDER = tuple(self.NAS_config['DEF_IN_FIELD_LEFTBORDER'])
        DEF_IN_FIELD_RIGHTBORDER = tuple(self.NAS_config['DEF_IN_FIELD_RIGHTBORDER'])
        DEF_BASE_CIRCLE = self.NAS_config['DEF_BASE_CIRCLE']
        OFS_IN_FIELD_BASE = tuple(self.NAS_config['OFS_IN_FIELD_BASE'])
        OFS_IN_FIELD_LEFTBORDER = tuple(self.NAS_config['OFS_IN_FIELD_LEFTBORDER'])
        OFS_IN_FIELD_RIGHTBORDER = tuple(self.NAS_config['OFS_IN_FIELD_RIGHTBORDER'])
        OFS_BASE_CIRCLE = self.NAS_config['OFS_BASE_CIRCLE']
        cv2.line(imageForShow, DEF_IN_FIELD_BASE, DEF_IN_FIELD_LEFTBORDER, (200, 0, 0), 3)
        cv2.line(imageForShow, DEF_IN_FIELD_BASE, DEF_IN_FIELD_RIGHTBORDER, (200, 0, 0), 3)
        cv2.circle(imageForShow, DEF_IN_FIELD_BASE, DEF_BASE_CIRCLE, (200, 0, 0), 3)
        cv2.line(imageForShow, OFS_IN_FIELD_BASE, OFS_IN_FIELD_LEFTBORDER, (0, 0, 200), 3)
        cv2.line(imageForShow, OFS_IN_FIELD_BASE, OFS_IN_FIELD_RIGHTBORDER, (0, 0, 200), 3)
        cv2.circle(imageForShow, OFS_IN_FIELD_BASE, OFS_BASE_CIRCLE, (0, 0, 200), 3)
        
        # Draw detectedPlayer on camera image
        imageForShow = CF.drawDetectedPlayer(imageForShow, playerPt)
        
        #   defenders
        defDetectedCameraPoseList = [defPose.player.detectedCameraPose for defPose in self.defPoseList if defPose.playerID > -1]
        adjustedDefDetectedCameraPoseList = [(defPose.player.detectedCameraPose[0], defPose.player.detectedBox[0][1] + defPose.player.detectedBox[1][1]) for defPose in self.defPoseList if defPose.playerID > -1]
        defDetectedPlayerIdxList = [defPose.positionID for defPose in self.defPoseList if defPose.playerID > -1]
        imageForShow = CF.drawDetectedPlayer(imageForShow, defDetectedCameraPoseList, (255, 0, 0), 15, adjustedDefDetectedCameraPoseList, defDetectedPlayerIdxList)
        # #   offenders
        ofsDetectedCameraPoseList = [ofsPose.player.detectedCameraPose for ofsPose in self.ofsPoseList if ofsPose.playerID > -1]
        adjustedOfsDetectedCameraPoseList = [(ofsPose.player.detectedCameraPose[0], ofsPose.player.detectedBox[0][1] + ofsPose.player.detectedBox[1][1]) for ofsPose in self.ofsPoseList if ofsPose.playerID > -1]
        ofsDetectedPlayerIdxList = [ofsPose.positionID for ofsPose in self.ofsPoseList if ofsPose.playerID > -1]
        imageForShow = CF.drawDetectedPlayer(imageForShow, ofsDetectedCameraPoseList, (0, 0, 255), 15, adjustedOfsDetectedCameraPoseList, ofsDetectedPlayerIdxList)
        
        # trackerBoxList = [trackerB.player.boundingBox for trackerB in defPoseList]
        # imageForShow = drawBoxOnCameraImage(imageForShow, trackerBoxList, (0, 255, 0))
        
        detectedBoxList = [[detectedB[0][0], detectedB[0][1], detectedB[1][0], detectedB[1][1]] for detectedB in detectedBoxLeftTopWH]
        imageForShow = CF.drawBoxOnCameraImage(imageForShow, detectedBoxList, (0, 200, 200))
        
        
        # Draw not defenders with black boxes with hue
        detectedBoxList_np = np.array(detectedBoxList)
        hueCompareResultList_np = np.array(hueCompareResultList)
        hueCompareResultNOTDefender = detectedBoxList_np[~hueCompareResultList_np].tolist()
        imageForShow = CF.drawBoxOnCameraImage(imageForShow, hueCompareResultNOTDefender, color=(255, 255, 255), thickness=5)

        # Draw not defenders with black boxes with YOLO
        umpireBox = []
        for j in range(len(player0umpire1)):
            if player0umpire1[j] == 1:
                umpireBox.append(detectedBoxList[j])
        imageForShow = CF.drawBoxOnCameraImage(imageForShow, umpireBox, color=(0, 0, 0))
        
        # for tmpPlayer in defPoseList:
        #     print("Position :",tmpPlayer.positionID, " playerID :", tmpPlayer.playerID, " player.detectedID :", tmpPlayer.player.detectedID)
        print("Defender : Position :",self.defPoseList[0].positionID, " playerID :", self.defPoseList[0].playerID, " player.detectedID :", self.defPoseList[0].player.detectedID)
        for tmpPlayer in self.ofsPoseList:
            if tmpPlayer.playerID > -1:
                print("Offender : Position :",tmpPlayer.positionID, " playerID :", tmpPlayer.playerID, " player.detectedID :", tmpPlayer.player.detectedID)
        
        # cv2.imshow("Camera", imageForShow)
        # cv2.imshow("Field", tmpField)
        self.cameraShow = imageForShow.copy()
        self.fieldShow = tmpField.copy()
        end_time = time.time()
        print("----- %s seconds with detection -----" % (end_time - start_time))
        cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Field", cv2.WINDOW_NORMAL)
        cv2.imshow("Camera", imageForShow)
        cv2.imshow("Field", tmpField)
        cv2.waitKey(1)
        # [[x, y], trackerID, "", defend0Attack1Judge2]
        results = CF.compute_result(self.defPoseList, self.ofsPoseList, self.NAS_config)
        return results
    
    def set_reset(self):
        self.defPoseList = CF.setupDefensivePose(self.NAS_config)
        self.ofsPoseList = CF.setupOffensivePose(self.NAS_config)
        self.isResetPressed = True
        pass
    
    def get_frame(self):
        return self.cameraShow, self.fieldShow
    
    def test_setup(self):
        # TODO: delete
        print("TEST START...")
        self.imageNames = CF.retrieveImagesFromFolder(PATH_OF_IMAGE_FOLDER)
        for i in range(len(self.imageNames)):
            self.cameraImage = cv2.imread(self.imageNames[i])
            # Get player position on camera
            self.playerInfoList = CF.findPlayerPoseFromTxt(self.imageNames[i])
            self.execute()
            
    def test_start(self):
        t = threading.Thread(target=self.test_setup)
        t.start()
    
    def test_stop(self):
        t.stop()
    
    
    
def main():
    q = queue.Queue(maxsize = 10)  
    CS = Camera_System(q)
    
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Field", cv2.WINDOW_NORMAL)
    cv2.waitKey(0)
    
    CS.test_start()
    
    while True:
        cameraShow, fieldShow = CS.get_frame()
        if cameraShow.size == 0 or fieldShow.size == 0:
            continue
        cv2.imshow('Camera', cameraShow)
        cv2.imshow('Field', fieldShow)
        k = cv2.waitKey(1000)
        if k == 27:
            cv2.destroyAllWindows()
            break
        elif k == ord('r'):
            print("Reinitializing defenders!!!")
            CS.set_reset()
    pass

if __name__ == '__main__':
    main()
