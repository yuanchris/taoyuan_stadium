#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 18:51:59 2019
2019-10-15:
    1. edit base1  & base2 region
    2. debug: is_in_region
@author: rayliu
"""

import numpy as np
from PIL import Image
import pickle
import os
import cv2
from datetime import datetime
import time
import glob

YOLO = None
"""" save high light parm """
SHOW_REPLAY_FRAME = False
SAVE_FRAME_COUNT = 30
HIGHTLIGHT_ROOT = './high_light_dir'

""" Condition for detect """
CONDICTION1_COUNT = 4
CONDICTION2_COUNT = 3

class Highlight_Repleay:
    """detect hight light base with frame
    args:
        replayId:        <str>   default:'1B'; ex:'1B', '2B', '3B', 'HB'. Edit base parameter on "self.get_region_param".
        1B : NAS 2
        2B : NAS 4
        3B : NAS 12
        HB : NAS 3
        saveHighLight:   <bool>  default: False. Save high light frame or not.
    """
    def __init__(self,replayId='1B',saveHighLight = False):
        self.replayId = replayId
        self.saveHighLight = saveHighLight
        self.demoFrame = None
        self.showFrame = None
        self.bbox_info = None
        self.lastSaveHightlight = None

        """ tmp saved frmae list & set tmp saved frmae list size """
        self.frameList = []
        self.maxNumberOfSaved = 200

        """ set number of hightlight frame would to save """
        self.numberOfSavedFrame = 30

        self.savedHighLightFrameList = []
        self.conditionRegionPointsInfo = []
        self.conditionRegion1 = 0
        self.conditionRegion2 = 0
        self.conditionFlag1 = False
        self.conditionFlag2 = False
        self.lastTrackingPoint = None
        self.initTrackingPoint =  []
        self.currentCount = None

        self.recordFlag = False
        self.countMissPointInRegion1 = 0


        self.replayFrameCrop,self.spotBottomL,self.spotBottomR,self.spotTopL,self.spotTopR = self.get_region_param()

        self.regionMask,self.regionCropPoints = self.get_region_mask([self.spotTopL,
                                                                       self.spotBottomL,
                                                                       self.spotBottomR,
                                                                       self.spotTopR])

        if self.replayId == 'HB':
            initX = self.spotBottomL[0]
            self.initTrackingPoint.append((initX,self.spotTopL[1]+int(round((self.spotBottomL[1]-self.spotTopL[1])/4))))
            self.initTrackingPoint.append((initX,self.spotTopL[1]+int(round((self.spotBottomL[1]-self.spotTopL[1])*2/4))))
            self.initTrackingPoint.append((initX,self.spotTopL[1]+int(round((self.spotBottomL[1]-self.spotTopL[1])*3/4))))
            self.runDis = 100

            # ver modify cam
            self.trackingDistance = 120
            
        elif self.replayId == '2B':
            self.initTrackingPoint.append((self.spotBottomR[0]-150,int(round((self.spotBottomR[1]+self.spotTopR[1])/2))))
            self.trackingDistance = 190
            self.runDis = 166
            
        else: 
            if self.replayId == '3B':
                self.initTrackingPoint.append((self.spotBottomR[0]-90,int(round((self.spotBottomR[1]+self.spotTopR[1])/2))))
                self.runDis = 100
                self.trackingDistance = 90

            elif self.replayId == '1B':
                self.initTrackingPoint.append((self.spotBottomR[0]-70,int(round((self.spotBottomR[1]+self.spotTopR[1])/2))))
                self.runDis = 80
                self.trackingDistance = 150

            else:
                raise ValueError('no id:{}'.format(self.replayId))

            
            

    def get_region_param(self):
        """get condition region parameter

        args:
            replayId:           <str>    default:1B
        return:
            replayFrameCrop:    <list>   replay Crop frame range. ex:[minx, miny, maxx, maxy]
            spotButtomL         <tuple>  coordinate of region bottom left
            spotBottomR         <tuple>  coordinate of region bottom right
            spotTopL            <tuple>  coordinate of region top left
            spotTopR            <tuple>  coordinate of region top right
        """
        if self.replayId == '1B':
            replayFrameCrop = [1900,1170,3100,1770] #(1200*600)
            # ver1. 
            # spotBottomL,spotBottomR = (2550,1409),(2837,1456)
            # spotTopL,spotTopR = (2612,1377),(2837,1410) # wide:(2612,1377),(2837,1410); narrow:(2620,1368),(2841,1401)

            # ver3. 191015
            # spotBottomL,spotBottomR = (2623,1364),(2836,1398) 
            # spotTopL,spotTopR = (2559,1409),(2836,1453)
            
            # ver4 191017 modify cam
            # spotBottomL,spotBottomR = (1894,1071),(2986,1248)
            # spotTopL,spotTopR = (2141,937),(2975,1065) 

            #ver4-2
            # spotBottomL,spotBottomR = (2102,1098),(3168,1262)
            # spotTopL,spotTopR = (2188,915),(3207,1084)

            #ver4-3
            spotBottomL,spotBottomR = (2098,1105),(3167,1272)
            spotTopL,spotTopR = (2188,915),(3207,1084)

        elif self.replayId == '2B':
            replayFrameCrop = [1630,475,2830,1075] #(1200*600)
            # spotBottomL,spotBottomR = (2014,728),(2289,757)    
            # spotTopL,spotTopR =  (2067,684),(2273,706) 
            
            spotBottomL,spotBottomR = (2224,1037),(4084,1288)    
            spotTopL,spotTopR =  (2269,797),(4084,1054) 

        elif self.replayId == '3B':
            replayFrameCrop = [1370,840,2570,1440] #(1200*600)
            # ver1. 
            # spotBottomL,spotBottomR = (1739,1042),(2042,1044)
            # spotTopL,spotTopR = (1819,976),(2056,984) # arrow:(1818,994),(2054,994)
            
            # ver2. 
            # spotBottomL,spotBottomR = (1884,974),(2125,974) 
            # spotTopL,spotTopR = (1845,1024),(2155,1030) 
            
            # ver4 191017 modify cam
            # spotBottomL,spotBottomR = (2902,1005),(3840,976)
            # spotTopL,spotTopR = (2508,1166),(3737,1171)  

            #ver4-2
            spotBottomL,spotBottomR = (2782,1052),(4027,1031)
            spotTopL,spotTopR = (2744,1170),(4065,1171)

        elif self.replayId == 'HB':
            replayFrameCrop = [1060,440,3460,1540] #(2400*1100)
            # ver run: 
            # spotBottomL,spotBottomR = (1686,1142),(2200,1188)
            # spotTopL,spotTopR = (1646,942),(2409,1016)
            
            # ver4 191017 modify cam
            # spotBottomL,spotBottomR = (807,1293),(1926,1268)
            # sspotTopL,spotTopR = (726,825),(1951,923)

            #ver4-2
            spotBottomL,spotBottomR = (817,1177),(1949,1180)
            spotTopL,spotTopR = (726,764),(2023,913)

        elif self.replayId == 'HB2':
            replayFrameCrop = [1060,440,3460,1540] #(2400*1100)
            spotBottomL,spotBottomR = (1686,1142),(2200,1188)
            spotTopL,spotTopR = (1646,942),(2212,998)

        else:
            raise ValueError ('Cannt get {} region parameter'.format(self.replayId))

        return replayFrameCrop,spotBottomL,spotBottomR,spotTopL,spotTopR


    def receive_frame_info(self,count,exportedData,verbose=0):
        """receive frame info and detect HightLight frame

        args:
            count:           <int>   processed count
            exportedData:    <list>  [[frame array, frame path], bbox info]
            verbose:         <int>   0: no show log; 1:show log
        return:
            returnInfo:      <list>  ['1',image file name] or [False,None]
        """

        imageName = exportedData[0][1]
        self.bbox_info = exportedData[1]
        self.demoFrame = exportedData[0][0]
        self.currentCount = count
        self.recordFlag = False
        self.excute()

        if self.recordFlag != False:
            returnInfo = [self.recordFlag,os.path.basename(imageName)]
            print ('{} >> {}'.format(self.replayId,returnInfo))
        else:
            returnInfo = [self.recordFlag,None]

        if verbose == 2:
            print ('{} HihtLight Replay is running. {}'.format(self.replayId,returnInfo))
            print ('{} >> {}\n{}\nlastSaveHightlight:{}\nHighLight_frame:{}\nc1f:{} c1:{} \nc2f:{} c2:{}\n{}\n'.format(
                    self.currentCount,
                    returnInfo,
                    os.path.basename(imageName),
                    self.lastSaveHightlight,
                    len(self.savedHighLightFrameList),
                    self.conditionFlag1,
                    self.conditionRegion1,
                    self.conditionFlag2,
                    self.conditionRegion2,
                    self.conditionRegionPointsInfo
                    ))
        elif verbose == 1:   
            with open('highlight_log.txt',"a") as fp:
                fp.write('{} >> {}\n{}\nlastSaveHightlight:{}\nHighLight_frame:{}\nc1f:{} c1:{} \nc2f:{} c2:{}\n{}\n\n'.format(
                        self.replayId,
                        returnInfo,
                        os.path.basename(imageName),
                        self.lastSaveHightlight,
                        len(self.savedHighLightFrameList),
                        self.conditionFlag1,
                        self.conditionRegion1,
                        self.conditionFlag2,
                        self.conditionRegion2,
                        self.conditionRegionPointsInfo
                        ))

        
        return returnInfo

    def excute(self):
        """回放設計：
        處理 [N-1壘 >>> N壘]
        觸發條件＆輸出偵測：
            1.   三寸線上有<條件框1>往一壘跑，框內持續偵測到3偵都有點，則觸法flag1
            1-1. 在框內有偵測到點則開始紀錄HightLight Frame
            2.   在條件框1內追到追蹤點（模擬跑者），且持續往一壘方向跑，持續5幀則觸法flag2
            3.   當flag1＆flag2皆為True，且紀錄的HighLight frames達30偵，則保存這筆資料

        """
        self.update_tmp_frame_list()
        self.update_condition1()
        self.update_condition2()

        canSaveHightLightFlag = False
        if self.conditionFlag1 and self.conditionFlag2:
            canSaveHightLightFlag = True
            self.recordFlag = '1'
            self.conditionFlag1 = False
            self.conditionFlag2 = False
            self.conditionRegion1 = 0
            self.conditionRegion2 = 0
            self.lastTrackingPoint = None # avoid repeatly send signal from the play
            
        self.save_highlight_frame_controller(canSaveHightLightFlag)

    def _two_points_length(self,p0,p1):
        return pow(pow(p0[0]-p1[0],2)+pow(p0[1]-p1[1],2),0.5)

    def is_point_inside_region(self,regionMask,regionCropPoints,point):
        """ detect the point is in the condition region

        args:
            regionMask:        <array>  region frame mask
            regionCropPoints:  <lsit>   [mixx,miny,maxx,maxy]
            point:             <tuple>  coordinate detect point
        return:
            isInsideRegion:    <bool>   True or False
        """

        isInsideRegion = False
        if (point[0] < regionCropPoints[2] and point[0] > regionCropPoints[0] and
            point[1] < regionCropPoints[3] and point[1] > regionCropPoints[1]):
            trnsferFootPoint = np.array(point) - np.array(regionCropPoints[:2])
            if regionMask[trnsferFootPoint[1],trnsferFootPoint[0]] > 0:
                isInsideRegion = True
        return isInsideRegion

    def get_region_mask(self,regionPoints):
        """ get the condition region mask

        args:
            regionPoints:      <list>     [region_topLeft_coord,region_bottomLeft_coord,region_bottomRight_coord,region_topRight_coord]
        return:
            mask:              <array>   condition region mask
            regionCropPoints   <list>    [minx,miny,maxx,maxy]
        """

        x_values = [v[0] for v in regionPoints]
        y_values = [v[1] for v in regionPoints]
        min_x_value = min(x_values)
        min_y_value = min(y_values)
        max_x_value = max(x_values)
        max_y_value = max(y_values)

        regionCropPoints = (min_x_value,min_y_value,max_x_value,max_y_value)
#        region_image = orig_image[min_y_value:max_y_value, min_x_value:max_x_value,:].copy()
        mask = np.zeros((max_y_value-min_y_value,max_x_value-min_x_value,3),np.uint8)
        points = np.array(regionPoints)-np.array([[min_x_value,min_y_value ] for _ in range(len(regionPoints))])
        mask = cv2.fillPoly(mask, [points], (255, 255, 255))[:,:,2]
#        region_image = cv2.bitwise_and(region_image,region_image, mask=mask)
        return mask,regionCropPoints

    def update_condition1(self):
        """ update condition1 releated parameter

        """
        self.conditionRegionPointsInfo = []
        for bbox in self.bbox_info:
            # print("bbox before :", bbox)
            bbox = list(map(int,bbox.split(' ')))[2:]
            bbox = [bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3],bbox[4]]
            # print("bbox after :", bbox)
            classLable = bbox[4]
            footPoint = (int(round((bbox[0]+bbox[2])/2)),bbox[3])

            isInsideRegion1 = self.is_point_inside_region(self.regionMask,self.regionCropPoints,footPoint)
            if isInsideRegion1 and classLable == 0:
                if self.lastTrackingPoint == None:
                    for point in self.initTrackingPoint:
                        self.conditionRegionPointsInfo.append((footPoint,point,self._two_points_length(footPoint,point)))
                else:
                    self.conditionRegionPointsInfo.append((footPoint,self.lastTrackingPoint,self._two_points_length(footPoint,self.lastTrackingPoint)))


        if len(self.conditionRegionPointsInfo) > 0:
            self.conditionRegion1 += 1
            self.countMissPointInRegion1 = 0
        else:
            self.conditionRegion1 = 0
            self.conditionFlag1 = False

            if self.countMissPointInRegion1 <= 3:
                self.countMissPointInRegion1 +=1

        # start conditionFlag1
        if self.conditionRegion1 >= CONDICTION1_COUNT:
            self.conditionFlag1= True


    def update_condition2(self):
        """ update condition2 releated parameter

        """
        # print ('lastTrackingPoint:',self.lastTrackingPoint)
        isForward = False
        trackingPointInfo = None
        if len(self.conditionRegionPointsInfo) > 0:
            if self.lastTrackingPoint != None:
                if self.replayId == 'HB':
                    trackingPointInfos = [v for v in self.conditionRegionPointsInfo if v[0][0] > self.lastTrackingPoint[0]]
                else:
                    trackingPointInfos = [v for v in self.conditionRegionPointsInfo if v[0][0] < self.lastTrackingPoint[0]]

                if len(trackingPointInfos) > 0 :
                    matchTrackingPointInfos = [v for v in trackingPointInfos if v[2] > self.runDis]
                    if len(matchTrackingPointInfos) > 0:
                        trackingPointInfo = min(matchTrackingPointInfos, key=lambda x:x[2])
                        isForward = True
                    else:
                        trackingPointInfo = min(trackingPointInfos, key=lambda x:x[2])

                    # update lastTrackingPoint
                    self.lastTrackingPoint = trackingPointInfo[0]

                    if isForward != True and self._two_points_length(trackingPointInfo[0],self.initTrackingPoint[0]) > self.trackingDistance:
                        self.lastTrackingPoint = None

            else:
                trackingPointInfo = min(self.conditionRegionPointsInfo, key=lambda x:x[2])
                if trackingPointInfo[2] < self.trackingDistance:
                    if self.replayId == 'HB':
                        if trackingPointInfo[1][0] > trackingPointInfo[0][0]:
                            self.lastTrackingPoint = trackingPointInfo[1]
                        else:
                            self.lastTrackingPoint = trackingPointInfo[0]
                    else:
                        if trackingPointInfo[1][0] < trackingPointInfo[0][0]:
                            self.lastTrackingPoint = trackingPointInfo[1]
                        else:
                            self.lastTrackingPoint = trackingPointInfo[0]
                else:
                    self.lastTrackingPoint = None
        else:
            self.conditionFlag2 = False

        # 連續兩幀沒偵測到則判定區塊內沒有點了
        if self.countMissPointInRegion1 >=2:
            self.lastTrackingPoint = None
            self.conditionRegion2 = 0

        # print ('trackingPointInfo:',trackingPointInfo)
        # print ("--------------------")
        # print ('isForward:',isForward)
        if isForward: # 有往前進
            self.conditionRegion2 += 1
        else:
            self.conditionRegion2 = 0


        if self.conditionRegion2 >= CONDICTION2_COUNT:
            self.conditionFlag2 = True


    def update_tmp_frame_list(self):
        """ update tmp frame list

        update:
            self.frameList  <list>  [[hightLight_frmae1,time], [hightLight_frame2,time], ...]
        """
        now = datetime.now()
        now = now.strftime("%Y%m%d_%H-%M-%S.%f")
        replayFrame = self.get_replay_frame()

        self.frameList.append([replayFrame,now])
        if len(self.frameList) > self.maxNumberOfSaved:
            self.frameList.pop(0)

    def save_highlight_frame_controller(self,canSaveHightLightFlag):
        """ decide to save highlight frame or not

        """
        if canSaveHightLightFlag and self.saveHighLight:
            if len(self.frameList)-int(self.numberOfSavedFrame/2) > 0:
                self.savedHighLightFrameList = self.frameList[len(self.frameList)-int(self.numberOfSavedFrame/2):]
            else:
                self.savedHighLightFrameList = self.frameList[:]

        if self.savedHighLightFrameList != [] and len(self.savedHighLightFrameList) < self.numberOfSavedFrame:
            self.savedHighLightFrameList.append(self.frameList[-1])

        if len(self.savedHighLightFrameList) == 30:
            self.output_hightlight_frames()
            self.savedHighLightFrameList = []


    def output_hightlight_frames(self):
        """ save highlight frames

        """
        if len(self.savedHighLightFrameList) != 0:
            startTimeStr = self.savedHighLightFrameList[0][1].split('_')[1]
            endTimeStr = self.savedHighLightFrameList[-1][1].split('_')[1]
            saveDir = os.path.join(HIGHTLIGHT_ROOT,self.replayId+'_'+startTimeStr+'-'+endTimeStr)
            if not os.path.exists(saveDir):
                os.makedirs(saveDir)
            self.lastSaveHightlight = self.replayId+'_'+startTimeStr+'-'+endTimeStr
            for v in self.savedHighLightFrameList:
#                Image.fromarray(v[0]).save(os.path.join(saveDir,v[1]+'.jpg'))
                cv2.imwrite(os.path.join(saveDir,v[1]+'.jpg'),v[0])
        else:
            raise ValueError('There isnt any hightlight frames in savedHighLightFrameList')


    def get_replay_frame(self):
        """ from input frame to return replay frame

        """
        replayFrame = self.demoFrame[self.replayFrameCrop[1]:self.replayFrameCrop[3],
                                     self.replayFrameCrop[0]:self.replayFrameCrop[2],:].copy()
        return replayFrame


class Data_Contrller:
    def __init__(self,replayId):
        self.yoloResult = []
        self.pklFileNames = []
        self.hightCount = None
        self.imageDirPath = None
        self.get_init_para()
        self.totalFiles = list(sorted(glob.glob(self.imageDirPath+'*.jpg')))
        if len(self.totalFiles) == 0:
            raise ValueError ('no found any files in {}'.format(self.imageDirPath))

        if len(self.pklFileNames) > 0:
            self.read_yolo_pkl()

    def get_init_para(self):
        if replayId == '1B':
            self.hightCount = 0 # 1B count:907 捕手漏接跑向一壘
            # self.pklFileNames = ['./190929_NAS/yolov3_labels/NAS_02_20190929-19-41.pkl','./190929_NAS/yolov3_labels/NAS_02_20190929-19-42.pkl']
            # self.imageDirPath = './190929_NAS/NAS02/20190929_19_41-42/'
            self.imageDirPath = './NAS_data_new/NAS02_20191020_19-01/'
            # self.imageDirPath = './NAS_data_new/NAS02_20191020_19/'
            # self.imageDirPath = './NAS_data_new/test3-3/'

        elif replayId == '2B':
            self.hightCount = 100 # 1B count:907 捕手漏接跑向一壘
            # self.pklFileNames = ['./190929_NAS/yolov3_labels/NAS_04_20190929-19-37.pkl']
#            self.imageDirPath = './190929_NAS/NAS04/20190929_19_41-42/'
            self.imageDirPath = './NAS_data/NAS04_20191014_20-12/'
            self.imageDirPath = './NAS_data_new/NAS04_20191030_10-00/'
            

        elif replayId == '3B':
            self.hightCount = 0 # 1B count:907 捕手漏接跑向一壘
            #self.pklFileNames = ['./190929_NAS/yolov3_labels/NAS_12_20190929-19-41.pkl','./190929_NAS/yolov3_labels/NAS_12_20190929-19-42.pkl']
#            self.imageDirPath = './190929_NAS/NAS12/20190929_19_41-42/'
            self.imageDirPath = './NAS_data_new/test3-3/'
            self.imageDirPath = './NAS_data_new/NAS12_20191020_19-01/' 

        elif replayId == 'HB':
            self.hightCount = 400 # 1B count:907 捕手漏接跑向一壘
            # self.pklFileNames = ['./191001_NAS/NAS03_20191001_18-38.pkl','./191001_NAS/NAS03_20191001_18-39.pkl']
#            self.imageDirPath = './191001_NAS/NAS03_20191001_18_38-39/'
            self.imageDirPath = './NAS_data_new/NAS03_20191020_19-01/' 
            #self.imageDirPath = './NAS_data_new/test3-3/'

        elif replayId == 'HB2':
            self.hightCount = 0
            self.imageDirPath = './191001_NAS/NAS01_20191001_18_39/'

    def read_yolo_pkl(self):
        def _read_pickleFile(file_path):
            objects = []
            with (open(file_path, "rb")) as openfile:
                while True:
                    try:
                        objects.append(pickle.load(openfile))
                    except EOFError:
                        break
            return objects

        if len(self.pklFileNames) > 0:
            for pklFileName in self.pklFileNames:
                 self.yoloResult += _read_pickleFile(pklFileName)[0]

    def export_image(self,count):
        if count < len(self.totalFiles):
            # return Image.open(self.totalFiles[count]) # self.totalFiles[count]
            # return [self.totalFiles[count],cv2.imread(self.totalFiles[count])]
            return self.totalFiles[count]
        else:
            return False

    def export_image_with_bbox(self,count,yoloResult)   :
        """Control output image and detect result
            args:
                count:       <int>
                yoloResult:  <list> [[image_name1,bbox_info], [image_name2,bbox_info],...]
            outputs:
                output:      <list> [array(image),[[minx,miny,maxx,maxy,class_lable],[minx,miny,maxx,maxy,class_lable],...]]

        """

        if count < len(self.yoloResult):
            image_name = self.yoloResult[count][0]
            if image_name[-4:] != '.jpg':
                image_name+'.jpg'
            frame = np.array(Image.open(os.path.join(self.imageDirPath,image_name)))
            detectResult = self.yoloResult[count][1:]
            output =  (frame,detectResult)
        else:
            print ('there are not any yoloResult!')
            output = False
        return output



def show_image(name,image,size=(720,640)):
    cv2.namedWindow(name,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, size[0],size[1])
    cv2.imshow(name,image)

def show_demo(demoFrame,HR):
    replayFrame = demoFrame[HR.replayFrameCrop[1]:HR.replayFrameCrop[3],
                            HR.replayFrameCrop[0]:HR.replayFrameCrop[2],:].copy()

    cv2.line(demoFrame, HR.spotBottomL, HR.spotBottomR, (0, 0, 255), 5)
    cv2.line(demoFrame, HR.spotTopL, HR.spotTopR, (0, 0, 255), 5)
    
    
    for initPoint in HR.initTrackingPoint:
        cv2.circle(demoFrame,initPoint,8,(0,255,0),-1)
        cv2.circle(demoFrame, initPoint,HR.trackingDistance,(0,255,0),2)

    for bbox in bbox_info:
        bbox = list(map(int,bbox.split(' ')))[2:]
        bbox = [bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3],bbox[4]]
        classLable = bbox[4]
        footPoint = (int(round((bbox[0]+bbox[2])/2)),bbox[3])
        midPoint = (int(round((bbox[0]+bbox[2])/2)),int(round((bbox[1]+bbox[3]))/2))

        if classLable == 0:
            color = (0, 255, 0)
        else:
            color = (0,0,0)

        cv2.rectangle(demoFrame, tuple(bbox[:2]), tuple(bbox[2:4]), color, 2)
        cv2.circle(demoFrame,footPoint,8,(0,0,255),-1)
        cv2.circle(demoFrame,midPoint,8,(255,0,0),-1)
    
    if SHOW_REPLAY_FRAME:
        show_image('highlight',replayFrame)
    show_image('demo',demoFrame)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        return False
    else:
        return True

def save_YOLO_result(fileName,imagePath):
    print ('save yolo bbox_info count:',str(count))
    bbox_info = YOLO.detect_img(yolov3, frame, save_img=False)
    save_list = [imagePath]
    for v in bbox_info:
        save_list.append(v)
    pickle.dump(save_list,open(fileName+'.pkl',"ab"))
    

def init_YOLO():
    global YOLO
    from yolov3_package_v2 import yolov3_predict_gpus_cv as YOLO
    yolov3 = YOLO.initial_model(model_name='yolov3', gpu_id=0)
    return yolov3

if __name__ == "__main__":

    """ set replay base """
    replayId = 'HB'

    """ init YOLO,  Highlight_Repleay, Data_Contrller """
    yolov3 = init_YOLO()
    HR = Highlight_Repleay(replayId=replayId, saveHighLight=False)
    DC = Data_Contrller(replayId)
    count = DC.hightCount

    while True:
        if count % 2 == 0:
            imagePath = DC.export_image(count)

            if imagePath != False:
                print ('count >>',count)
                """ generate input frame info:[frame array, image path] """
                frame = cv2.imread(imagePath)
                frame_info = [frame,imagePath]

                st_time = time.time()

                """ generate YOLO detect result, bbox_info:[[mix,miny,maxx,maxy,int(class)],[mix,miny,maxx,maxy,int(class)],...] """
                bbox_info = YOLO.detect_img(yolov3, frame, save_img=False)

                """ combine input frame info and YOLO detect result as exportedData. """
                exportedData = [frame_info,bbox_info]

                """ put exportedData and count input Highlight_Repleay """
                highlightFlag = HR.receive_frame_info(count,exportedData,verbose=2)
                # print ('timeTotal:',time.time()-st_time) # 處理時間約：0.000152s, YOLO=0.1s

                if show_demo(frame,HR) == False:
                    break

                count += 1
            else:
                break
        else:
            count += 1