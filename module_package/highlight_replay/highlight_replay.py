# -*- encoding: utf-8 -*-
'''
@File    :   highlight_replay.py
@Time    :   2021/06/25 23:02:49
@Author  :   MattLiu <RayHliu> 
@Version :   ver.2021
'''

import numpy as np
from PIL import Image
import os
import cv2
from datetime import datetime
from . import replay_parameters as rp 

HIGHLIGHT_FRAMES_ROOT = './high_light_dir'

class Highlight_Repleay:
    """detect hight light base with frame
    args:
        replayId:        <str>   default:'1B'; ex:'1B', '2B', '3B', 'HB'. Create base parameter on replay_parameters.py.
        1B : NAS 2
        2B : NAS 4
        3B : NAS 12
        HB : NAS 3
        saveHighLight:   <bool>  default: False. Save high light frame or not.
    """
    def __init__(self,replayId='1B',saveHighLight = False):
        self.replayId = replayId
        assert self.replayId in rp.parameters.keys()
        self.replayParams = rp.parameters[self.replayId] 
        self.saveHighLight = saveHighLight
        self.demoFrame = None
        self.bbox_info = None
        self.lastSaveHightlight = None

        """ tmp saved frmae list & set tmp saved frmae list size """
        self.frameList = []
        self.maxNumOfSaved = 50
        self.numberOfSavedFrame = 30

        self.savedHighLightFrameList = []
        self.conditionRegionPointsInfo = []
        self.conditionRegion1 = 0
        self.conditionRegion2 = 0
        self.conditionFlag1 = False
        self.conditionFlag2 = False
        self.isForward = False
        self.lastTrackingPoint = None
        self.initTrackingPoint =  []
        self.currentCount = None

        self.recordFlag = False
        self.recordTime = 'None'
        self.countMissPointInRegion1 = 0
        self.countUnForward = 0

        # Force out params
        self.countC1C2 = 0
        self.countC3 = 0
        self.isOnBase = False
        self.numOfFRPlayer = 0

        # start controller flag
        self.start_flag = True

        # TODO player recognization
        # self.trackingPlayerImg = None
        # self.forcePlayerImg = None

        if self.replayId == 'HB':
            initX = self.replayParams['detect_c12_bottom_left'][0]
            self.initTrackingPoint.append((initX, self.replayParams['detect_c12_top_left'][1]+int(round((self.replayParams['detect_c12_bottom_left'][1]- self.replayParams['detect_c12_top_left'][1])/4))))
            self.initTrackingPoint.append((initX, self.replayParams['detect_c12_top_left'][1]+int(round((self.replayParams['detect_c12_bottom_left'][1]- self.replayParams['detect_c12_top_left'][1])*2/4))))
            self.initTrackingPoint.append((initX, self.replayParams['detect_c12_top_left'][1]+int(round((self.replayParams['detect_c12_bottom_left'][1]- self.replayParams['detect_c12_top_left'][1])*3/4))))
        elif self.replayId == '1B_v3':
            self.initTrackingPoint.append((self.replayParams['detect_c12_bottom_right'][0]-self.replayParams['init_tracking_distance'],int(round((self.replayParams['detect_c12_bottom_right'][1]+self.replayParams['detect_c12_top_right'][1])/2))))
        else:
            self.initTrackingPoint.append((self.replayParams['detect_c12_bottom_right'][0],int(round((self.replayParams['detect_c12_bottom_right'][1]+self.replayParams['detect_c12_top_right'][1])/2))))

        self.regionMask,self.regionCropPoints = self.get_region_mask([self.replayParams['detect_c12_top_left'],self.replayParams['detect_c12_bottom_left'],self.replayParams['detect_c12_bottom_right'],self.replayParams['detect_c12_top_right']])

        if self.replayParams['base_region'] is not None:
            self.baseMask,self.baseCropPoints = self.get_region_mask(self.replayParams['base_region'])
        
        if self.replayParams['force_region'] is not None:
            self.frMask,self.frCropPoints = self.get_region_mask(self.replayParams['force_region'])


    def receive_frame_info(self,count,exportedData,verbose=0):
        """receive frame info and detect HightLight frame
        args:
            count:           <int>   processed count
            exportedData:    <list>  [[frame array, frame path], bbox info, pause_flag]
            verbose:         <int>   0: no show log; 1:show log
        return:
            returnInfo:      <list>  ['1',image file name] or [False,None]
        """

        imageName = exportedData[0][1]
        self.start_flag = exportedData[2] # True:Start, False:Pause
        self.bbox_info = self._convert_boxInfo(exportedData[1])
        self.demoFrame = exportedData[0][0]
        self.currentCount = count
        self.recordFlag = False
        if self.start_flag == False:
            self.conditionFlag1 = False
            self.conditionFlag2 = False
            self.conditionRegion1 = 0
            self.conditionRegion2 = 0
            self.lastTrackingPoint = None # avoid repeatly send signal from the play

            self.trackingPlayerImg = None
            self.forcePlayerImg = None
            self.countC1C2 = 0
            self.countC3 = 0
        else:
            self.excute()

        if self.recordFlag != False:
            returnInfo = [self.recordFlag,os.path.basename(imageName)]
            print ('{} >> {}'.format(self.replayId,returnInfo))
        else:
            returnInfo = [self.recordFlag,None]
        
        # show the current info
        if verbose == 2:
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

        print ('<< {} >> {} - Replay is running.'.format(self.replayId,count))
        
        return returnInfo

    def excute(self):
        self.update_tmp_frame_list()
        self.update_condition1()
        self.update_condition2()

        # << The replay bang is on when conditionFlag1 and conditionFlag2 are true. >>
        # canSaveHightLightFlag = False
        # if self.conditionFlag1 and self.conditionFlag2:
        #     canSaveHightLightFlag = True
        #     self.recordFlag = '1'
        #     self.recordTime = self.frameList[-1][1][9:]
        #     self.conditionFlag1 = False
        #     self.conditionFlag2 = False
        #     self.conditionRegion1 = 0
        #     self.conditionRegion2 = 0
        #     self.lastTrackingPoint = None # avoid repeatly send signal from the play
        
        # << update 0904 add forced tag: The replay bang is on when force tag >>
        c1c2Flag = False
        if self.conditionFlag1 and self.conditionFlag2:
            c1c2Flag = True
            self.countC1C2=1
        
        if self.lastTrackingPoint is not None and self.replayParams['force_tracking_over_info'][2](self.lastTrackingPoint[self.replayParams['force_tracking_over_info'][1]],self.replayParams['force_tracking_over_info'][0]) == True:
            if self.numOfFRPlayer == 0:
                self.countC3 = 0
                self.countC1C2 = 0

        if self.countC3 > 0 :
            if self.isOnBase:
                self.countC3 =1
            else:
                self.countC3+=1 
        
        if self.countC1C2 > 0 :
            if c1c2Flag:
                self.countC1C2=1
            else:
                self.countC1C2+=1 
        
        if self.countC3 > 5: 
            self.countC3 = 0
            # self.forcePlayerImg = None

        if self.countC1C2 > 5: 
            self.countC1C2 = 0
            # self.trackingPlayerImg = None 

        print ("[c12, c3]>>:",self.countC1C2,self.countC3)
        canSaveHightLightFlag = False
        if self.countC1C2 != 0 and self.countC3 !=0: 
            # TODO compare players
            # print ('self.forcePlayerImg:',self.forcePlayerImg is not None)
            # print ('self.trackingPlayerImg:',self.trackingPlayerImg is not None)
            # fpHist = self.get_player_color_hsv_his(self.forcePlayerImg)
            # tpHist = self.get_player_color_hsv_his(self.trackingPlayerImg)
            # pred = cv2.compareHist(tpHist,fpHist,cv2.HISTCMP_CORREL)
            # print ('player_recognization_pred:',pred)

            # init c1,c2 param
            self.conditionFlag1 = False
            self.conditionFlag2 = False
            self.conditionRegion1 = 0
            self.conditionRegion2 = 0
            self.lastTrackingPoint = None # avoid repeatly send signal from the play

            canSaveHightLightFlag = True
            self.recordFlag = '1'
            self.recordTime = self.frameList[-1][1][9:]
            self.trackingPlayerImg = None
            self.forcePlayerImg = None
            self.countC1C2 = 0
            self.countC3 = 0

        self.save_highlight_frame_controller(canSaveHightLightFlag)

    def _two_points_length(self,p0,p1):
        return pow(pow(p0[0]-p1[0],2)+pow(p0[1]-p1[1],2),0.5)
    
    def _convert_boxInfo(self,bbox_info):
        # receive bbox format: c_x, c_y, left, top, w, h, c, score
        allBboxInfo = []
        if len(bbox_info)>0:
            if isinstance(bbox_info[0],str):
                for bbox in bbox_info:
                    bbox = bbox.split(' ')
                    score = bbox[-1]
                    cls = bbox[-2]
                    bbox = list(map(int,bbox[:-2]))[2:]
                    bbox = [bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3],int(cls)]
                    footPoint = (int(round((bbox[0]+bbox[2])/2)),bbox[3])
                    bbox.append(footPoint)
                    bbox.append(float(score))
                    allBboxInfo.append(bbox)
            else:
                allBboxInfo = bbox_info
        return allBboxInfo
                
    def _is_point_inside_region(self,regionMask,regionCropPoints,point):
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
        # region_image = orig_image[min_y_value:max_y_value, min_x_value:max_x_value,:].copy()
        mask = np.zeros((max_y_value-min_y_value,max_x_value-min_x_value,3),np.uint8)
        points = np.array(regionPoints)-np.array([[min_x_value,min_y_value ] for _ in range(len(regionPoints))])
        mask = cv2.fillPoly(mask, [points], (255, 255, 255))[:,:,2]
        # region_image = cv2.bitwise_and(region_image,region_image, mask=mask)
        return mask,regionCropPoints

    def update_condition1(self):
        """ update condition1 parameters
        """
        # print ('>>> c1',self.lastTrackingPoint)
        self.conditionRegionPointsInfo = []
        self.isOnBase = False
        numOfForceRegionPlayer = 0
        for bbox in self.bbox_info:
            if (bbox[2]-bbox[0])*(bbox[3]-bbox[1]) > self.replayParams['bbox_area_threshold']:
                classLable = bbox[4]
                footPoint = bbox[5]

                isInsideRegion1 = self._is_point_inside_region(self.regionMask,self.regionCropPoints,footPoint)
                if isInsideRegion1 and classLable == 0:
                    """ Record the diff value of current points and last tracking points"""
                    if self.lastTrackingPoint == None:
                        for point in self.initTrackingPoint:
                            self.conditionRegionPointsInfo.append((footPoint,point,self._two_points_length(footPoint,point)))
                    else:
                        self.conditionRegionPointsInfo.append((footPoint,self.lastTrackingPoint,self._two_points_length(footPoint,self.lastTrackingPoint)))
                if self.replayId in ['1B','1B_v2','1B_v3','2B','3B']:
                    # self.isOnBase = self._is_point_inside_region(self.baseMask,self.baseCropPoints,bbox[2:4]) or self._is_point_inside_region(self.baseMask,self.baseCropPoints,footPoint)
                    self.isOnBase = (len(set(range(self.baseCropPoints[0],self.baseCropPoints[2])).intersection(set(range(footPoint[0],bbox[2]))))>1 and 
                                    len(set(range(self.baseCropPoints[1],self.baseCropPoints[3])).intersection(set(range(bbox[3]-(bbox[3]-bbox[1])//6,bbox[3]))))>1)
    
                else:        
                    self.isOnBase = (len(set(range(self.baseCropPoints[0],self.baseCropPoints[2])).intersection(set(range(bbox[0],bbox[2]))))>1 and 
                                    bbox[3] in range(self.baseCropPoints[1],self.baseCropPoints[3]))     

                if self.isOnBase: 
                    self.countC3 = 1
                    # TODO player recognization
                    # self.forcePlayerImg = self.demoFrame[bbox[1]:(bbox[1]+bbox[3])//2,bbox[0]+(bbox[2]-bbox[0])//3:bbox[2]-(bbox[2]-bbox[0])//3,:]
                
                if self._is_point_inside_region(self.frMask,self.frCropPoints,bbox[2:4]) or self._is_point_inside_region(self.frMask,self.frCropPoints,footPoint):
                    numOfForceRegionPlayer +=1 
        
        self.numOfFRPlayer = numOfForceRegionPlayer

        if len(self.conditionRegionPointsInfo) > 0:
            self.conditionRegion1 += 1
            self.countMissPointInRegion1 = 0
        else:
            self.conditionRegion1 = 0
            self.conditionFlag1 = False

            if self.countMissPointInRegion1 <= 3:
                self.countMissPointInRegion1 +=1

        # start conditionFlag1
        if self.conditionRegion1 >= self.replayParams['condiction1_count']:
            self.conditionFlag1= True

    def update_condition2(self):
        """ update condition2 parameters
        """
        self.isForward = False
        trackingPointInfo = None
        self.conditionFlag2 = False
        if len(self.conditionRegionPointsInfo) > 0:
            if self.lastTrackingPoint != None:
                """ Find the points that is in front of last tracking point """
                if self.replayId == 'HB':
                    forwardPoints = [v for v in self.conditionRegionPointsInfo if v[0][0] > self.lastTrackingPoint[0]]
                else:
                    forwardPoints = [v for v in self.conditionRegionPointsInfo if v[0][0] < self.lastTrackingPoint[0]]

                """ Find the forward points that fit the reqirements (run_distance_thres) """
                # print ('Good Forward Pts::',forwardPoints)
                if len(forwardPoints) > 0 :
                    fitForwardPoints = [v for v in forwardPoints if v[2] > self.replayParams['min_run_distance'] and v[2] < self.replayParams['max_run_distance']]
                    if len(fitForwardPoints) > 0:
                        trackingPointInfo = min(fitForwardPoints, key=lambda x:x[2])
                        self.isForward = True
                    else:
                        trackingPointInfo = min(forwardPoints, key=lambda x:x[2])
                        self.countUnForward += 1

                    # update lastTrackingPoint
                    self.lastTrackingPoint = trackingPointInfo[0]

                    # if self.isForward != True and self._two_points_length(trackingPointInfo[0],self.initTrackingPoint[0]) > self.replayParams['init_tracking_distance']:
                        # self.lastTrackingPoint = None

            else:
                trackingPointInfo = min(self.conditionRegionPointsInfo, key=lambda x:x[2])
                if trackingPointInfo[2] < self.replayParams['init_tracking_distance']:
                    self.lastTrackingPoint = trackingPointInfo[0]
                else:
                    self.lastTrackingPoint = None
        else:
            self.conditionFlag2 = False

        # 連續兩幀沒偵測到則判定區塊內沒有點了
        if self.countMissPointInRegion1 >=2:
            self.lastTrackingPoint = None
            self.conditionRegion2 = 0

        if self.isForward: # 有往前進
            self.conditionRegion2 += 1
        else:
            self.conditionRegion2 += 0
        
        if self.countUnForward >= 2:
            self.lastTrackingPoint = None
            self.countUnForward = 0
            self.conditionRegion2 = 0

        if self.conditionRegion2 >= self.replayParams['condiction2_count']:
            self.conditionFlag2 = True

            # TODO player recognization
            # bboxIdx = [v[5] for v in bbox_info].index(self.lastTrackingPoint)
            # tBBox = bbox_info[bboxIdx]
            # self.trackingPlayerImg = self.demoFrame[tBBox[1]:(tBBox[1]+tBBox[3])//2,tBBox[0]+(tBBox[2]-tBBox[0])//3:tBBox[2]-(tBBox[2]-tBBox[0])//3,:]
            
    def detect_coacher(self):
        """ TODO detect the coacher """
        pass
    
    def recognize_batboy(self):
        """ TODO detect the batboy"""
        pass

    def recognize_umpire_pose(self):
        """ TODO recognize the safe and out """
        pass

    def recognize_player(self):
        """ TODO recognize the player """
        pass

    def get_player_color_hsv_his(self,bboxImg):
        bboxHSV = cv2.cvtColor(bboxImg,cv2.COLOR_BGR2HSV)
        bboxHist = cv2.calcHist([bboxHSV], [0,1,2], None, [180,256,256], [0,180,0,256,0,256])
        norBBoxHist = cv2.normalize(bboxHist, None , 0, 1, cv2.NORM_MINMAX)          
        return norBBoxHist

    def update_tmp_frame_list(self):
        """ update tmp frame list

        update:
            self.frameList  <list>  [[hightLight_frmae1,time], [hightLight_frame2,time], ...]
        """
        now = datetime.now()
        now = now.strftime("%Y%m%d_%H-%M-%S.%f")
        replayFrame = self.get_crop_replay_frame()

        self.frameList.append([replayFrame,now])
        if len(self.frameList) > self.maxNumOfSaved:
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
            saveDir = os.path.join(HIGHLIGHT_FRAMES_ROOT,self.replayId+'_'+startTimeStr+'-'+endTimeStr)
            if not os.path.exists(saveDir):
                os.makedirs(saveDir)
            self.lastSaveHightlight = self.replayId+'_'+startTimeStr+'-'+endTimeStr
            for v in self.savedHighLightFrameList:
                # Image.fromarray(v[0]).save(os.path.join(saveDir,v[1]+'.jpg'))
                cv2.imwrite(os.path.join(saveDir,v[1]+'.jpg'),v[0])
        else:
            raise ValueError('There isnt any hightlight frames in savedHighLightFrameList')

    def get_crop_replay_frame(self):
        """ from input frame to return replay frame
        """
        replayFrame = self.demoFrame[self.replayParams['replay_frame_crop_region'][1]:self.replayParams['replay_frame_crop_region'][3],
                                     self.replayParams['replay_frame_crop_region'][0]:self.replayParams['replay_frame_crop_region'][2],:].copy()
        return replayFrame