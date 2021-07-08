# -*- encoding: utf-8 -*-
'''
@File    :   run.py
@Time    :   2021/06/25 23:00:58
@Author  :   MattLiu <RayHliu> 
@Version :   ver.2021
'''

import pickle
import cv2
import numpy as np 
import time

import replay_parameters as rp 
from highlight_replay import Highlight_Repleay
from data_controller import Data_Contrller


YOLO = None
USE_PKL_FILE = False
SHOW_BANG_FRAME = True
SAVE_BANG_FRAME = False
SAVE_VIDEO = False

if SAVE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('demo.mp4', fourcc, 10.0, (4096,  2160))

def show_image(name,image):
    cv2.namedWindow(name,cv2.WINDOW_NORMAL)
    cv2.imshow(name,image)

def show_demo(demoFrame,HR,bbox_info):
    """ Set detect area """
    cv2.line(demoFrame, HR.replayParams['detect_c12_bottom_left'], HR.replayParams['detect_c12_bottom_right'], (0, 0, 255), 5)
    cv2.line(demoFrame, HR.replayParams['detect_c12_top_left'], HR.replayParams['detect_c12_top_right'], (0, 0, 255), 5)
    cv2.line(demoFrame, HR.replayParams['detect_c12_bottom_left'], HR.replayParams['detect_c12_top_left'], (0, 0, 255),5)
    cv2.line(demoFrame, HR.replayParams['detect_c12_bottom_right'], HR.replayParams['detect_c12_top_right'], (0, 0, 255),5)
    
    for initPoint in HR.initTrackingPoint:
        cv2.circle(demoFrame,initPoint,8,(0,255,0),-1)
        cv2.circle(demoFrame, initPoint,HR.replayParams['init_tracking_distance'],(0,255,0),2)

    """ draw base detection """
    if HR.replayParams['base_region'] is not None:
        if HR.countC3!=0: force_base_color = (0,0,255)
        else: force_base_color = (255,255,0)
        cv2.polylines(demoFrame, [np.array(HR.replayParams['base_region'])], True, force_base_color, 4)

    """ draw base detection """
    if HR.replayParams['force_region'] is not None:
        cv2.polylines(demoFrame, [np.array(HR.replayParams['force_region'])], True, (255,144,30), 4)
    
    """ Get YOLO Result """
    bboxInfo = HR._convert_boxInfo(bbox_info)
    inRegionFootPoints = [v[0] for v in HR.conditionRegionPointsInfo]
    rt = 10
    for bbox in bboxInfo:
        """ Ignore samll bboxes """
        if (bbox[2]-bbox[0])*(bbox[3]-bbox[1]) < HR.replayParams['bbox_area_threshold']:
            continue
        
        classLable = bbox[4]
        """ show foot point and mid point """
        footPoint = (int(round((bbox[0]+bbox[2])/2)),bbox[3])
        midPoint = (int(round((bbox[0]+bbox[2])/2)),int(round((bbox[1]+bbox[3]))/2))
        

        """ Show the specify bboxes """
        if classLable == 0:
            # color = (255, 255, 255)
            if footPoint in inRegionFootPoints:
                color = (0,69,255)
                if footPoint == HR.lastTrackingPoint and HR.isForward is True:
                    cv2.rectangle(demoFrame, (bbox[0]+rt,bbox[1]+rt), (bbox[2]-rt,bbox[3]-rt), (255,144,30), rt)
            else:
                color = (255, 255, 255)
        else:
            color = (10,10,10)
        
        cv2.rectangle(demoFrame, tuple(bbox[:2]), tuple(bbox[2:4]), color, rt)
        cv2.circle(demoFrame,footPoint,15,(0,0,255),-1)
        cv2.circle(demoFrame,midPoint,15,(255,0,0),-1)
        
    
    """ Show process Info """
    show_process_info = True
    if show_process_info:
        size = demoFrame.shape
        infoBox = [(int(size[1]*7/8)-300,int(size[0]*3/4)), (size[1]-50,size[0]-50)]
        infoBoxW = infoBox[1][0]-infoBox[0][0] 
        infoBoxH = infoBox[1][1]-infoBox[0][1]
        # cv2.rectangle(demoFrame,infoBox[0],infoBox[1],(220,220,220),-1)

        info1 = 'Frame: %s'%HR.currentCount
        info2 = 'Cond1: %s'%HR.conditionRegion1
        info3 = 'Cond2: %s'%HR.conditionRegion2
        info4 = 'Highlight: %s'%HR.recordFlag
        info5 = HR.recordTime
        totalInfo = [info1,info2,info3,info4,info5]
        for infoIdx,info in enumerate(totalInfo):
            # print ('>>>>>>',info)
            if infoIdx == 1:
                if HR.conditionFlag1:
                    infoColor = (0,128,0)
                else:
                    infoColor = (0,0,200)
            elif infoIdx == 2:
                if HR.conditionFlag2:
                    infoColor = (0,128,0)
                else:
                    infoColor = (0,0,200)
            elif infoIdx == 3:
                if HR.recordFlag == '1':
                    infoColor = (0,128,0)
                else:
                    infoColor = (0,0,200)
            
            showPosit = (infoBox[0][0]+10,int(infoBox[0][1]+infoBoxH*(infoIdx+1)/(len(totalInfo)+1)))

            if infoIdx not in [0,4]:
                cv2.circle(demoFrame, (showPosit[0]+20,showPosit[1]-20), 25, infoColor,-1)
                showTxtPosit = (showPosit[0]+70,showPosit[1])
            else:
                showTxtPosit = showPosit
            cv2.putText(demoFrame, info, showTxtPosit, cv2.FONT_HERSHEY_DUPLEX,2, (34, 34, 34), 5, cv2.LINE_AA)
    

    if HR.start_flag == False:
        pause_img = np.zeros(demoFrame.shape,dtype=np.uint8)
        cv2.circle(pause_img,(pause_img.shape[1]//2,pause_img.shape[0]//2),pause_img.shape[0]//4,(255,255,255),10)
        cv2.putText(pause_img, 'PAUSE',(pause_img.shape[1]//2-pause_img.shape[0]//4+30,pause_img.shape[0]//2+150), cv2.FONT_HERSHEY_DUPLEX,10, (255,255,255), 10, cv2.LINE_AA)
        demoFrame = cv2.addWeighted(demoFrame,0.3,pause_img,0.7,1)
    
    if SAVE_VIDEO:
        out.write(demoFrame)
        
    if SAVE_BANG_FRAME:
        cv2.imwrite('./demo_images/'+str(HR.currentCount)+'.jpg',demoFrame)

    if SHOW_BANG_FRAME and HR.recordFlag == '1':
        # replayFrame = demoFrame[HR.replayParams['replay_frame_crop_region'][1]:HR.replayParams['replay_frame_crop_region'][3],
        #                         HR.replayParams['replay_frame_crop_region'][0]:HR.replayParams['replay_frame_crop_region'][2],:].copy()
        show_image('highlight',demoFrame)
        cv2.waitKey(0)

    show_image('demo',demoFrame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        return False
    else:
        return True

def init_YOLO():
    global YOLO
    from yolov3_package_v2 import yolov3_predict_gpus_cv as YOLO
    yolov3 = YOLO.initial_model(model_name='yolov3', gpu_id=1)
    return yolov3

def main():
    pause_range = [100,105]
    if USE_PKL_FILE:
        with open ('./data/20200628_1Base.pkl','rb') as file:
            allBoxInfo = pickle.load(file)

    """ set replay base """
    replayId = '1B'

    """ Init YOLO,  Highlight_Repleay, Data_Contrller """
    if not USE_PKL_FILE:
        yolov3 = init_YOLO()
    HR = Highlight_Repleay(replayId=replayId, saveHighLight=False)
    DC = Data_Contrller(replayId)
    count = DC.hightCount

    while True:
        st = time.time()
        if count % 1 == 0:
            imagePath = DC.export_image(count)

            if imagePath != False:
                print ('count >>',count)
                """ generate input frame info:[frame array, image path] """
                frame = cv2.imread(imagePath)
                if replayId == '1B_v2':
                    frame = frame[rp.crop_size[1]:,rp.crop_size[0]:,:].copy()
                elif replayId == '1B_v3':
                    frame = frame[rp.crop_size[1]:,:frame.shape[1]-rp.crop_size[0],:].copy()
                frame_info = [frame,imagePath]

                st_time = time.time()
                """ generate YOLO detect result, bbox_info:[[mix,miny,maxx,maxy,int(class)],[mix,miny,maxx,maxy,int(class)],...] """
                if USE_PKL_FILE:
                    bbox_info = allBoxInfo[imagePath]
                else:
                    bbox_info = YOLO.detect_img(yolov3, frame, save_img=False)

                """ start & pause """
                start_flag = True
                if count in range(pause_range[0],pause_range[1]):
                    start_flag = False
                    
                """ combine input frame info and YOLO detect result as exportedData. """
                input_data = [frame_info,bbox_info,start_flag]

                """ put input_data and count input Highlight_Repleay """
                highlightFlag = HR.receive_frame_info(count,input_data,verbose=2)
                print ('cost time:',time.time()-st_time) #0.000152s, YOLO=0.1s

                if show_demo(frame,HR,bbox_info) == False:
                    break
                count += 1
            else:
                break
        else:
            count += 1

    if SAVE_VIDEO:
        out.releas()

if __name__ == "__main__":
    main()
