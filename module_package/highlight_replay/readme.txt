#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 16:09:01 2019

@author: rayhliu
"""


STEP1: Set replay Base
    replayId = '1B'

STEP2: Initial Highlight_Repleay, if you would like to save highlight frame, set saveHighLight for  True
    HR = Highlight_Repleay(replayId=replayId, saveHighLight=False)

STEP3: Generate frame data and bbox info as "exportedData"
    bbox_info:[[mix,miny,maxx,maxy,int(class)],[mix,miny,maxx,maxy,int(class)],...]
    frame = cv2.imread(imagePath)
    frame_info = [frame,imagePath]
    exportedData = [frame_info,bbox_info]

STEP4: Put exportedData into Highlight_Repleay. verbose is 1: show detect imformation
    HR.receive_frame_info(count,exportedData,verbose=1)