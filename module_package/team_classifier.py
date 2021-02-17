#!/usr/bin/env python
# -*- coding: UTF-8 -*- 

'''
=========================================================
File Name   :   team_classifier.py
Copyright   :   Chao-Chuan Lu
Create date :   Sep. 9, 2019
Description :   Team classifier
=========================================================
'''

import math
import time
import cv2
import numpy as np

def calc_defenders_possibilities(ROIImageList, pitcherImage, threshold=0.2):
    """Calculate the similarities between every ROI image and pitcherImage.
    Threshold value :   1.0 -> most likely.
                        0.0 -> least likely.
    Return a list of boolean values based on the threshold value.
    
    Args:
        ROIImageList:       [np.array]
        pitcherImage:       np.array
        threshold:          float
    Returns:
        defenderList:       [boolean]
    """
    defenderList = []
    pitcherHueHist = get_hueHist(pitcherImage)
    for ROIImage in ROIImageList:
        tmpHueHist = get_hueHist(ROIImage)
        tmpScore = calc_hist_similarity(tmpHueHist, pitcherHueHist)
        if tmpScore < threshold:
            defenderList.append(False)
        else:
            defenderList.append(True)
    return defenderList
    

def get_hueHist(image):
    """Convert BGR image into HSV and get hue histagram.
    
    Args:
        image:       np.array
    Returns:
        hueHist:     np.array
    """
    hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hueHist = cv2.calcHist([hsvImage], [0], None, [180], [0, 180])
    return hueHist
    
def calc_hist_similarity(hist, referenceHist):
    """Calculate the similarity between hist and referenceHist.
    1. normalize hist
    2. compare hist
    
    HISTCMP_CORREL
    HISTCMP_CHISQR
    HISTCMP_INTERSECT
    HISTCMP_BHATTACHARYYA
    HISTCMP_HELLINGER       
    HISTCMP_CHISQR_ALT
    HISTCMP_KL_DIV
    
    Args:
        hist:              np.array
        referenceHist:     np.array
    Returns:
        similarity:        float
    """
    histNormed = cv2.normalize(hist, None, norm_type=cv2.NORM_MINMAX)
    refHistNormed = cv2.normalize(referenceHist, None, norm_type=cv2.NORM_MINMAX)
    similarity = cv2.compareHist(histNormed, refHistNormed, cv2.HISTCMP_CORREL)
    return similarity
