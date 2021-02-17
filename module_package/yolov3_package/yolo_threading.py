#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:54:36 2019

@author: rayliu
"""


import threading
import yolov3_predict_2 as YOLO
import os 
import datetime
import cv2
from PIL import Image

result1 = None
result2 = None
result3 = None


def yolo_tracking_thread(GPUID):
    yolov3 = YOLO.initial_model('yolov3',gpu_id=str(GPUID))
    path = "/home/osense/Desktop/smart_stadium/local_test"
    filenames = os.listdir(path)
    filenames.sort()
    filenames = filter(lambda x : -1 != x.find('jpg') and not x.startswith("."), filenames)
    first_file_time = ""
    start = datetime.datetime.now()
    for filename in filenames:
        # start = timeit.default_timer()
        if first_file_time == "":
            first_file_time = datetime.datetime.strptime(filename[:-4], '%Y%m%d_%H-%M-%S.%f')
        else:
            current_file_time = datetime.datetime.strptime(filename[:-4], '%Y%m%d_%H-%M-%S.%f')
            time_interval = datetime.datetime.now() - start
            file_interval = current_file_time - first_file_time
            time_interval =  int(time_interval.total_seconds() * 1000) - int(file_interval.total_seconds() * 1000) 
            print (GPUID,'---',time_interval)
            if time_interval > 100:
                continue
        # print (os.path.join(path, filename))
        image = Image.open(os.path.join(path, filename))
        # print ("image read :", image.size)
        yoloResults = YOLO.detect_img(yolov3, image)
        print("tracking result :", yoloResults)

def run():
   tracking_thread1 = threading.Thread(target=yolo_tracking_thread, args=(0,))
   tracking_thread1.start()

   tracking_thread2 = threading.Thread(target=yolo_tracking_thread, args=(1,))
   tracking_thread2.start()
""""
   tracking_thread3 = threading.Thread(target=yolo_tracking_thread, args=(2,))
   tracking_thread3.start()
"""
run()
   
   
