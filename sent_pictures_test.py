import sys,os
import time
import logging
import socket, json
import threading
from cv2 import cv2
from multiprocessing import Process
# 指定要查詢的路徑
imgPath124 = '/home/chris/Desktop/taoyuan/taoyuan_img/Taoyuan_highlight_replay_data/124'
imgPath125 = '/home/chris/Desktop/taoyuan/taoyuan_img/Taoyuan_highlight_replay_data/125'

imgPath128 = '/home/chris/Desktop/taoyuan/taoyuan_img/tracking/128'
imgPath130 = '/home/chris/Desktop/taoyuan/taoyuan_img/tracking/130'


# 列出指定路徑底下所有檔案(包含資料夾)

allFileList124 = os.listdir(imgPath124)
allFileList124.sort()
allFileList125 = os.listdir(imgPath125)
allFileList125.sort()

allFileList128 = os.listdir(imgPath128)
allFileList128.sort()
allFileList130 = os.listdir(imgPath130)
allFileList130.sort()
# path1 = '/home/chris/Downloads/pause_start_v1/pause_start_v1/pause_start_info/test_210417_19-02-24_19-03-15_128_west_info.txt'
# path2 = '/home/chris/Downloads/pause_start_v1/pause_start_v1/pause_start_info/test_210417_19-02-24_19-03-15_130_east_info.txt'
# allFileList = []
# allFileList2 = []
# with open(path1) as f:
#     for line in f.readlines():
#         s = line.split(',')
#         allFileList.append(s[0])

# with open(path2) as f:
#     for line in f.readlines():
#         s = line.split(',')
#         allFileList2.append(s[0])

def sent_pic124():
    for file in allFileList124:
        time.sleep(0.285)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('127.0.0.1', 8901)) # camera 124
            filePath = os.path.join(imgPath124,file)
            print(filePath)
            sock.send(filePath.encode('UTF-8'))
            sock.close()

            frame = cv2.imread(filePath)  
            frame = cv2.resize(frame, (1024, 540))  

            cv2.imshow('imageOne',frame)
            cv2.waitKey(1)

        except socket.error as e:
            print("[ERROR] ", e)  

def sent_pic125():
    for file in allFileList125:
        time.sleep(0.285)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('127.0.0.1', 8902)) # camera 124
            filePath = os.path.join(imgPath125,file)
            print(filePath)
            sock.send(filePath.encode('UTF-8'))
            sock.close()

            frame = cv2.imread(filePath)  
            frame = cv2.resize(frame, (1024, 540))  

            cv2.imshow('imageOne',frame)
            cv2.waitKey(1)

        except socket.error as e:
            print("[ERROR] ", e)  



def sent_pic128():
    for file in allFileList128:
        time.sleep(0.5)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('127.0.0.1', 8903)) # camera 128
            filePath = os.path.join(imgPath128,file)
            print(filePath)
            sock.send(filePath.encode('UTF-8'))
            sock.close()

            frame = cv2.imread(filePath)  
            frame = cv2.resize(frame, (1024, 540))  

            cv2.imshow('imageOne',frame)
            cv2.waitKey(1)

        except socket.error as e:
            print("[ERROR] ", e)  

def sent_pic130():  
    for file in allFileList130:
        time.sleep(0.58)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('127.0.0.1', 8904)) # camera 130
            filePath = os.path.join(imgPath130,file)
            print(filePath)
            sock.send(filePath.encode('UTF-8'))
            sock.close()

            frame = cv2.imread(filePath)  
            frame = cv2.resize(frame, (1024, 540)) 

            cv2.imshow('imageTwo',frame)
            cv2.waitKey(1)
        except socket.error as e:
            print("[ERROR] ", e)      



sent_thread124 = Process(target=sent_pic124)
sent_thread125 = Process(target=sent_pic125)
sent_thread128 = Process(target=sent_pic128)
sent_thread130 = Process(target=sent_pic130)

sent_thread124.start()
# sent_thread125.start()
# sent_thread128.start()
# sent_thread130.start()
sent_thread124.join()
# sent_thread125.join()
# sent_thread128.join()
# sent_thread130.join()
print("Done.")

# thread wait: 0.315 show / 0.5
# process wait: 0.285  show/ 0.26 show 