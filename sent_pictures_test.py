import sys,os
import time
import logging
import socket, json
import threading
from cv2 import cv2
from multiprocessing import Process
# 指定要查詢的路徑
yourPath = '/home/chris/Desktop/taoyuan/taoyuan_img/test_210417_19-02-24_19-03-15_128_west'
yourPath2 = '/home/chris/Desktop/taoyuan/taoyuan_img/test_210417_19-02-24_19-03-15_130_east'

# 列出指定路徑底下所有檔案(包含資料夾)
# allFileList = os.listdir(yourPath)
# allFileList.sort()
# allFileList2 = os.listdir(yourPath2)
# allFileList2.sort()
path1 = '/home/chris/Downloads/pause_start_v1/pause_start_v1/pause_start_info/test_210417_19-02-24_19-03-15_128_west_info.txt'
path2 = '/home/chris/Downloads/pause_start_v1/pause_start_v1/pause_start_info/test_210417_19-02-24_19-03-15_130_east_info.txt'

allFileList = []
allFileList2 = []
with open(path1) as f:
    for line in f.readlines():
        s = line.split(',')
        allFileList.append(s[0])

with open(path2) as f:
    for line in f.readlines():
        s = line.split(',')
        allFileList2.append(s[0])

def sent_pic():
    for file in allFileList:
        time.sleep(0.285)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('127.0.0.1', 8902)) # camera 2
            filePath = os.path.join(yourPath,file)
            print(filePath)
            sock.send(filePath.encode('UTF-8'))
            sock.close()

            frame = cv2.imread(filePath)  
            frame = cv2.resize(frame, (1024, 540))  

            cv2.imshow('imageOne',frame)
            cv2.waitKey(1)

        except socket.error as e:
            print("[ERROR] ", e)  

def sent_pic2():  
    for file in allFileList2:
        time.sleep(0.26)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('127.0.0.1', 8903)) # camera 3
            filePath = os.path.join(yourPath2,file)
            print(filePath)
            sock.send(filePath.encode('UTF-8'))
            sock.close()

            frame = cv2.imread(filePath)  
            frame = cv2.resize(frame, (1024, 540)) 

            cv2.imshow('imageTwo',frame)
            cv2.waitKey(1)
        except socket.error as e:
            print("[ERROR] ", e)      




sent_thread = Process(target=sent_pic)
sent_thread2 = Process(target=sent_pic2)

sent_thread.start()
sent_thread2.start()
sent_thread.join()
sent_thread2.join()
print("Done.")
# thread wait: 0.315 / 0.5
# process wait: 0.285 / 0.26