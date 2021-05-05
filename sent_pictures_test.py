import sys,os
import time
import logging
import socket, json

# 指定要查詢的路徑
yourPath = '/home/chris/Desktop/taoyuan/NAS02_20191020_19-01'
# 列出指定路徑底下所有檔案(包含資料夾)
allFileList = os.listdir(yourPath)
allFileList.sort()
# print(allFileList)
for file in allFileList:
    time.sleep(0.2)
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('127.0.0.1', 8902)) # camera 3
        filePath = os.path.join(yourPath,file)
        print(filePath)
        sock.send(filePath.encode('UTF-8'))
        sock.close()
        

    except socket.error as e:
        print("[ERROR] ", e)      