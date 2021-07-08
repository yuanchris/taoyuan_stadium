# -*- encoding: utf-8 -*-
'''
@File    :   data_controller.py
@Time    :   2021/06/25 23:02:12
@Author  :   MattLiu <RayHliu> 
@Version :   ver.2021
'''

import numpy as np
import glob
import os 
from PIL import Image

class Data_Contrller:
    def __init__(self,replayId):
        self.yoloResult = []
        self.pklFileNames = []
        self.hightCount = None
        self.imageDirPath = None
        self.get_init_para(replayId)
        self.totalFiles = list(sorted(glob.glob(self.imageDirPath+'*.jpg')))
        if len(self.totalFiles) == 0:
            raise ValueError ('no found any files in {}'.format(self.imageDirPath))

        if len(self.pklFileNames) > 0:
            self.read_yolo_pkl()

    def get_init_para(self,replayId):
        if replayId == '1B'or replayId == '1B_v2' or replayId == '1B_v3':
            """ ==== 2019 Taichung Brothers ===="""
            # self.pklFileNames = ['./190929_NAS/yolov3_labels/NAS_02_20190929-19-41.pkl','./190929_NAS/yolov3_labels/NAS_02_20190929-19-42.pkl']
            # self.imageDirPath = './NAS_data_new/tmp/'
            # self.imageDirPath = './NAS_data_new/NAS02_20191020_19-01/'
            # self.imageDirPath = './NAS_data_new/NAS02_20191020_19/'
            # self.imageDirPath = './NAS_data_new/NAS02_20191103_20-25/'
            """ ==== 2020 Taoyuan Rakuten: 20200628_1Base:(8600) ==== """
            # self.hightCount = 8600
            # self.imageDirPath = './data/20200628_1Base/'

            """ ==== 2021 Taoyuan Rakuten ==== 
                Error: 19400,26400 runner detected, force error
                Correct:30640, 29600
            """

            self.hightCount = 26400                                                 
            self.imageDirPath = '/data/smart_stadium_2020/highlight_replay/data/20210417_1B_3B/10.2.8.124/*'
            
        elif replayId == '2B':
            self.hightCount = 20286 # 1B count:907 捕手漏接跑向一壘
            # self.pklFileNames = ['./190929_NAS/yolov3_labels/NAS_04_20190929-19-37.pkl']
            # self.imageDirPath = './190929_NAS/NAS04/20190929_19_41-42/'
            self.imageDirPath = './NAS_data/NAS04_20191014_20-12/'
            self.imageDirPath = './NAS_data_new/NAS04_20191030_10-00/'
            self.imageDirPath = './NAS_data_new/tmp/'
            
        elif replayId == '3B':
            self.hightCount = 44 # 1B count:907 捕手漏接跑向一壘
            # self.pklFileNames = ['./190929_NAS/yolov3_labels/NAS_12_20190929-19-41.pkl','./190929_NAS/yolov3_labels/NAS_12_20190929-19-42.pkl']
            # self.imageDirPath = './190929_NAS/NAS12/20190929_19_41-42/'
            self.imageDirPath = './NAS_data_new/test3-3/'
            self.imageDirPath = './NAS_data_new/tmp2/' 

        elif replayId == 'HB':
            self.hightCount = 0 # 1B count:907 捕手漏接跑向一壘
            # self.pklFileNames = ['./191001_NAS/NAS03_20191001_18-38.pkl','./191001_NAS/NAS03_20191001_18-39.pkl']
            # self.imageDirPath = './191001_NAS/NAS03_20191001_18_38-39/'
            self.imageDirPath = './NAS_data_new/NAS03_20191102_19-1/' 
            self.imageDirPath = './NAS_data_new/NAS03_20191103_19-08/'
            # self.imageDirPath = './NAS_data_new/NAS03_20191103_19-58/'
            self.imageDirPath = './NAS_data_new/tmp2/'

            """ ==== 20210416-18 Taoyuan Rakuten ===== """
            self.hightCount = 22630      
            self.imageDirPath = './data/20210417_1B_3B/10.2.8.125/*'

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

    def export_image_with_bbox(self,count,yoloResult):
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

