import cv2
import numpy as np
import time
import pickle, os

class histogram():
    def __init__(self,PS):
        self.PS = PS
    baezu = 1
    count = 0 
    def get_hueHist(self,image,mask):
        hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hueHist = cv2.calcHist([hsvImage], [0], mask.astype(np.uint8), [180], [0, 180])
        hueHist = cv2.normalize(hueHist, None, norm_type=cv2.NORM_MINMAX)
        return hueHist

    def compare(self, origin_hist, goal_hist):
        return cv2.compareHist(origin_hist, goal_hist, cv2.HISTCMP_CORREL)
    def get_hue_list(self,crops):
        epic = []
        for i in crops :
            h,w = i.shape[:2]
            i = i[:int(h/2)][:]
            # cv2.imwrite('./test_res/Q'+str(h)+str(w)+'.bmp',i)
            mask = self.get_mask(i)
            # cv2.imwrite('./test_res/Q'+str(h)+str(w)+'_mask.bmp',mask)
            epic.append(self.get_hueHist(i , mask))
        return epic
    def near_no_H(self, code, cfg):
        pi = self.PS.players[code]
        nearH = [self.lsm(pi,self.PS.players['H']),\
                self.lsm(pi,self.PS.players['1H']),\
                self.lsm(pi,self.PS.players['2H']),\
                self.lsm(pi,self.PS.players['3H'])]
        return min(nearH) > cfg.no_H_in_D_range

    def who_is_the_one(self, pois,cfg):#[[],[],[]]
        distance = []
        # pi = self.PS.players['p'][:2] if self.PS.players['P'][3] else cfg.mound
        # print(pi)
        pi = cfg.mound
        # if self.PS.players['P'][0] < 5 or self.PS.players['P'][0] > 91:
        if self.PS.players['P'][2] < 1 and\
            (self.PS.players['P'][0] < 10 or self.PS.players['P'][0] > 91 \
                or self.PS.players['P'][1] > 389):
            if self.near_no_H('SS',cfg):
                pi = self.PS.players['SS']
            elif self.near_no_H('2B',cfg):
                pi = self.PS.players['2B']
                
        for poi_list in pois:
            # x = int(roi_list[2]) - cfg.mound_ew[is_east][0]
            x = int(poi_list[0]) - pi[0]
            y = int(poi_list[1]) - pi[1]
            distance.append( x*x + y*y )
        return distance.index( min(distance) )

    def get_mask(self, img):
        h , w,c = img.shape
        data = img.reshape((-1,3))
        data = np.float32(data)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) 
        ret,label,center=cv2.kmeans(data, 5, None, criteria, 5 , cv2.KMEANS_RANDOM_CENTERS)

        index = label[0][0]
        center = np.uint8(center)
        color = center[0]
        mask = np.ones((h, w), dtype=np.uint8)*255.
        label = np.reshape(label, (h, w))
        mask[label == index] = 0
        # mask = np.reshape(mask,  (h,w,1))
        # print(mask)
        return mask

    def get_score(self, crops, poi, cfg, name ,flag):
        more = ''
        if flag and os.path.exists('crop_img_{}{}/{}.pk'.format(name[4:11],more,name)):
            f = open('crop_img_{}{}/{}.pk'.format(name[4:11],more,name), 'rb')
            imgs = pickle.load(f)
        else:
            imgs = self.get_hue_list(crops)
        the_one  = self.who_is_the_one(poi, cfg)# 0 for is , 1 for not
        score_list = []
        for i in imgs:
            score_list.append( self.compare(imgs[the_one] , i ) )
        return score_list

    def lsm(self, p1 , p2):
        a = p1[0] - p2[0]
        b = p1[1] - p2[1]
        return np.sqrt(a*a + b*b)
        #  [1.0, 0.8244857166099974, 0.4865887555708959, 0.6121779048653377, 0.44470506328584697, 0.8372317721146937, -0.06200860043345016, -0.13675182997329896, 0.7891419109198712, 0.8380294475067013, 0.7921500453242172, 0.3570844351075479, 0.5439017082798175]