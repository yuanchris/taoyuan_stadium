import cv2
import numpy as np
from .color_histogram import histogram
from .merge import merge
from .constant import cfg
from . import location_tools

WEST            = 1
EAST            = 0
FILE_NAME_LEN   = 29
LEFT            = 0
RIGHT           = 1
TOP             = 2
BOTTOM          = 3


class system_status():
    def __init__(self):
        self.is_pause = True
        self.is_base = [True,False,False,False]
    
    def set_base(self, a , b ,c):
        self.is_base = [True,a,b,c]



class player_status():
    '''
    c_x, c_y, left, top, w, h, c, score

    nas_1 = west
    '''  
    def __init__(self):
        self.SS = system_status()
        self.LT = location_tools.location_tools( self , self.SS ,cfg)
        self.CH = histogram(self)
        self.MG = merge()
        self.reset()

    def remove_judge(self,roi):
        res = [roi[0]]
        for i in range(1, len(roi)):
            if roi[i][0] == '0' and roi[i][1] > cfg.score_limit:
                res.append(roi[i])
        return res# [name , [],[],[]]

    def in_range(self, center , _range , point):
        dis = self.LT.lsm(center, point)
        return dis < _range
    def the_edge_one(self, base,_list):
        if base == 1:
            tmp = [0,0]
            for i in _list:
                if i[1]>tmp[1]:
                    tmp = i
        else:
            tmp = [999,0]
            for i in _list:
                if i[0]<tmp[0]:
                    tmp = i
        return tmp
                
    def remove_couch(self , base, merge_poi):
        _range = 33
        # print(self.SS.is_base[base] )
        if self.SS.is_base[base]:
            center = cfg.field_strategy[base]
            # print(self.in_range(center, _range, self.players[str(base) + 'B']) , self.in_range(center, _range, self.players[str(base) + 'H']))
            if self.in_range(center, _range, self.players[str(base) + 'B']) and self.in_range(center, _range, self.players[str(base) + 'H']):
                in_list = []
                for i in merge_poi:
                    if self.in_range(center, _range , i):
                        in_list.append(i)
                # print(len(in_list))
                if len(in_list) == 3:
                    tmp = self.the_edge_one(base , in_list)
                    merge_poi.remove(tmp)

    def set_my_roi(self, roi):
        '''
        current_1 c, score, left, top, right, bottom
        current c_x,c_y,left,top,w,h,c, score
        need c, score, c_x, bottom, top, left, right -> str, float, int ,int,int,int,int
        '''
        # tmp = roi.split('\t')
        # res = [tmp[0]]
        res = ['']
        # for i in range(1, len(tmp)):
        for i in roi:
            info = i.split(' ')
            # c_x = int((int(info[4]) + int(info[2]))/2)
            # info_res = [info[0],float(info[1]),c_x,int(info[5]),int(info[3]),int(info[2]),int(info[4])]
            info_res = [info[6],float(info[7]), int(info[0]),int(info[3]) + int(info[5]), int(info[3]) ,int(info[2]), int(info[2]) + int(info[4]) ]
            res.append(info_res)
        return res

    def crop_img(self,roi, img):
        l = len(roi)
        res = []
        for i in range(1,l):
            info = roi[i]
            tmp = img[int(info[4]):int(info[3]), int(info[5]):int(info[6])]
            res.append(tmp)
        return res

    def init_position(self, west_img, east_img , west_roi, east_roi,west_name = '', east_name = '', flag = False):
        '''
        init & update pre state 
        '''
        # if len(self.old_players) > 1:
        #     self.pre_state_player = self.old_players[-2:]
        # else:
        #     self.pre_state_player = self.players

        west_roi = self.remove_judge(self.set_my_roi(west_roi))# [name , [],[],[]]
        east_roi = self.remove_judge(self.set_my_roi(east_roi))
        # print(east_roi)
        east_crop = [] #self.crop_img(east_roi, east_img)
        west_crop = [] #self.crop_img(west_roi, west_img)
        # east_img = cv2.imread('./test.jpg')
        if not flag:
            east_crop = self.crop_img(east_roi, east_img)
            west_crop = self.crop_img(west_roi, west_img)
        east_poi , west_poi = self.roi_to_poi(east_roi, west_roi)#[[],[],[]]
        west_hue = self.CH.get_score(west_crop , west_poi , cfg, west_name,flag)#[[],[],[]]
        east_hue = self.CH.get_score(east_crop , east_poi , cfg, east_name,flag)
        
        # print(len(west_crop),len(west_poi),len(east_poi),len(east_crop),len(west_crop),len(east_crop))
        # print(east_hue)
        # print(west_hue)
        # step 1 : find the LF and RF position which cant merge by two scene
        self.LT.set_location_list(east_poi)
        # print(len(east_hue))
        reduce_idx = self.LT.set_players('LF')
        if not reduce_idx == -1 : del east_hue[reduce_idx]
        self.LT.set_location_list(west_poi)
        reduce_idx = self.LT.set_players('RF')
        if not reduce_idx == -1 : del west_hue[reduce_idx]

        # step 2 : merge two scene by opposite point and calculate points score
        # west view is better than east ,so the view will base on west view
        merge_poi = self.MG.merge_with_hue(west_poi , east_poi , west_hue , east_hue)
        self.remove_couch(1, merge_poi)
        self.remove_couch(3, merge_poi)
        # print(merge_poi)
        # return merge_poi

        # find out P and CF
        self.LT.set_location_list(merge_poi)
        #if self.visible[0]:
        # print('p error',self.players['P'][0])
        if self.players['P'][0] > 3:
            self.LT.set_players('P')
        self.LT.set_players('CF')
        self.LT.who_at_home()
        
        H_p = self.players['H']
        if H_p[0] > cfg.H_left_limit and H_p[0] < cfg.H_right_limit and H_p[1] > 340:
            if H_p[0] > self.H_passby_dis:
                self.H_passby_dis = H_p[0]
            else:
                if self.H_passby_dis - H_p[0] > 3 :
                    print(east_name+'\tH returned')
                    self.SS.is_pause = True
        
        if self.SS.is_base[3] :
            if self.LT.lsm(self.players['3H'],self.players['3B']) <  self.LT.dis_initialized_spe\
                and self.players['3H'][3] and self.players['3B'][3]:
                self.LT.set_two_players('3H','3B')
            else:
                self.LT.set_players('3H')
                self.LT.set_players('3B')
            self.exchange_position_by_score('3B', '3H')
        else:
            self.LT.set_players('3B')
        if self.SS.is_base[1] :
            # print(self.players['1H'][3],self.players['1B'][3])
            if self.LT.lsm(self.players['1H'],self.players['1B']) <  self.LT.dis_initialized_spe\
                and self.players['1H'][3] and self.players['1B'][3]:
                self.LT.set_two_players('1H','1B')
            else:
                self.LT.set_players('1H')
                self.LT.set_players('1B')
            self.exchange_position_by_score('1B', '1H')
        else:
            self.LT.set_players('1B')
        # print(self.players['SS'])
        old_SS_player = self.players['SS'].copy()
        self.LT.set_players('SS')
        self.LT.set_players('2B')
        # print('be',self.players['SS'])
        # SS_flag = self.PS.players['SS']
        # _2H_flag = self.PS.players['2H']
        # self.exchange_position_by_position('SS' ,LEFT, '2B')
        # print(self.SS.is_base[2])
        if self.SS.is_base[2] :
            if self.LT.lsm(self.players['2H'],self.players['SS']) <  self.LT.dis_initialized_spe\
                and self.players['2H'][3] and self.players['SS'][3]:
                # print('set 2 player')
                self.LT.location_list.append(self.players['SS'].copy())
                self.players['SS'] = old_SS_player
                self.LT.set_two_players('2H','SS')
            else:
                # print('set 2H')
                self.LT.set_players('2H')
            self.exchange_position_by_score('SS', '2H')
        # print('af',self.players['SS'])
        # if not self.visible[0]:
        if not self.players['P'][0] > 3:
            self.LT.set_players('P')

        # if not len(self.old_players) == 0:
        self.LT.two_guys_are_close()

        # P >> C
        self.LT.check_PC()

        # check when HCJ is ture , the position of H & C is correct
        self.LT.check_HC_by_HCJ()

        # final check runner confusion with runner couch
        for i in merge_poi:
            keys = ['H' , '1H', '2H']
            for k in keys :
                if self.LT.lsm(self.players[k],i) < 14 :
                    # print(self.LT.lsm(i ,[10,310]) < 25 , self.LT.lsm(i , [105,390]) < 25)
                    if self.LT.lsm(i ,[10,310]) < 25:
                        # print('big',i[1] , self.players[k][1])
                        if i[1] > self.players[k][1]:
                            self.players[k][0] = i[0]
                            self.players[k][1] = i[1]
                    elif self.LT.lsm(i , [105,390]) < 25:
                        # print('small',i[1] , self.players[k][1])
                        if i[1] < self.players[k][1]:
                            self.players[k][0] = i[0]
                            self.players[k][1] = i[1]
        
        # remove who return home
        self.check_123H_return_home()
        # define the players is stuck or not
        if len(self.old_players) >= cfg.stuck_frames:
            self.people_block_each()
            self.old_players.pop(0)
            self.check_if_stuck(self.old_players[3], self.players)

        self.old_players.append(self.players.copy())
        # 
        self.merge_visible_and_ss()
        # print(self.players['H'])
        return merge_poi

    def people_block_each(self):
        range_ = ['1B','2B','3B','SS','H','1H','2H','3H']
        checked = []
        pl = len(range_)
        for i in range(pl):
            for j in range(i+1,pl):
                if not self.old_players[-1][range_[i]][3] or\
                    not self.old_players[-1][range_[j]][3] or \
                    range_[i] in checked or \
                    range_[j] in checked:
                    continue
                if self.LT.lsm(self.players[range_[i]],self.players[range_[j]]) <2:
                    if self.LT.lsm(self.players[range_[i]],self.old_players[-1][range_[i]]) > self.LT.dis_initialized +2:
                        self.players[range_[j]] = self.players[range_[i]].copy()
                        self.players[range_[i]] = self.old_players[-1][range_[i]].copy()
                        checked.append(range_[j])
                        checked.append(range_[i])
                    elif self.LT.lsm(self.players[range_[j]],self.old_players[-1][range_[j]]) > self.LT.dis_initialized+2:
                        self.players[range_[i]] = self.players[range_[j]].copy()
                        self.players[range_[j]] = self.old_players[-1][range_[j]].copy()
                        checked.append(range_[j])
                        checked.append(range_[i])
            

    def check_123H_return_home(self):
        for i in range(1,4):
            code = str(i) + 'H'
            v = self.players[code]
            # print(v)
            if (v[0] < 25 and v[1] > 375):
                self.SS.is_base[i] = False
        # print()

    def check_if_stuck(self, be,af):
        for b,a in zip(be.items() , af.items()):
            left, bott = 15, 385
            idx = cfg.always_in_ground.index(b[0])
            if b[1][0] < left+5 and b[1][1] > bott-5:
                left,bott = 5, 395
            if b[0] == '1H':
                # print(self.visible[idx])
                bott = 388
            range = ['1H','2H','3H']
            if b[0] in range and b[1][0]<25 and b[1][1]>=370 and b[1][:3] == a[1][:3]:
                self.visible[idx] = False
            else:
                self.visible[idx] = not ( b[1][:3] == a[1][:3] and (b[1][0] < left or b[1][1] > bott) )

                
    def merge_visible_and_ss(self):
        for i in range(1,4):
            code = str(i) + 'H'
            idx = cfg.always_in_ground.index(code)
            self.visible[idx] = self.visible[idx] and self.SS.is_base[i]

    def roi_to_poi(self,east_roi , west_roi):
        east_result = []
        for i in east_roi[1:]:
            roi = np.array([[i[2],i[3]]],dtype=np.float32 )
            result = cv2.perspectiveTransform(roi[None,:,:], cfg.east_perspective)
            poi = [int(result[0,0,0]),int(result[0,0,1])]
            # if (poi[0] > 50 and poi[1] > 387) or (poi[0] < 10  and  poi[1] < 350):
            #     continue
            east_result.append(poi)

        west_result = []
        for i in west_roi[1:]:
            roi = np.array([[i[2],i[3]]],dtype=np.float32 )
            result = cv2.perspectiveTransform(roi[None,:,:], cfg.west_perspective)
            poi = [int(result[0,0,0]),int(result[0,0,1])]
            # if (poi[0] > 50 and poi[1] > 387) or (poi[0] < 10  and  poi[1] < 350):
            #     continue
            west_result.append(poi)
        return east_result, west_result
    def reset(self):
        self.HCJ                = False
        # self.is_H_passby        = False
        self.H_passby_dis       = 15
        self.base_exchange_init = [False, False ,False]
        self.visible            = [ True for i in range(13)]
        self.old_players        = []
        self.players            = {}
        self.players['P']       = [60,340,1.0 , True] # x , y , score , init or not
        self.players['C']       = [9,391,0 , True]
        self.players['H']       = [15,385,0 , True]
        self.players['1B']      = [123,363,0 , False]
        self.players['2B']      = [150,310,0 , False]
        self.players['3B']      = [34,280,0 , False]
        self.players['SS']      = [90,250,0 , False]
        self.players['LF']      = [93,100,0 , False]
        self.players['CF']      = [220,145,0 , False]
        self.players['RF']      = [311,288,0 , False]
        self.players['1H']      = [114,372,0 , False]
        self.players['2H']      = [94,286,0 , False]
        self.players['3H']      = [12,305,0 , False]
        # self.players['LC']    = [142,285,0]
        # self.players['RC']    = [263,286,0]

    def exchange_position_by_score(self, b,h):
        if not self.players[h][2] == 0 and not self.players[b][2] == 0 and not self.base_exchange_init[int(h[0])-1]:
            self.base_exchange_init[int(h[0])-1] = True
            if self.players[h][2] > self.players[b][2] + cfg.score_diff and self.players[h][2] > 0.25:
                tmp = self.players[h].copy()
                self.players[h] = self.players[b].copy()
                self.players[b] = tmp
    def exchange_position_by_position(self, a,command ,b): # left if B > A -> exchange -> left rihgt top bottom
        flag = False
        if command == 0:
            flag = self.players[b][0] < self.players[a][0] 
        elif command == 1:
            flag = self.players[b][0] > self.players[a][0] 
        elif command == 2:
            flag = self.players[b][1] < self.players[a][1] 
        elif command == 3:
            flag = self.players[b][1] > self.players[a][1] 
        if flag:
            tmp = self.players[a].copy()
            self.players[a] = self.players[b].copy()
            self.players[b] = tmp

    def get_result_x_y_pause(self):
        result = {}
        c = 0
        for k,v in self.players.items():
            poi = np.array([[v[0],v[1]]],dtype=np.float32 )
            change = cv2.perspectiveTransform(poi[None,:,:], cfg.result_perspective)
            poi = [int(change[0,0,0]),int(change[0,0,1]),self.visible[c]]
            result[k] = poi
            c += 1
        '''
        the result form is {'P':[300,300,False], ...}
        ['P','C','H','1B','2B','3B','SS','LF','CF','RF','1H','2H','3H']
        '''
        return result

    def set_pause(self, HCJ, system_start, servitor_start):
        '''
        HCJ is bool which means hit and catcher and judge is ready or not
        system_start is bool from yolo
        servitor_start is someone operating the panel and needs [is_start_or_not, [1h,2h,3h are avaliable?]]
        '''
        tmp = servitor_start[1]
        self.SS.set_base(tmp[0],tmp[1],tmp[2])
        if not system_start or not servitor_start:
            self.reset()
            self.SS.is_pause = True
        if system_start and servitor_start:
            self.SS.is_pause = False
        for i in range(3):
            if not tmp[i] :
                self.players[str(i+1)+'H'][:2] = [-1,-1]
        self.HCJ = HCJ
        return not self.SS.is_pause

        
        

