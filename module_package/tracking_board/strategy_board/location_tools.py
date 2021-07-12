import numpy as np
from .import status
from .constant import cfg

class location_tools():
    def __init__(self,PS,SS ,PG):
        self.PS                     = PS
        self.SS                     = SS
        self.PG                     = PG
        self.dis_initialized        = 12
        self.dis_initialized_spe    = 16
        self.dis_uninitialized      = 50
        self.dis_uninitialized_CF   = 80
    def check_PC(self):
        if self.PS.players['P'][0] > 5 and self.PS.players['P'][0] < 394\
             and self.PS.players['P'][0] - self.PS.players['P'][1]\
             < self.PS.players['C'][0] - self.PS.players['C'][1]\
                and self.PS.players['C'][0] - self.PS.players['C'][1] > -355 :
            # print('exchange')
            self.exchange_position('P','C')
       
    def check_HC_by_HCJ(self):
        if self.PS.HCJ and self.PS.players['H'][0]-self.PS.players['H'][1] < \
            self.PS.players['C'][0]-self.PS.players['C'][1]:
            self.exchange_position('H','C')

    def set_location_list(self,location_list):
        self.location_list       = location_list

    def is_between(self, arr , goal):
        return arr[0] <= goal and arr[1] >= goal

    def check_who_is_more_close(self, goal , pt1 , pt2 ):
        pt1_dis = self.lsm(goal, pt1)
        pt2_dis = self.lsm(goal, pt2)
        return pt1_dis < pt2_dis
    def exchange_position(self, k,k2):
        tmp = self.PS.players[k].copy()
        self.PS.players[k] = self.PS.players[k2].copy()
        self.PS.players[k2] = tmp
    def two_guys_are_close(self):
        team_list = []
        for  k,v  in self.PS.players.items():
            team_list.append([k,v])
        for i in range(len(team_list)-1):
            if team_list[i][0] == 'C':
                continue
            for j in range(i+1 , len(team_list)):
                if team_list[j][0] == 'C':
                    continue
                k = team_list[i][0] 
                k2 = team_list[j][0]
                v = team_list[i][1] 
                v2 = team_list[j][1]
                if not k == k2 and self.lsm(v,v2) < 17:#12
                    check = 'H' in k 
                    check2 = 'H' in k2
                    if check and check2:
                        if not v[:2] == [-1,-1] and not v2[:2] == [-1,-1]:
                            self.distribute_by_position(k,k2,v,v2,\
                                self.PS.old_players[-1][k],self.PS.old_players[-1][k2])
                    elif check or check2:# not same team collipse
                            if check and v[2] > v2[2] + cfg.score_diff  and v[2] > 0.25:
                                self.exchange_position(k , k2)
                            elif check2 and v[2] + cfg.score_diff < v2[2] and v2[2] > 0.25:
                                self.exchange_position(k , k2)
                    else:
                        old_status = status.player_status().players[k]
                        if not self.check_who_is_more_close(old_status, v, v2):
                            self.exchange_position(k , k2)
    def get_dis_threshold( self, code):
        if self.PS.players[code][3]:
            threshold = self.dis_initialized
        elif code == 'CF':
            threshold = self.dis_uninitialized_CF
        else :
            threshold = self.dis_uninitialized
        return threshold

    def set_player_info(self, code, location ,flag):
        if len(location) < 3 : 
            location.append(0)
            location.append(flag)
        elif len(location) < 4:
            location.append(flag)
        else:
            location[3] = flag
        self.PS.players[code] = location

    def set_players (self , code ):
        threshold = self.get_dis_threshold(code)

        location, flag, reduce_idx, _ = self.nearest(code,threshold)
        if not reduce_idx == -1:
            del self.location_list[reduce_idx]
        self.set_player_info(code, location, flag)
        return reduce_idx

    def players_supplement(self, code, threshold = 0):
        threshold = self.get_dis_threshold(code)
        # print(threshold)
        location, flag, reduce_idx, _ = self.nearest(code,threshold)
        if not reduce_idx == -1:
            del self.location_list[reduce_idx]
        self.set_player_info(code, location, True)
        return reduce_idx

    def set_two_players (self , code_1, code_2):
        threshold_1 = self.get_dis_threshold(code_1)
        threshold_2 = self.get_dis_threshold(code_2)
        # threshold = max(threshold_1, threshold_2,m_threshold)
        # print(threshold_1, threshold_2, threshold)
        location_1, flag_1, reduce_idx_1,reduce_min_1 = self.nearest(code_1,threshold_1)
        location_2, flag_2, reduce_idx_2,reduce_min_2 = self.nearest(code_2,threshold_2)
        # print(location_1, flag_1, reduce_idx_1,reduce_min_1)
        # print(location_2, flag_2, reduce_idx_2,reduce_min_2)
        if not reduce_idx_1 == -1 and reduce_idx_1 == reduce_idx_2:
            if reduce_min_1 < reduce_min_2:
                self.set_player_info(code_1, location_1, flag_1)
                rest_code = code_2
            else:
                self.set_player_info(code_2, location_2, flag_2)
                rest_code = code_1
            del self.location_list[reduce_idx_1]
            # print(rest_code)
            self.players_supplement(rest_code, 16)
        else:
            # print('hello')
            self.set_player_info(code_1, location_1, flag_1)
            self.set_player_info(code_2, location_2, flag_2)
            if not reduce_idx_1 == -1 and not reduce_idx_2 == -1:
                del self.location_list[max(reduce_idx_1,reduce_idx_2)]
                del self.location_list[min(reduce_idx_1,reduce_idx_2)]
            elif not reduce_idx_1 == -1:
                del self.location_list[reduce_idx_1]
            elif not reduce_idx_2 == -1:
                del self.location_list[reduce_idx_2]
    def find_diff(self,code):
        if not self.PS.old_players[-1][code][:3] == self.PS.old_players[-2][code][:3]:
            return self.PS.old_players[-1][code][0] - self.PS.old_players[-2][code][0],\
                self.PS.old_players[-1][code][1] - self.PS.old_players[-2][code][1]
        count = 0
        diff = 1
        for i in range(len(self.PS.old_players)-2,1,-1):
            # print(i)
            if count > 6:break
            if self.PS.old_players[i][code][:3] == self.PS.old_players[i-1][code][:3]:
                diff += 1
            else:
                # print(count)
                x = self.PS.old_players[i][code][0] - self.PS.old_players[i-2][code][0]
                y = self.PS.old_players[i][code][1] - self.PS.old_players[i-2][code][1]
                count_sup = (count + 2) /10 + 1
                return diff * np.sign(x) * count_sup, diff * np.sign(y) * count_sup
            count+=1
        return 0,0

    def nearest(self,code , threshold):
        location = self.PS.players[code]
        if len(self.location_list) == 0:
            return location, location[3],-1, -1
        distance_list = []
        special_code_range = ['3B','1B','SS','2H','H','1H','3H']
        if code in special_code_range and threshold == self.dis_initialized:
            threshold = self.dis_initialized_spe
        # print(threshold)
        location_x = float(location[0])
        location_y = float(location[1])
        if 'H' in code and len(self.PS.old_players) > 1:
            _max = 30000
            diff_x, diff_y = self.find_diff(code)
            # print(code,diff_x,diff_y)
            location_x = location_x + np.sign(diff_x) * (_max if abs(diff_x) > _max else abs(diff_x))
            location_y = location_y + np.sign(diff_y) * (_max if abs(diff_y) > _max else abs(diff_y))
        for i in self.location_list:
            x = abs( float(i[0]) - location_x)
            y = abs( float(i[1]) - location_y )
            distance_list.append( np.sqrt( x*x + y*y) )
        while True:
            my_min = min(distance_list)
            if my_min > threshold :
                return location,location[3], -1, -1
            tmp = self.location_list[ distance_list.index( my_min ) ]
            code_range = ['1B','2B','3B','SS','CF','RF','LF']
            if tmp[1] > 400 or tmp[1] < 0 or tmp[0] > 400 or tmp[0] < 0:
                distance_list[ distance_list.index( my_min )] = 9999
                continue
            if code in code_range:
                if tmp[0] < 12 or tmp[1] > 388:
                    distance_list[ distance_list.index( my_min )] = 9999
                    continue
            if 'H' in code:
                if tmp[0] > 23 and tmp[0] < 104 and tmp[1]>297 and tmp[1] < 378:
                    distance_list[ distance_list.index( my_min )] = 9999
                    continue
            if code == '1H' :
                if tmp[1] > 387 or tmp[1] < 400 -cfg.H_border or tmp[0] > cfg.H_border:
                    distance_list[ distance_list.index( my_min )] = 9999
                    continue
            if code == '2H' :
                if tmp[1] < 400 -cfg.H_border or tmp[0] > cfg.H_border:
                    distance_list[ distance_list.index( my_min )] = 9999
                    continue
            if code == '3H' :
                if tmp[1] < 285 or tmp[1] < 400 -cfg.H_border or tmp[0] > cfg.H_border:
                    distance_list[ distance_list.index( my_min )] = 9999
                    continue
            break
        return tmp , True , distance_list.index( my_min ) ,my_min
    
    def distribute_by_position(self, code1,code2,p1,p2, old1,old2):
        s1_c = self.lsm(old1,p1)
        s1_h = self.lsm(old2,p1)
        s2_c = self.lsm(old1,p2)
        s2_h = self.lsm(old2,p2)
        if s1_c < s2_c and s1_h > s2_h: 
            self.PS.players[code2] = p2
            self.PS.players[code1] = p1
        elif s1_c > s2_c and s1_h < s2_h: 
            self.PS.players[code2] = p1
            self.PS.players[code1] = p2
        elif s1_c > s2_c and s1_h > s2_h:
            if s1_c > s1_h: # h for s1
                self.PS.players[code2] = p1
                self.PS.players[code1] = p2
            else:
                self.PS.players[code2] = p2
                self.PS.players[code1] = p1
        else:
            if s2_c < s2_h: # h for s1
                self.PS.players[code2] = p1
                self.PS.players[code1] = p2
            else:
                self.PS.players[code2] = p2
                self.PS.players[code1] = p1

    def who_at_home(self):
        # home = PG.field_strategy[0]
        home = [12,388]

        c_home = self.lsm(home , self.PS.players['C'])
        h_home = self.lsm(home , self.PS.players['H'])
        # print(self.PS.players['H'])
        # if c_home < 10  and h_home < 10:
        if self.PS.players['C'][0] < 20 and self.PS.players['C'][1] > 380  and self.PS.players['H'][0] < 22 and self.PS.players['H'][1] > 378:
            # print(self.PS.players['C'])
            # print(self.PS.players['H'])
            players = []
            # middle = []
            remove_list = []
            # middle_list = []
            for count ,i in enumerate(self.location_list ):
                dis = self.lsm(i , home)
                if dis < 17:
                    players.append(count)
                    remove_list.append(i)
                # elif dis < 25:
                #     middle.append(count)
                #     middle_list.append(i)
            # print(len(players))
            if len(players) >= 2:#只有一個人就看P or F比較近
                s1 = self.smallest_in_list(players)
                del players[s1[1]]
                s2 = self.smallest_in_list(players)
                # print(s2)
                if ((self.location_list[s1[0]][2] - self.location_list[s2[0]][2] > cfg.score_diff+0.07\
                         and self.location_list[s1[0]][2] > 0.2)):
                    self.PS.players['H'] = self.location_list[s2[0]]
                    self.PS.players['C'] = self.location_list[s1[0]]
                elif((self.location_list[s2[0]][2] - self.location_list[s1[0]][2] > cfg.score_diff+0.07\
                         and self.location_list[s2[0]][2] > 0.2)):
                    self.PS.players['H'] = self.location_list[s1[0]]
                    self.PS.players['C'] = self.location_list[s2[0]]
                else:
                    self.distribute_by_position('C','H',self.location_list[s1[0]],\
                        self.location_list[s2[0]],self.PS.players['C'],self.PS.players['H'])

                self.PS.players['H'].append(False)
                self.PS.players['C'].append(False)
                for i in remove_list:
                    self.location_list.remove(i)

            elif len(players) == 1:
                c_lsm = self.lsm(self.PS.players['C'] , self.location_list[players[0]])
                h_lsm = self.lsm(self.PS.players['H'] , self.location_list[players[0]])
                if c_lsm <= h_lsm:
                    self.PS.players['C'] = self.location_list[players[0]]
                    self.PS.players['C'].append(False)
                    for i in remove_list:
                        self.location_list.remove(i)
                    self.players_supplement('H')
                else : 
                    self.PS.players['H'] = self.location_list[players[0]]
                    self.PS.players['H'].append(False)
                    for i in remove_list:
                        self.location_list.remove(i)
                    self.players_supplement('C')
                # del self.location_list[players[0]]
                # print('=1')
        else:
            self.set_two_players('C','H')
            if self.lsm(self.PS.players['C'][:2],self.PS.players['H'][:2])< 17:
                if self.PS.players['H'][2] > 0.2 and self.PS.players['H'][2] - \
                    self.PS.players['C'][2] > cfg.score_diff:
                    self.exchange_position('H','C')

            # print('out')
            # print('hello')
            # self.set_players('H')   
            # self.set_players('C')
            
            
    def lsm(self, p1 , p2):
        a = p1[0] - p2[0]
        b = p1[1] - p2[1]
        return np.sqrt(a*a + b*b)


    
    def smallest_in_list( self , l ):
        _min   = 9999
        _count = 0
        for count , i in enumerate ( l ):
            tmp_min = (400 - self.location_list[i][0]) + self.location_list[i][1]
            if tmp_min < _min:
                _min = tmp_min
                _count = [i , count]
        return _count


