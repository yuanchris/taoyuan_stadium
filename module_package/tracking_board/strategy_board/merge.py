import numpy as np
import cv2
import math

class merge():

    def __init__(self):
        self.nearest_threshold = 15

    def judge_team(self,a,b):
        return (a+b) / 2

    def nearby_with_hue(self,goal,compare_list , goal_hue , hue_list):
        nearest = 100
        info = 0
        for count, i in enumerate(compare_list):
            x = i[0] - goal[0]
            y = i[1] - goal[1]
            lsm = math.sqrt( x*x + y*y )
            if lsm < nearest:
                nearest = lsm
                info = count
        # print(nearest)
        if nearest < self.nearest_threshold :
            # print(hue_list)
            # print(info)
            team = self.judge_team(goal_hue , hue_list[info])
            del compare_list[info]
            del hue_list[info]
            return True,team
        return False , -1

    def merge_with_hue(self,east_poi,west_poi,east_hue , west_hue):
        result_list = []
        for count, i in enumerate (east_poi):
            flag , team = self.nearby_with_hue(i, west_poi, east_hue[count] , west_hue)
            if flag :
                i.append(team)
                result_list.append(i)
        return result_list