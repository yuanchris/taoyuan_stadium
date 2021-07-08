# -*- encoding: utf-8 -*-
'''
@File    :   replay_parameters.py
@Time    :   2021/06/25 23:03:32
@Author  :   MattLiu <RayHliu> 
@Version :   Taoyuan_20210417
'''

smaller = lambda x,y: x<=y
bigger = lambda x,y: x>=y

crop_size = [830,530]

""" parameters
    base params controller
        - replay_frame_crop_region:     <list> replay Crop frame range. ex:[minx, miny, maxx, maxy]
        - detect_c12_bottom_left:       <tuple>  coordinate of region bottom left.  bottom left of c12 region
        - detect_c12_bottom_right:      <tuple>  coordinate of region bottom right. bottom right of c12 region
        - detect_c12_top_left:          <tuple>  coordinate of region top left.     top left of c12 region
        - detect_c12_top_right:         <tuple>  coordinate of region top right.    top right of c12 region 
        - base_region:                  <list>
        - force_region:                 <list>

    detection and tracking params info
        - bbox_area_threshold:          Ignore the smaller bboxes that area smaller the value 
        - init_tracking_distance:       If the value of distance from init points to the bbox smaller than the value, the bbox is tracked.
        - min_run_distance:             If the value of distance from the bbox to last tracking bbox bigger than the value, the bbox is recognized as running.
        - max_run_distance:             The value is similiar to init_tracking_distance. When the bbox is initily tracked, tracking_distance is replaced to max_run_distance.
        - force_tracking_over_info:     [tracking_line_x/y, x/y_index, bigger/small_than] Set the end of tracking line. If the runner over tracking line,and there are not any force players in force ready region, the force out will not work.
"""

parameters = {
    '1B':{
        'condiction1_count':4,
        'condiction2_count':3,
        'min_run_distance':80, # min of tracking range
        'max_run_distance':550, # max of tracking range
        'init_tracking_distance':300,
        'bbox_area_threshold':(1250-1056)*(1862-1457)/3,
        'force_tracking_over_info':[2548,0,smaller],
        'replay_frame_crop_region':[1900,1170,3100,1770], 
        'detect_c12_bottom_left':(1979,1402), 
        'detect_c12_bottom_right':(3530,1867), 
        'detect_c12_top_left':(2239,1232), 
        'detect_c12_top_right':(3530,1510), 
        'base_region':[[1800,1270],[1726,1286],[1784,1312],[1860,1297]],
        'force_region':[[1375,1172],[1185,1373],[2068,1609],[2181,1396]]
    },

    '1B_v2':{
        'condiction1_count':4,
        'condiction2_count':3,
        'min_run_distance':80,
        'max_run_distance':550,
        'init_tracking_distance':300,
        'bbox_area_threshold':(1250-1056)*(1862-1457)/3,
        'force_tracking_over_info':[2548-crop_size[1],0,smaller],
        'replay_frame_crop_region':[1900-crop_size[0],1170-crop_size[1],3100-crop_size[0],1770-crop_size[1]], 
        'detect_c12_bottom_left':(1979-crop_size[0],1402-crop_size[1]), 
        'detect_c12_bottom_right':(3530-crop_size[0],1867-crop_size[1]), 
        'detect_c12_top_left':(2239-crop_size[0],1232-crop_size[1]), 
        'detect_c12_top_right':(3530-crop_size[0],1510-crop_size[1]), 
        'base_region':[[1800-crop_size[0],1270-crop_size[1]],[1726-crop_size[0],1286-crop_size[1]],[1784-crop_size[0],1312-crop_size[1]],[1860-crop_size[0],1297-crop_size[1]]],
        'force_region':[[1375-crop_size[0],1172-crop_size[1]],[1185-crop_size[0],1373-crop_size[1]],[2068-crop_size[0],1609-crop_size[1]],[2181-crop_size[0],1396-crop_size[1]]]
    },

    '1B_v3':{
        'condiction1_count':4,
        'condiction2_count':3,
        'min_run_distance':80,
        'max_run_distance':550,
        'init_tracking_distance':200,
        'bbox_area_threshold':(1250-1056)*(1862-1457)/3,
        'force_tracking_over_info':[2548-crop_size[1],0,smaller],
        'replay_frame_crop_region':[1900,1170-crop_size[1],3100,1770-crop_size[1]], 
        'detect_c12_bottom_left':(1979,1402-crop_size[1]), 
        'detect_c12_bottom_right':(3257,1181), 
        'detect_c12_top_left':(2239,1232-crop_size[1]), 
        'detect_c12_top_right':(3257,954), 
        'base_region':[[1800,1270-crop_size[1]],[1726,1286-crop_size[1]],[1784,1312-crop_size[1]],[1860,1297-crop_size[1]]],
        'force_region':[[1375,1172-crop_size[1]],[1185,1373-crop_size[1]],[2068,1609-crop_size[1]],[2181,1396-crop_size[1]]]
    },

    '2B':{
        'condiction1_count':4,
        'condiction2_count':3,
        'min_run_distance':80,
        'max_run_distance':550,
        'tracking_distance':300,
        'bbox_area_threshold':(1733-1503)*(839-468)/3,
        'force_tracking_over_info':[2389,0,smaller],
        'replay_frame_crop_region':[1900,1170,3100,1770], 
        'detect_c12_bottom_left':(1607,969), 
        'detect_c12_bottom_right':(3813,1574), 
        'detect_c12_top_left':(2066,718), 
        'detect_c12_top_right':(3809,1167), 
        'base_region':[[1467,827],[1351,854],[1448,892],[1570,873]],
        'force_region':[[1138,767],[993,897],[1747,1152],[1924,982]]
    },

    '3B':{
        'condiction1_count':4,
        'condiction2_count':3,
        'min_run_distance':80,
        'max_run_distance':550,
        'tracking_distance':300,
        'bbox_area_threshold':(1733-1503)*(839-468)/3,
        'force_tracking_over_info':[2389,0,smaller],
        'replay_frame_crop_region':[1900,1170,3100,1770], 
        'detect_c12_bottom_left':(1607,969), 
        'detect_c12_bottom_right':(3813,1574), 
        'detect_c12_top_left':(2066,718), 
        'detect_c12_top_right':(3809,1167), 
        'base_region':[[1467,827],[1351,854],[1448,892],[1570,873]],
        'force_region':[[1138,767],[993,897],[1747,1152],[1924,982]]
    },

    'HB':{
        'condiction1_count':4,
        'condiction2_count':3,
        'min_run_distance':80,
        'max_run_distance':550,
        'init_tracking_distance':300,
        'bbox_area_threshold':(1250-1056)*(1862-1457)/3,
        'force_tracking_over_info':[1760,0,bigger],
        'replay_frame_crop_region':[1900,1170,3100,1770], 
        'detect_c12_bottom_left':(351,1563), 
        'detect_c12_bottom_right':(2136,1433), 
        'detect_c12_top_left':(253,1144), 
        'detect_c12_top_right':(2342,1090), 
        'base_region':[[2886,1091],[2861,1116],[2950,1141],[2966,1100]],
        'force_region':[[2791,994],[2651,1144],[3022,1208],[3167,1063]]
    }
}
