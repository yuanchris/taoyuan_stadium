import os
import numpy as np
from collections import defaultdict
from timeit import default_timer as timer

INFO_DIR = 'pause_start_info/'
WEST_INFO_FILE = INFO_DIR + 'test_210417_19-02-24_19-03-15_128_west_info.txt'
EAST_INFO_FILE = INFO_DIR + 'test_210417_19-02-24_19-03-15_130_east_info.txt'
# WEST_INFO_FILE = INFO_DIR + 'test_210417_19-01-51_19-02-11_128_west_info.txt'
# EAST_INFO_FILE = INFO_DIR + 'test_210417_19-01-51_19-02-11_130_east_info.txt'
# HB_INFO_FILE = INFO_DIR + 'test_210417_125_hb_info.txt'


SYSTEM_STATE = False
NOW_IMGID = defaultdict(int)
TRACK_DUGOUT = defaultdict(list)


def read_lines(file_name):
    """
    Read lines from the file.
    :param file_name: str, 'file path'
    :return list of strings
    """
    file_name = os.path.expanduser(file_name)
    with open(file_name, encoding='utf-8') as f:
        lines = f.readlines()
        return list(map(str.strip, lines))


def read_detect_txt(file_name):
    """
    Read detect information from txt file and convert to the indeed format.
    :param file_name: str, 'file path'
    :return img_detects: list of lists, [['img_name', 'cx cy lx ty w h class score', ...], ...]
    """
    infos = read_lines(file_name)
    img_detects = [info.split(',') for info in infos]
    return img_detects


def get_detect_info(info):
    """
    Get detect information and convert to the indeed format.
    :param info: str, 'cx cy lx ty w h class score'
    :return new_item: list of objects, [bcx, bcy, w, h, class, score]
    """
    _item = info.split()
    del _item[2]
    del _item[2]
    new_item = list(map(int, _item[:-1]))
    new_item.append(float(_item[-1]))
    new_item[1] = new_item[1] + new_item[3] // 2
    return new_item


def phcu_count(infos, p_lxtywh, hcu_lxtywh, phcu_bcxh):
    """
    Check the number of pitcher, hitter, catcher, and umpire in specific range to start.
    :param infos: list of strings, ['cx cy lx ty w h class score', ...]
    :param p_lxtywh: tuple of ints, (lx, ty, w, h) of pitcher range
    :param hcu_lxtywh: tuple of ints, (lx, ty, w, h) of hitter, catcher, and umpire range
    :param phcu_bcxh: dict, variable object, {'p': [[bc_x, h], ...], ...}
    :return None
    """
    p_lx, p_ty, p_w, p_h = p_lxtywh
    hcu_lx, hcu_ty, hcu_w, hcu_h = hcu_lxtywh
    for info in infos:
        bcx, bcy, w, h, c, score = get_detect_info(info)
        p_x_cond = p_lx < bcx < (p_lx + p_w)
        p_y_cond = p_ty < bcy < (p_ty + p_h)
        hcu_x_cond = hcu_lx < bcx < (hcu_lx + hcu_w)
        hcu_y_cond = hcu_ty < bcy < (hcu_ty + hcu_h)
        if c == 0 and p_x_cond and p_y_cond:
            print('- Pitcher is detected.')
            phcu_bcxh['p'].append([bcx, h])
        elif c == 0 and hcu_x_cond and hcu_y_cond:
            print('- Hitter or Catcher is detected.')
            phcu_bcxh['hc'].append([bcx, h])
        elif c == 1 and hcu_x_cond and hcu_y_cond:
            print('- Umpire is detected.')
            phcu_bcxh['u'].append([bcx, h])


def start(camera_detect, use_hb, west_p_lxtywh=(2330, 1010, 150, 70), east_p_lxtywh=(1375, 1030, 135, 65),
          west_hcu_lxtywh=(3390, 1360, 410, 155), east_hcu_lxtywh=(225, 1360, 340, 150), hb_hcu_lxtywh=(2675, 1010, 880, 250)):
    """
    Determine start by west and east or hb camera.
    :param use_hb: bool, use hb camera or not.
    :param camera_detect: dict, {'camera_ip': ['img_name', 'cx cy lx ty w h class score', ...], ...}
    :param west_p_lxtywh: tuple of ints, (lx, ty, w, h) of pitcher range in west camera
    :param east_p_lxtywh: tuple of ints, (lx, ty, w, h) of pitcher range in east camera
    :param west_hcu_lxtywh: tuple of ints, (lx, ty, w, h) of hitter, catcher, and umpire range in west camera
    :param east_hcu_lxtywh: tuple of ints, (lx, ty, w, h) of hitter, catcher, and umpire range in east camera
    :param hb_hcu_lxtywh: tuple of ints, (lx, ty, w, h) of hitter, catcher, and umpire range in hb camera
    :return start_state: bool, start or not
    """
    start_state = False
    west_phcu_bcxh = defaultdict(list)
    east_phcu_bcxh = defaultdict(list)
    west_p_bcxh = defaultdict(list)
    east_p_bcxh = defaultdict(list)
    hb_hcu_bcxh = defaultdict(list)

    # ---Count statistic.--- #
    pass_lxtywh = (0, 0, 0, 0)
    for camera, detect in camera_detect.items():
        if camera == '124': continue
        if not use_hb and camera == '125': continue
        img_name, infos = detect[0], detect[1:]
        print('\n[{} by camera {}] :'.format(img_name, camera))
        if use_hb and camera == '125':
            phcu_count(infos, pass_lxtywh, hb_hcu_lxtywh, hb_hcu_bcxh)
        elif use_hb and camera == '128':
            phcu_count(infos, west_p_lxtywh, pass_lxtywh, west_p_bcxh)
        elif use_hb and camera == '130':
            phcu_count(infos, east_p_lxtywh, pass_lxtywh, east_p_bcxh)
        elif not use_hb and camera == '128':
            phcu_count(infos, west_p_lxtywh, west_hcu_lxtywh, west_phcu_bcxh)
        elif not use_hb and camera == '130':
            phcu_count(infos, east_p_lxtywh, east_hcu_lxtywh, east_phcu_bcxh)

    # ---Condition check.--- #
    if use_hb:
        p_count_cond = len(west_p_bcxh['p']) == 1 and len(east_p_bcxh['p']) == 1
        u_count_cond = len(hb_hcu_bcxh['u']) == 1
        hc_count_cond = len(hb_hcu_bcxh['hc']) == 2
        if p_count_cond and u_count_cond and hc_count_cond:
            hb_c_id, hb_h_id = (1, 0) if hb_hcu_bcxh['hc'][0][0] < hb_hcu_bcxh['hc'][1][0] else (0, 1)
            hb_c_cond = hb_hcu_bcxh['hc'][hb_c_id][1] < hb_hcu_bcxh['hc'][hb_h_id][1]
            hb_u_cond = hb_hcu_bcxh['hc'][hb_c_id][0] < hb_hcu_bcxh['u'][0][0]
            if hb_c_cond and hb_u_cond: start_state = True
    else:
        p_count_cond = len(west_phcu_bcxh['p']) == 1 and len(east_phcu_bcxh['p']) == 1
        u_count_cond = len(west_phcu_bcxh['u']) == 1 and len(east_phcu_bcxh['u']) == 1
        hc_count_cond = len(west_phcu_bcxh['hc']) == 2 and len(east_phcu_bcxh['hc']) == 2
        if p_count_cond and u_count_cond and hc_count_cond:
            west_c_id, west_h_id = (1, 0) if west_phcu_bcxh['hc'][0][0] < west_phcu_bcxh['hc'][1][0] else (0, 1)
            east_c_id, east_h_id = (1, 0) if east_phcu_bcxh['hc'][0][0] > east_phcu_bcxh['hc'][1][0] else (0, 1)
            west_c_cond = west_phcu_bcxh['hc'][west_c_id][1] < west_phcu_bcxh['hc'][west_h_id][1]
            east_c_cond = east_phcu_bcxh['hc'][east_c_id][1] < east_phcu_bcxh['hc'][east_h_id][1]
            west_u_cond = west_phcu_bcxh['hc'][west_c_id][0] < west_phcu_bcxh['u'][0][0]
            east_u_cond = east_phcu_bcxh['hc'][east_c_id][0] > east_phcu_bcxh['u'][0][0]
            if west_c_cond and east_c_cond and west_u_cond and east_u_cond: start_state = True

    if start_state:
        print('++++++++++++++++  START TRIGGER  ++++++++++++++++')
    else:
        print('-----------------  START PASS  -----------------')

    return start_state


def calculate_distance(now_point, other_points):
    """
    Calculate euclidean distance between points.
    :param now_point: list, [x, y]
    :param other_points: ndarray, [[x, y], ...]
    :return tuple of objects: (min_id, min_dist)
            min_id: int, the index of minimum distance
            min_dist: float, the minimum distance
    """
    now_point = np.array(now_point)
    distances = np.square(now_point - other_points)
    distances = np.sum(distances, axis=1)
    distances = np.sqrt(distances)
    min_id = np.argmin(distances)
    min_dist = distances[min_id]
    return min_id, min_dist


def pause_initial():
    """
    Initialize global pause variable.
    :return None
    """
    global NOW_IMGID
    global TRACK_DUGOUT

    NOW_IMGID = defaultdict(int)
    TRACK_DUGOUT = defaultdict(list)


def all_initial():
    """
    Initialize global all variable.
    :return None
    """
    global SYSTEM_STATE

    SYSTEM_STATE = False
    pause_initial()


def pause_trigger(camera, infos, img_wh, point_dist, dugout_in_ty, dugout_out_wty):
    global NOW_IMGID
    global TRACK_DUGOUT

    NOW_IMGID[camera] += 1
    for i, info in enumerate(infos, 1):
        bcx, bcy, w, h, c, score = get_detect_info(info)

        # ---Check dugout.--- #
        imgid_bcx_bcy = [NOW_IMGID[camera], bcx, bcy]
        if not TRACK_DUGOUT[camera] and dugout_in_ty < bcy <= img_wh[1]:
            print('- in dugout triggered by ({}, {})'.format(bcx, bcy))
            TRACK_DUGOUT[camera].append([imgid_bcx_bcy + [0]])
        elif TRACK_DUGOUT[camera] and dugout_in_ty < bcy <= img_wh[1]:
            track_infos = np.array([_info[-1] for _info in TRACK_DUGOUT[camera]])
            min_id, min_dist = calculate_distance([bcx, bcy], track_infos[:, 1:3])
            track_imgid, track_bcx, track_bcy, track_state = track_infos[min_id]
            dist_cond = min_dist < point_dist
            if not dist_cond:
                print('- in dugout triggered by ({}, {})'.format(bcx, bcy))
                TRACK_DUGOUT[camera].append([imgid_bcx_bcy + [0]])
            elif dist_cond and not track_imgid == NOW_IMGID[camera]:
                if track_bcy - bcy >= 20 and NOW_IMGID[camera] - track_imgid < 10 and not track_state == 2:
                    print('- in dugout tracked by forward ({}, {})'.format(bcx, bcy))
                    TRACK_DUGOUT[camera][min_id].append(imgid_bcx_bcy + [1])
                elif track_bcy - bcy >= 20 and track_state == 2:
                    print('- in dugout triggered by player ({}, {})'.format(bcx, bcy))
                    TRACK_DUGOUT[camera][min_id].append(imgid_bcx_bcy + [1])
                elif bcy - track_bcy >= 20 and NOW_IMGID[camera] - track_imgid < 10 and not track_state == 1:
                    print('- in dugout tracked by backward ({}, {})'.format(bcx, bcy))
                    TRACK_DUGOUT[camera][min_id].append(imgid_bcx_bcy + [2])
                elif bcy - track_bcy >= 20 and track_state == 1:
                    print('- in dugout triggered by ({}, {})'.format(bcx, bcy))
                    TRACK_DUGOUT[camera].append([imgid_bcx_bcy + [0]])
        elif TRACK_DUGOUT[camera] and dugout_out_wty[1] < bcy <= dugout_in_ty:
            track_infos = np.array([_info[-1] for _info in TRACK_DUGOUT[camera]])
            min_id, min_dist = calculate_distance([bcx, bcy], track_infos[:, 1:3])
            track_imgid, track_bcx, track_bcy, track_state = track_infos[min_id]
            dist_cond = min_dist < point_dist
            imgid_cond = not track_imgid == NOW_IMGID[camera] and NOW_IMGID[camera] - track_imgid < 10
            state_cond = track_state == 1
            forward_cond = track_bcy - bcy >= 20
            x_cond = img_wh[0] - 900 < bcx < img_wh[0] if camera == '128' else 0 < bcx < dugout_out_wty[0]
            if dist_cond and imgid_cond and state_cond and forward_cond and x_cond:
                print('- out dugout tracked by forward ({}, {})'.format(bcx, bcy))
                TRACK_DUGOUT[camera][min_id].append(imgid_bcx_bcy + [1])


def pause(camera_detect, img_wh=(4096, 2160), track_interval=3, point_dist=170, dugout_in_ty=1800,
          dugout_out_wty=(900, 1490), west_hitter_lxrx=(3340, 3600), east_hitter_lxrx=(380, 620)):
    """
    Determine pause by certain condition.
    :param camera_detect: dict, {'camera_ip': ['img_name', 'cx cy lx ty w h class score', ...], ...}
    :param img_wh: tuple of ints, (width, height) of image
    :param track_interval: int, the interval number of tracked image
    :param point_dist: int, the distance between two points
    :param dugout_in_ty: int, the top y of dugout range in area
    :param dugout_out_wty: tuple of ints, (width, top y) of dugout range out area
    :param west_hitter_lxrx: tuple of ints, (left x, right x) of hitter range in west camera
    :param east_hitter_lxrx: tuple of ints, (left x, right x) of hitter range in east camera
    :return pause_state: bool, start or not
    """

    for camera, detect in camera_detect.items():
        if camera == '124' or camera == '125': continue
        img_name, infos = detect[0], detect[1:]
        print('\n[{} by camera {}] :'.format(img_name, camera))
        pause_trigger(camera, infos, img_wh, point_dist, dugout_in_ty, dugout_out_wty)

    pause_state = 'initial'
    state_count = defaultdict(int)
    dugout_in_removes = defaultdict(list)
    for camera, track_records in TRACK_DUGOUT.items():
        print('\n* Track', camera)
        forward_num = 0
        backward_num = 0
        for track_record in track_records:
            record_num = len(track_record)
            track_imgid, track_bcx, track_bcy, track_state = track_record[-1]
            if record_num > track_interval and track_record[1][-1] == 1 and track_state == 1 and track_bcy < dugout_in_ty + 150:
                print('+', track_record)
                forward_num += 1
            elif record_num > track_interval and track_record[1][-1] == 2 and track_state == 2:
                print('-', track_record)
                backward_num += 1
            elif record_num < track_interval and NOW_IMGID[camera] - track_imgid >= 30:
                print('del noise', track_record)
                dugout_in_removes[camera].append(track_record)
            elif record_num > track_interval and track_record[1][-1] == 2 and track_state == 1 and NOW_IMGID[camera] - track_imgid >= 15:
                print('del player', track_record)
                dugout_in_removes[camera].append(track_record)
            else:
                print('.', track_record)

            if camera == '128': hitter_x_cond = west_hitter_lxrx[0] < track_bcx < west_hitter_lxrx[1]
            else: hitter_x_cond = east_hitter_lxrx[0] < track_bcx < east_hitter_lxrx[1]
            hitter_y_cond = dugout_out_wty[1] < track_bcy < dugout_out_wty[1] + 150
            if hitter_x_cond and hitter_y_cond:
                print('!!!!!!!!!!!!!!!!!  MAYBE HITTER  !!!!!!!!!!!!!!!!!')
                pause_state = 'all_close'
                return pause_state

        if forward_num > 0 and not forward_num - backward_num == 0:
            state_count['trigger'] += 1
        elif forward_num > 0 and backward_num > 0 and forward_num - backward_num == 0:
            state_count['close'] += 1

    for camera, track_records in dugout_in_removes.items():
        for track_record in track_records:
            TRACK_DUGOUT[camera].remove(track_record)

    if state_count['trigger'] > 0:
        print('****************  PAUSE TRIGGER  ****************')
        pause_state = 'any_trigger'
    elif state_count['close'] > 0 and state_count['close'] == len([rs for rs in TRACK_DUGOUT.values() if rs]):
        print('+++++++++++++++++  PAUSE CLOSE  +++++++++++++++++')
        pause_state = 'all_close'
    elif pause_state == 'initial':
        print('-----------------  PAUSE PASS  -----------------')

    return pause_state


def pause_start_fusion(camera_detect, external_state='auto', start_use_hb=False):
    """
    Using different yolo detect results to get start or pause state.
    :param camera_detect: dict, {'camera_ip': ['img_name', 'cx cy lx ty w h class score', ...], ...}
    :param external_state: str, the state gave by external input
            'auto': follow the original system state
            'start': have started and will detect pause state
            'pause': have paused and will detect start state
            'initial': initialize the whole pause and start system
    :param start_use_hb: bool, use hb camera or not to detect start state
    :return states: bool, True=start or False=pause
    """
    print('\n=======================================================')
    now_state = False
    start_state = start(camera_detect, start_use_hb)
    # start_state = True

    global SYSTEM_STATE

    if external_state == 'start':
        now_state = True
        SYSTEM_STATE = True
        pause_initial()
    elif external_state == 'pause':
        now_state = False
        SYSTEM_STATE = False
    elif external_state == 'initial':
        all_initial()
    elif not external_state == 'auto':
        raise ValueError('Must be "auto", "initial", "start", or "pause", '
                         'but get external_state = {}'.format(external_state))

    if not SYSTEM_STATE and start_state:
        now_state = True
        SYSTEM_STATE = True
    elif SYSTEM_STATE:
        pause_state = pause(camera_detect)
        if pause_state == 'initial':
            now_state = True
        elif pause_state == 'any_trigger':
            now_state = False
        elif pause_state == 'all_close':
            now_state = False
            SYSTEM_STATE = False
            pause_initial()
        else:
            raise ValueError('Check pause_state = {}'.format(pause_state))
    print('======now_state:', now_state)
    return now_state, start_state


if __name__ == '__main__':
    print()
    s = timer()

    # ---Start Verify--- #
    # west_img_detects = read_detect_txt(WEST_INFO_FILE)
    # east_img_detects = read_detect_txt(EAST_INFO_FILE)
    # for west_img_detect, east_img_detect in zip(west_img_detects, east_img_detects):
    #     _camera_detect = {'128': west_img_detect, '130': east_img_detect}
    #     states = pause_start_fusion(_camera_detect, start_use_hb=False)
    #     print(states)

    # hb_img_detects = read_detect_txt(HB_INFO_FILE)
    # for hb_img_detect in hb_img_detects:
    #     _camera_detect = {'125': hb_img_detect}
    #     states = pause_start_fusion(_camera_detect, start_use_hb=True)
    #     print(states)

    # # ---Pause Verify--- #
    # west_img_detects = read_detect_txt(WEST_INFO_FILE)
    # east_img_detects = read_detect_txt(EAST_INFO_FILE)
    # start_id = 0
    # end_id = 200
    # for i, img_detects in enumerate(zip(west_img_detects, east_img_detects)):
    #     west_img_detect, east_img_detect = img_detects
    #     if i < start_id: continue
    #     _camera_detect = {'128': west_img_detect, '130': east_img_detect}
    #     # _camera_detect = {'128': west_img_detect}
    #     # _camera_detect = {'130': east_img_detect}
    #
    #     if i == start_id: external_state = 'start'
    #     else: external_state = 'auto'
    #     states = pause_start_fusion(_camera_detect, external_state=external_state, start_use_hb=False)
    #     # print(states)
    #     if i == end_id: break

    # ---All Verify--- #
    west_img_detects = read_detect_txt(WEST_INFO_FILE)
    east_img_detects = read_detect_txt(EAST_INFO_FILE)
    start_id = 0
    end_id = 270
    for i, img_detects in enumerate(zip(west_img_detects, east_img_detects)):
        west_img_detect, east_img_detect = img_detects
        if i < start_id: continue
        _camera_detect = {'128': west_img_detect, '130': east_img_detect}

        if i == start_id: external_state = 'start'
        else: external_state = 'auto'
        states = pause_start_fusion(_camera_detect, external_state=external_state, start_use_hb=False)
        print(states)
        if i == end_id: break

    e = timer()
    print('\nDetect time :', e - s)
