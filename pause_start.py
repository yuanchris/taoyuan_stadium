from yolov3.utils import *

INFO_DIR = 'pause_start_info/'
# INFO_FILE = INFO_DIR + 'test_0623_west_right_pause_4_info.txt'
INFO_FILE = INFO_DIR + 'test_0623_east_left_pause_1_info.txt'
# INFO_FILE = INFO_DIR + 'test_0623_west_right_192914_195314_info.txt'
# INFO_FILE = INFO_DIR + 'test_0623_east_left_192907_195307_info.txt'

PAUSE_IMG_INDEX = 0
PAUSE_START_1 = 0
PAUSE_START_2 = 0
PAUSE_INFO_1 = []
PAUSE_INFO_2 = []

SOMEBODY_IMG_INDEX = 0
SOMEBODY_START = 0
SOMEBODY_INFO = None
SOMEBODY_STATE = False

PLAYER_IMG_INDEX_1 = 0
PLAYER_START_1 = 0
PLAYER_INFO_1 = None
PLAYER_STATE_1 = False

PLAYER_IMG_INDEX_2 = 0
PLAYER_START_2 = 0
PLAYER_INFO_2 = None
PLAYER_STATE_2 = False

PAUSE_STATE = False
START_STATE = True


def get_item(item, get_score=False):
    """
    Get item and convert to the indeed format.
    :param item: str, 'c_x c_y bc_x bc_y w h class score'
    :param get_score: bool, get score or not
    :return: new_item: list of object
             get score: [bc_x, bc_y, w, h, class, score]
             not score: [bc_x, bc_y, w, h, class]
    """
    _item = item.split()
    new_item = list(map(int, _item[2:-1]))
    if get_score: new_item.append(float(_item[-1]))
    return new_item


def pause_initial():
    """
    Pause global variable initialize.
    :return: None
    """
    global PAUSE_IMG_INDEX
    global PAUSE_START_1
    global PAUSE_START_2
    global PAUSE_INFO_1
    global PAUSE_INFO_2
    PAUSE_IMG_INDEX = 0
    PAUSE_START_1 = 0
    PAUSE_START_2 = 0
    PAUSE_INFO_1 = []
    PAUSE_INFO_2 = []


def track_initial(range_num):
    """
    Track player variable initialize.
    :param range_num: int, the number of range
            0: somebody like batboy or coach
            1: player in range 1
            2: player in range 2
    :return: None
    """
    if range_num == 0:
        global SOMEBODY_IMG_INDEX
        global SOMEBODY_START
        global SOMEBODY_INFO
        global SOMEBODY_STATE
        SOMEBODY_IMG_INDEX = 0
        SOMEBODY_START = 0
        SOMEBODY_INFO = None
        SOMEBODY_STATE = False
        pause_initial()
    elif range_num == 1:
        global PLAYER_IMG_INDEX_1
        global PLAYER_START_1
        global PLAYER_INFO_1
        global PLAYER_STATE_1
        PLAYER_IMG_INDEX_1 = 0
        PLAYER_START_1 = 0
        PLAYER_INFO_1 = None
        PLAYER_STATE_1 = False
        pause_initial()
    elif range_num == 2:
        global PLAYER_IMG_INDEX_2
        global PLAYER_START_2
        global PLAYER_INFO_2
        global PLAYER_STATE_2
        PLAYER_IMG_INDEX_2 = 0
        PLAYER_START_2 = 0
        PLAYER_INFO_2 = None
        PLAYER_STATE_2 = False
        pause_initial()
    else:
        raise ValueError('Check Range Number.')


def track_player_1(info, range1_w, track_distance=100):
    """
    Track player in range 1 to start.
    :param info: list of strings, ['img_name', 'c_x c_y bc_x bc_y w h class score', ...]
    :param range1_w: int, the width of range 1
    :param track_distance: int, the distance want to track
    :return: None
    """
    global PLAYER_INFO_1
    global PLAYER_IMG_INDEX_1
    PLAYER_IMG_INDEX_1 += 1
    items = info.split(',')
    img_name = items[0]
    min_info = None
    min_distance = float('inf')
    player_bcx, player_bcy = PLAYER_INFO_1
    for item in items[1:]:
        bc_x, bc_y, w, h, c = get_item(item)
        d = abs(player_bcx - bc_x) + abs(player_bcy - bc_y)
        if d <= min_distance:
            min_distance = d
            min_info = [bc_x, bc_y, w, h, c]

    global PLAYER_STATE_1
    global PLAYER_START_1
    if min_distance <= track_distance:
        bc_x, bc_y = min_info[:2]
        PLAYER_INFO_1 = (bc_x, bc_y)
        PLAYER_STATE_1 = True
        PLAYER_START_1 = PLAYER_IMG_INDEX_1
        # print('{}  ({}, {}) {}'.format(img_name, bc_x, bc_y, ' Player1 Exist'))
        # ---Out of Range1.--- #
        if bc_x > range1_w:
            track_initial(range_num=1)
    else:
        if PLAYER_STATE_1 and ((PLAYER_IMG_INDEX_1 - PLAYER_START_1) < 4):
            pass
        else:
            # print(img_name, '+++ player1 +++')
            track_initial(range_num=1)


def track_player_2(info, range2_sy, track_distance=100):
    """
    Track player in range 2 to start.
    :param info: list of strings, ['img_name', 'c_x c_y bc_x bc_y w h class score', ...]
    :param range2_sy: int, the start y of range 2
    :param track_distance: int, the distance want to track
    :return: None
    """
    global PLAYER_INFO_2
    global PLAYER_IMG_INDEX_2
    PLAYER_IMG_INDEX_2 += 1
    items = info.split(',')
    img_name = items[0]
    min_info = None
    min_distance = float('inf')
    player_bcx, player_bcy = PLAYER_INFO_2
    for item in items[1:]:
        bc_x, bc_y, w, h, c = get_item(item)
        d = abs(player_bcx - bc_x) + abs(player_bcy - bc_y)
        if d <= min_distance:
            min_distance = d
            min_info = [bc_x, bc_y, w, h, c]

    global PLAYER_STATE_2
    global PLAYER_START_2
    if min_distance <= track_distance:
        bc_x, bc_y = min_info[:2]
        PLAYER_INFO_2 = (bc_x, bc_y)
        PLAYER_STATE_2 = True
        PLAYER_START_2 = PLAYER_IMG_INDEX_2
        # print('{}  ({}, {}) {}'.format(img_name, bc_x, bc_y, ' Player2 Exist'))
        # ---Out of Range2.--- #
        if bc_y < (range2_sy - 200):
            track_initial(range_num=2)
            # print(img_name, bc_x, bc_y, PLAYER_STATE_2)
    else:
        if PLAYER_STATE_2 and ((PLAYER_IMG_INDEX_2 - PLAYER_START_2) < 4):
            pass
        else:
            # print(img_name, '+++ player2 +++')
            track_initial(range_num=2)


def track_somebody(info, camera, track_distance=100, pitcher_mound_wh=(540, 250)):
    """
    Track somebody except the player and umpire until disappear.
    :param info: list of strings, ['img_name', 'c_x c_y bc_x bc_y w h class score', ...]
    :param camera: str, 'west' or 'east', camera position
    :param track_distance: int, the distance want to track
    :param pitcher_mound_wh: tuple, (pitcher_mound_w, pitcher_mound_h), the width and height of pitcher mound
    :return: SOMEBODY_INFO: bool, somebody exist or not
    """
    global SOMEBODY_INFO
    global SOMEBODY_IMG_INDEX
    SOMEBODY_IMG_INDEX += 1
    items = info.split(',')
    img_name = items[0]
    min_info = None
    min_distance = float('inf')
    t_bcx, t_bcy = SOMEBODY_INFO
    pitcher_mound_w, pitcher_mound_h = pitcher_mound_wh
    for item in items[1:]:
        bc_x, bc_y, w, h, c = get_item(item)
        if h > 160 and c == 1: continue
        d = abs(t_bcx - bc_x) + abs(t_bcy - bc_y)
        if d <= min_distance:
            min_distance = d
            min_info = [bc_x, bc_y, w, h, c]

    global SOMEBODY_STATE
    global SOMEBODY_START
    if min_distance <= track_distance:
        bc_x, bc_y = min_info[:2]
        SOMEBODY_INFO = (bc_x, bc_y)
        SOMEBODY_STATE = True
        SOMEBODY_START = SOMEBODY_IMG_INDEX
        # print('{} ({}, {}) {}'.format(img_name, bc_x, bc_y, ' Exist'))
        # ---Coach in pitcher's mound.--- #
        if camera == 'west': pitcher_mound_sx = 3050
        else: pitcher_mound_sx = 1550
        pitcher_mound_sy = 1300
        pitcher_mound_x_cond = pitcher_mound_sx > bc_x > (pitcher_mound_sx - pitcher_mound_w)
        pitcher_mound_y_cond = (pitcher_mound_sy + pitcher_mound_h) > bc_y > pitcher_mound_sy
        pitcher_mound_cond = pitcher_mound_x_cond and pitcher_mound_y_cond
        if pitcher_mound_cond:
            track_initial(range_num=0)
    else:
        if SOMEBODY_STATE and ((SOMEBODY_IMG_INDEX - SOMEBODY_START) < 15):
            pass
        else:
            track_initial(range_num=0)

    return SOMEBODY_INFO


def pause(info, camera, track_count=4, filter_score=0.6, range1_sy=1100,
          range2_sy=2070, range1_wh=(250, 500), range2_w=1600, range_m=0.1):
    """
    Determine pause by certain condition.
    :param info: list of strings, ['img_name', 'c_x c_y bc_x bc_y w h class score', ...]
    :param camera: str, 'west' or 'east', camera position
    :param track_count: int, the number for tracking
    :param filter_score: float, filter the certain box by score
    :param range1_sy: int, the start y of range 1
    :param range2_sy: int, the start y of range 2
    :param range1_wh: tuple, (range1_w, range1_h), the width and height of range 1
    :param range2_w: int, the width of range 2
    :param range_m: float, the slope of range
    :return: pause_state: bool, pause or not
    """
    pause_state = False
    global PLAYER_INFO_1
    global PLAYER_INFO_2

    if PLAYER_INFO_1:
        track_player_1(info, range1_wh[0])
    if PLAYER_INFO_2:
        track_player_2(info, range2_sy)

    if (not PLAYER_INFO_1) and (not PLAYER_INFO_2):
        global PAUSE_IMG_INDEX
        PAUSE_IMG_INDEX += 1
        items = info.split(',')
        img_name = items[0]

        range1_w, range1_h = range1_wh
        if camera == 'west':
            range1_sx = range1_w
            range2_sx = 4096
        else:
            range_m = (-1) * abs(range_m)
            range1_sx = 4096 - range1_w
            range2_sx = 0

        for item in items[1:]:
            bc_x, bc_y, w, h, c, score = get_item(item, get_score=True)
            if w > h: continue
            _y = range2_sy - range_m * int(range2_sx - bc_x)

            # ---Range Filter.--- #
            if camera == 'west':
                range1_cond = (range1_sx > bc_x) and ((range1_sy + range1_h) > bc_y > range1_sy)
                range2_cond = (range2_sx > bc_x > (range2_sx - range2_w)) and (bc_y > _y)
            else:
                range1_cond = (bc_x > range1_sx) and ((range1_sy + range1_h) > bc_y > range1_sy)
                range2_cond = (range2_w > bc_x > range2_sx) and (bc_y > _y)

            global SOMEBODY_INFO
            # ---Range 1 - First or Third Field Track.--- #
            if range1_cond and score >= filter_score:
                global PAUSE_START_1
                global PAUSE_INFO_1
                if PAUSE_START_1 == 0:
                    PAUSE_START_1 = PAUSE_IMG_INDEX
                    PAUSE_INFO_1.append((bc_x, bc_y))
                    # print(img_name, '', PAUSE_INFO_1[0])
                else:
                    pause_count = PAUSE_IMG_INDEX - PAUSE_START_1
                    if pause_count == 0:
                        # ---Same Image.--- #
                        PAUSE_INFO_1.append((bc_x, bc_y))
                    else:
                        # ---Other Image.--- #
                        if pause_count > track_count:
                            pause_initial()
                            break
                        distances = [abs(p_bcx - bc_x) + abs(p_bcy - bc_y) for p_bcx, p_bcy in PAUSE_INFO_1]
                        distances = np.array(distances)
                        min_index = np.argmin(distances)
                        pause_bcx, pause_bcy = PAUSE_INFO_1[min_index]
                        west_player_cond = (camera == 'west') and ((pause_bcx - bc_x) > 10)
                        east_player_cond = (camera == 'east') and ((bc_x - pause_bcx) > 10)
                        west_somebody_cond = (camera == 'west') and ((bc_x - pause_bcx) > 10)
                        east_somebody_cond = (camera == 'east') and ((pause_bcx - bc_x) > 10)
                        # ---Track Player1.--- #
                        if west_player_cond or east_player_cond:
                            # print('{}  ({}, {}) {}'.format(img_name, bc_x, bc_y, ' Player1 Tracked'))
                            PLAYER_INFO_1 = (bc_x, bc_y)
                            break
                        # ---Track Somebody.--- #
                        elif west_somebody_cond or east_somebody_cond:
                            # print('{}  ({}, {}) {}'.format(img_name, bc_x, bc_y, ' Range1 Tracked'))
                            SOMEBODY_INFO = (bc_x, bc_y)
                            pause_state = True
                            pause_initial()
                            print('{}  {}  PAUSE  !!!!!!!!!!!!!!!!!!!!!!!'.format(img_name, camera.upper()))
                            break
                        else:
                            # print('{}  ({}, {}) {}'.format(img_name, bc_x, bc_y, ' !!!'))
                            pass

            # ---Range 2 - Dugout Track.--- #
            if range2_cond and score >= filter_score:
                global PAUSE_START_2
                global PAUSE_INFO_2
                if PAUSE_START_2 == 0:
                    PAUSE_START_2 = PAUSE_IMG_INDEX
                    PAUSE_INFO_2.append((bc_x, bc_y))
                    # print(img_name, '', PAUSE_INFO_2[0])
                else:
                    pause_count = PAUSE_IMG_INDEX - PAUSE_START_2
                    if pause_count == 0:
                        # ---Same Image.--- #
                        PAUSE_INFO_2.append((bc_x, bc_y))
                    else:
                        # ---Other Image.--- #
                        if pause_count > track_count:
                            pause_initial()
                            break
                        distances = [abs(p_bcx - bc_x) + abs(p_bcy - bc_y) for p_bcx, p_bcy in PAUSE_INFO_2]
                        distances = np.array(distances)
                        min_index = np.argmin(distances)
                        pause_bcx, pause_bcy = PAUSE_INFO_2[min_index]
                        # ---Track Player2.--- #
                        if bc_y - pause_bcy >= 15 and abs(pause_bcx - bc_x) <= 150:
                            # print('{}  ({}, {}) {}'.format(img_name, bc_x, bc_y, ' Player2 Tracked'))
                            PLAYER_INFO_2 = (bc_x, bc_y)
                            break
                        # ---Track Somebody.--- #
                        elif pause_bcy - bc_y >= 10 and abs(pause_bcx - bc_x) <= 150:
                            # print('{}  ({}, {}) {}'.format(img_name, bc_x, bc_y, ' Range2 Tracked'))
                            SOMEBODY_INFO = (bc_x, bc_y)
                            pause_state = True
                            pause_initial()
                            print('{}  {}  PAUSE  !!!!!!!!!!!!!!!!!!!!!!!'.format(img_name, camera.upper()))
                            break
                        else:
                            # print('{}  ({}, {}) {}'.format(img_name, bc_x, bc_y, ' !!!'))
                            pass

    return pause_state


def start(info, camera, range_sy=1950, range_w=800, pitcher_wh=(100, 40), catcher_wh=(90, 50),
          pitcher_rxy=(2810, 1360), catcher_rxy=(3910, 1845), hitter_mxy=(-0.3, 0.45),
          hitter_1xy=(3755, 1850), hitter_2xy=(3593, 1821),
          hitter_3xy=(3875, 1815), hitter_4xy=(3695, 1790)):
    """
    Determine start by certain condition.
    :param info: list of strings, ['img_name', 'c_x c_y bc_x bc_y w h class score', ...]
    :param camera: str, 'west' or 'east', camera position
    :param range_sy: int, the start y of range
    :param range_w: int, the width of range
    :param pitcher_wh: tuple, (pitcher_w, pitcher_h), the width and height of pitcher range
    :param catcher_wh: tuple, (catcher_w, catcher_h), the width and height of catcher range
    :param pitcher_rxy: tuple, (pitcher_rx, pitcher_ry), the right x and y of pitcher range
    :param catcher_rxy: tuple, (catcher_rx, catcher_ry), the right x and y of catcher range
    :param hitter_mxy: tuple, (hitter_mx, hitter_my), the slope x and y of hitter range
    :param hitter_1xy: tuple, (hitter_1x, hitter_1y), the right x and y of left hitter
    :param hitter_2xy: tuple, (hitter_2x, hitter_2y), the left x and y of left hitter
    :param hitter_3xy: tuple, (hitter_3x, hitter_3y), the right x and y of right hitter
    :param hitter_4xy: tuple, (hitter_4x, hitter_4y), the left x and y of right hitter
    :return: start_state: bool, start or not
    """
    items = info.split(',')
    img_name = items[0]
    start_state = False
    pitcher_count = 0
    catcher_count = 0
    hitter_count = 0

    for item in items[1:]:
        bc_x, bc_y, w, h, c, score = get_item(item, get_score=True)
        if c == 1 or w > h: continue

        # ---Range Filter.--- #
        if camera == 'west': range_cond = (4096 > bc_x > (4096 - range_w)) and (bc_y > range_sy)
        else: range_cond = (range_w > bc_x > 0) and (bc_y > range_sy)
        if range_cond: break

        # ---Pitcher.--- #
        if pitcher_count == 0:
            p_w, p_h = pitcher_wh
            p_rx, p_ry = pitcher_rxy
            p_x_cond = p_rx > bc_x > (p_rx - p_w)
            p_y_cond = (p_ry + p_h) > bc_y > p_ry
            if p_x_cond and p_y_cond:
                # print('Pitcher Checked.')
                pitcher_count += 1
                continue

        # ---Catcher.--- #
        if catcher_count == 0:
            c_w, c_h = catcher_wh
            c_rx, c_ry = catcher_rxy
            c_x_cond = c_rx > bc_x > (c_rx - c_w)
            c_y_cond = (c_ry + c_h) > bc_y > c_ry
            if c_x_cond and c_y_cond:
                # print('Catcher Checked.')
                catcher_count += 1
                continue

        # ---Hitter.--- #
        h_d = 10
        h_mx, h_my = hitter_mxy
        h_1x, h_1y = hitter_1xy
        h_2x, h_2y = hitter_2xy
        h_3x, h_3y = hitter_3xy
        h_4x, h_4y = hitter_4xy
        l_cond1 = h_1x > bc_x > h_2x
        r_cond1 = h_3x > bc_x > h_4x
        l_cond2 = (h_1y + h_d) > bc_y > (h_2y - h_d)
        r_cond2 = (h_3y + h_d) > bc_y > (h_4y - h_d)
        if l_cond1 and l_cond2:
            m_1b = (h_1y - bc_y) / (h_1x - bc_x)
            m_2b = (h_2y - bc_y) / (h_2x - bc_x)
            l_cond3 = h_my > m_1b > h_mx
            l_cond4 = h_my > m_2b > h_mx
            if l_cond3 and l_cond4:
                # print('Left Hitter Checked.')
                if score < 0.9: continue
                hitter_count += 1
        if r_cond1 and r_cond2:
            if h < 100: continue
            m_3b = (h_3y - bc_y) / (h_3x - bc_x)
            m_4b = (h_4y - bc_y) / (h_4x - bc_x)
            r_cond3 = h_my > m_3b > h_mx
            r_cond4 = h_my > m_4b > h_mx
            if r_cond3 and r_cond4:
                # print('Right Hitter Checked.')
                if score < 0.9: continue
                hitter_count += 1

    if pitcher_count and catcher_count and hitter_count == 1: start_state = True
    if start_state: print('{}  {}  START  ++++++++++++++++++++++++++++++++++'.format(img_name, camera.upper()))
    return start_state


def pause_start_fusion(camera_detects):
    """
    Using different yolo detect result to get start or pause state.
    :param camera_detects: dict, {str: list}, ex: {'camera_ip': ['c_x, c_y, left, top, w, h, c, score', ...]}
    :return state: tuple of bools, (pause, start), ex: (True, False)
    """
    pause_state = False
    start_state = False
    # for camera, yolo_info in camera_detects:
    #     pause_state = pause(yolo_info, camera)
    #     start_state = start(yolo_info, camera)

    return pause_state, start_state


if __name__ == '__main__':
    print()

    _camera = INFO_FILE.split('/')[-1].split('_')[2]
    infos = read_lines(INFO_FILE)
    for _info in infos:
        _pause_state = pause(_info, _camera)

    # _somebody_exist = False
    # _camera = INFO_FILE.split('/')[-1].split('_')[2]
    # infos = read_lines(INFO_FILE)
    # for _info in infos:
    #     if (not PAUSE_STATE) and START_STATE:
    #         _pause_state = pause(_info, _camera)
    #         if _pause_state:
    #             PAUSE_STATE = True
    #             START_STATE = False
    #             _somebody_exist = True
    #             continue
    #
    #     elif PAUSE_STATE and (not START_STATE):
    #         if _somebody_exist:
    #             _somebody_exist = track_somebody(_info, _camera)
    #         else:
    #             _start_state = start(_info, _camera)
    #             if _start_state:
    #                 START_STATE = True
    #                 PAUSE_STATE = False

