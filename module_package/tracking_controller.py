import myqueue, threading, os, timeit, time, requests
import socket, json
import websockets
import asyncio

from .yolo_controller import yolo_controller
from .highlight_replay.highlight_replay import Highlight_Repleay as replay_controller
from .tracking_board.strategy_board import status as ST
from .tracking_board.track_test import Camera_System
from .baseball_tracking_app import PitecedBallTrackingApp
from .pause_start import pause_start_fusion

class tracking_controller:
    def __init__(self, image_queues, raw_queues, parameters, replay_parameters, redis_enabled):
        self.image_queues = image_queues
        self.raw_queues = raw_queues
        self.parameters = parameters
        self.replay_parameters = replay_parameters
        self.redis_upload_enabled = redis_enabled
        self.tracking_is_running = False

        self.detection_module = dict()
        self.tracking_module = dict()

        self.detection_result = dict()
        self.detection_result_toTrack = dict()
        self.detection_result_toreplay = dict()

        self.tracking_camera_list = list()
        self.tracking_result = list()
        self.tracking_result_lock = threading.Lock()
        self.tracking_reset = False
        # self.loop = None

        self.game_information = {"Code":"E000","Message":"查詢成功!",
            "GameNo":104,"CurrentInning":0,"GameState":1, 'pause_start': True, 
            "Atk_H":"11 林晨樺","Atk_1H":"","Atk_2H":"","Atk_3H":"",
            "Def_P":"01 蘇俊璋","Def_C":"15 劉昱言","Def_1B":"12 蔡豐安","Def_2B":"30 林泓育",
            "Def_3B":"15 陳志遠","Def_SS":"17 Claire","Def_LF":"35 成晉","Def_CF":"18 Matt",
            "Def_RF":"19 Lily"}

        self.now_state = True
        self.start_state = True

        for camera in parameters:
            self.detection_result_toTrack[camera["index"]] = None
            self.detection_result_toreplay[camera["index"]] = None
            if camera["tracking_activate"]:
                new_queue = myqueue.Queue(1)
                self.detection_result_toTrack[camera["index"]] = new_queue
            #     self.tracking_result[camera["index"]] = list()
            if camera["replay_activate"]:
                new_queue = myqueue.Queue(1)
                self.detection_result_toreplay[camera["index"]] = new_queue
            if camera["detection_activate"]:
                new_queue = myqueue.Queue(1)
                self.detection_result[camera["index"]] = new_queue
                # if camera["replay_activate"]:
                #     detection_module = yolo_controller(camera["gpu_id"], camera["mount_root"], self.image_queues[camera["index"]], self.detection_result[camera["index"]], self.detection_result_toreplay[camera["index"]])
                # else:
                #     detection_module = yolo_controller(camera["gpu_id"], camera["mount_root"], self.image_queues[camera["index"]], self.detection_result[camera["index"]])

                detection_module = yolo_controller(camera["gpu_id"], camera["mount_root"], self.image_queues[camera["index"]], 
                    self.detection_result[camera["index"]], self.detection_result_toTrack[camera["index"]], self.detection_result_toreplay[camera["index"]])
                self.detection_module[camera["index"]] = detection_module
            # if camera["baseball_tracking_activate"]:
            #     # TODO
    
    def run(self):
        self.tracking_is_running = True
        for camera in self.parameters:
            if camera["detection_activate"]:
                self.detection_module[camera["index"]].run()
            if camera["replay_activate"]:
                replay_thread = threading.Thread(target=self._replay_thread, args=(camera["index"],))
                replay_thread.start()
            if camera["tracking_activate"]:
                self.tracking_camera_list.append(camera["index"])
            # if camera["baseball_tracking_activate"]:
            #     baseball_tracking_thread = threading.Thread(target=self._baseball_tracking_thread, args=(camera["index"],))
            #     baseball_tracking_thread.start()

        # pause_start_thread = threading.Thread(target=self._pause_start_thread, args=(self.tracking_camera_list,))
        # pause_start_thread.start()

        # tracking_thread = threading.Thread(target=self._tracking_thread, args=(self.tracking_camera_list,))
        # tracking_sendToWeb_thread = threading.Thread(target=self._tracking_sendToWeb_thread, args=(self.tracking_camera_list,))
        # tracking_thread.start()
        # tracking_sendToWeb_thread.start()


    def _pause_start_thread(self, tracking_camera_list):
        count = 0
        external_state = 'start'
        while self.tracking_is_running:
            count += 1
            detect_result = dict()
            camera_detect = dict()
            for id in tracking_camera_list:
                detect_result[id] = self.detection_result[id].get()
            
            for item in detect_result:
                camera_detect[item] = [detect_result[item]['path']]
                camera_detect[item].extend(detect_result[item]['result'])
            if count != 1:
                external_state = 'auto'
            self.now_state, self.start_state = pause_start_fusion(camera_detect, external_state= external_state, start_use_hb=False)
            
            # for camera in self.parameters:
            #     if camera["replay_activate"]:
            #         detect_result[camera["index"]]['now_state'] = now_state
            #         detect_result[camera["index"]]['start_state'] = start_state
            #         self.detection_result_toreplay[camera["index"]].put(detect_result[camera["index"]])
            #     if camera["tracking_activate"]:
            #         detect_result[camera["index"]]['now_state'] = now_state
            #         detect_result[camera["index"]]['start_state'] = start_state
            #         self.detection_result_toTrack[camera["index"]].put(detect_result[camera["index"]])
            # print('self.detection_result_toreplay:', self.detection_result_toreplay)
            # print('self.detection_result_toTrack:', self.detection_result_toTrack)

    def _replay_thread(self, camera_id):
        print("initialize replayer :", camera_id, self.replay_parameters[camera_id])
        if camera_id not in self.replay_parameters:
            print("key error : camera index not in replay parameters")
            return        
        replayer = replay_controller(self.replay_parameters[camera_id], False)
        
        while self.tracking_is_running:
            # print("================replay trying to retrive image from queue ========")
            result = self.detection_result_toreplay[camera_id].get()

            frame_info = [result["image"], result["path"]]
            exportedData = [frame_info, result["result"], self.now_state]
            try:
                # verbose 0: no log, verbose 2: show log
                result = replayer.receive_frame_info(0, exportedData, verbose = 2) 
                # result = ['1',image file name] or [False,None]    
            except Exception as e:
                print(e)
                
            if result[0]:
                print("==== got replay signal =====")
                path = ""
                for camera in self.parameters:
                    if camera["index"] == str(camera_id):
                        path = camera["replay_root"]
                if self.redis_upload_enabled:
                    print("==== new clip =====")
                    self.new_clip(result[1],path)

    def new_clip(self, filename, mount_root):
        data_to_send = dict()
        data_to_send["message"] = "new_clip"
        data_to_send["datas"] = dict()
        data_to_send["datas"]["filename"] = filename
        data_to_send["datas"]["game"] = 0
        data_to_send["datas"]["mount_path"] = mount_root
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except socket.error as e:
            print("[ERROR] ", e)

        try:
            # sock.connect(('169.254.100.98', 8940))
            sock.connect(('127.0.0.1', 8940))
        except socket.error as e:
            print("[ERROR] ", e)
        message = json.dumps(data_to_send)
        sock.send(message.encode('utf-8'))
        receive_message = sock.recv(1024).decode('utf-8')
        receive_dict = json.loads(receive_message)
        print("repaly controller reply :", receive_dict)
        sock.close()
    
    def _baseball_tracking_thread(self, camera_id):
        tracking_module = PitecedBallTrackingApp(self.raw_queues[camera_id])

    def _tracking_thread(self, tracking_camera_list):
        # tracking_module = Camera_System(tracking_camera_list)
        PS = ST.player_status()
        detect_image = dict()
        result = None
        while self.tracking_is_running:
            # try:
                # if self.tracking_reset:
                #     self.tracking_reset = False
                #     print("tracking reset")
                #     tracking_module.set_reset()

                for item in tracking_camera_list:
                    detect_image[item] = self.detection_result_toTrack[item].get()
                # print('tracking detect_image:', detect_image)
                # game_no = 104 # need to call hugo's api 
                # game_reuturn_data = self._call_api('https://osense.azurewebsites.net/taoyuanbs/app/querybattlecontrol', data = {"game_no":game_no})
                
                servitor_start = [self.game_information['pause_start'], 
                    [bool(self.game_information['Atk_1H']), bool(self.game_information['Atk_2H']), bool(self.game_information['Atk_3H'])]] # by hugo's backend
                print('servitor_start:', servitor_start)
                is_start = PS.set_pause(self.start_state, self.now_state, servitor_start)
                
                if is_start:
                    PS.init_position(detect_image['128']['image'], detect_image['130']['image'],detect_image['128']['result'] , detect_image['130']['result'] )

                    result = PS.get_result_x_y_pause()
                    '''
                    the result form is {'P':[300,300,False], ...}
                    ['P','C','H','1B','2B','3B','SS','LF','CF','RF','1H','2H','3H']
                    '''
                
                    for key in self.game_information:
                        try:
                            result[key.split('_')[1]].append(self.game_information[key])
                        except:
                            # print('tracking key error:', key)
                            continue
                else:
                    result = {'pause_start':'false'}
                # result = tracking_module.execute(detect_image, self.game_information, self.now_state) # wait to modify after fin's module
                # result = tracking_module.execute(detect_image["image"], detect_image["result"], detect_image2["image"], detect_image2["result"], 
                #     detect_image["pause_start"], game_reuturn_data)
                self.tracking_result_lock.acquire()
                self.tracking_result = result
                self.tracking_result_lock.release()
                # print('track_result:', self.tracking_result[camera_id][0])

            # except:
            #     print("tracking except continue!!!!")

    def _tracking_sendToWeb_thread(self, tracking_camera_list):
        serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        serversocket.bind(("127.0.0.1", 8062))
        serversocket.setblocking(False) # set socket to non-blocking
        serversocket.listen(5)
        while self.tracking_is_running:
            client_list = []
            try:
                csock, addr = serversocket.accept()
                print('Connected by ', addr)
                csock.settimeout(5)
                client_list.append(csock)
            except BlockingIOError:
                pass
            
            for client in client_list:
                if self.tracking_result:
                    # print('=====sent tracking_result to socket=====')
                    csock.send(json.dumps(self.tracking_result).encode('utf-8'))

                    client.close()
                    client_list.remove(client)
        serversocket.close()

        # Users = set()
        # async def counter(websocket,path):
        #     print('====in counter=====', websocket)
        #     Users.add(websocket)
        #     print(Users)
    
        #     while self.tracking_is_running:
        #         # for user in Users:
        #         # if self.tracking_result[camera_id]:
        #         await websocket.send(self.tracking_result[camera_id][1])
        #     print('====sent track over======')
        # self.loop = asyncio.new_event_loop()
        # asyncio.set_event_loop(self.loop)
        # self.loop = asyncio.get_event_loop()
        # # start_server = websockets.serve(plot_track, "127.0.0.1", 8062,reuse_port=True)
        # start_server = websockets.serve(counter, "127.0.0.1", 8062) # python 3.6: no reuse_port
        # self.loop.run_until_complete(start_server)
        # self.loop.run_forever()
        print('=====tracking sent to web close======')

    def stop(self):
        self.tracking_is_running = False

        # self.loop.call_soon_threadsafe(self.loop.stop)
        # time.sleep(1)
        # self.loop.close()
        for k, v in self.detection_module.items():
            v.stop()
        
    
    def enable_redis(self):
        self.redis_upload_enabled = True

    def disable_redis(self):
        self.redis_upload_enabled = False

    def reset_tracking(self):
        self.tracking_reset = True

    def _call_api(self, url, data=None):
        if data:
            r = requests.post(url, data = data)
        else:
            r = requests.post(url)
        result = json.loads(r.text)
        # print(r.text)
        return result