import myqueue, threading, os, timeit, time, requests
import socket, json
import websockets
import asyncio

from .yolo_controller import yolo_controller
from .highlight_replay.hightlight_replay_gpu import Highlight_Repleay as replay_controller
# from .tracking_board.camera_system import Camera_System
from .tracking_board.track_test import Camera_System
from .baseball_tracking_app import PitecedBallTrackingApp

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
        self.detection_result_toreplay = dict()
        self.tracking_result = dict()
        self.tracking_result_lock = threading.Lock()
        self.tracking_reset = False
        # self.loop = None

        for camera in parameters:

            if camera["tracking_activate"]:
                self.tracking_result[camera["index"]] = list()
            #     self.tracking_module[camera["index"]] = Camera_System(int(camera["index"]), camera["gpu_id"], self.image_queues[camera["index"]])
            if camera["replay_activate"]:
                new_queue = myqueue.Queue(1)
                self.detection_result_toreplay[camera["index"]] = new_queue
            if camera["detection_activate"]:
                new_queue = myqueue.Queue(1)
                self.detection_result[camera["index"]] = new_queue
                if camera["replay_activate"]:
                    detection_module = yolo_controller(camera["gpu_id"], camera["mount_root"], self.image_queues[camera["index"]], self.detection_result[camera["index"]], self.detection_result_toreplay[camera["index"]])
                else:
                    detection_module = yolo_controller(camera["gpu_id"], camera["mount_root"], self.image_queues[camera["index"]], self.detection_result[camera["index"]])
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
                tracking_thread = threading.Thread(target=self._tracking_thread, args=(camera["index"],))
                tracking_sendToWeb_thread = threading.Thread(target=self._tracking_sendToWeb_thread, args=(camera["index"],))
                tracking_thread.start()
                tracking_sendToWeb_thread.start()
            # if camera["baseball_tracking_activate"]:
            #     baseball_tracking_thread = threading.Thread(target=self._baseball_tracking_thread, args=(camera["index"],))
            #     baseball_tracking_thread.start()

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
            exportedData = [frame_info, result["result"]]
            print('replay running')
            try:
                # result = ['1',image file name] or [False,None]
                if int(camera_id) == 2:
                    # verbose 0: no log, verbose 2: show log
                    result = replayer.receive_frame_info(0, exportedData, verbose = 2) 
                else:
                    result = replayer.receive_frame_info(0, exportedData, verbose = 2)
            except:
                continue
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

    def _tracking_thread(self, camera_id):
        tracking_module = Camera_System(camera_id)
        while self.tracking_is_running:
            try:
                if self.tracking_reset:
                    self.tracking_reset = False
                    print("tracking reset")
                    tracking_module.set_reset()

                detect_image = self.detection_result[camera_id].get()
                # detect_image2 = self.detection_result["3"].get()  # another camera for tracking board

                game_no = 104 # need to call hugo's api 
                game_reuturn_data = self._call_api('https://osense.azurewebsites.net/taoyuanbs/app/querybattlecontrol', data = {"game_no":game_no})

                result = tracking_module.execute(detect_image["result"], game_reuturn_data)
                # result = tracking_module.execute(detect_image["image"], detect_image["result"], detect_image2["image"], detect_image2["result"], 
                #     detect_image["pause_start"], game_reuturn_data)

                self.tracking_result_lock.acquire()
                self.tracking_result[camera_id] = result
                self.tracking_result_lock.release()
                # print('track_result:', self.tracking_result[camera_id][0])

            except:
                print("tracking except continue!!!!")
                # continue

    def _tracking_sendToWeb_thread(self, camera_id):
        print('===== tracking sent to web start=====')

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
                if self.tracking_result[camera_id]:
                    # print('=====sent tracking_result to socket=====')
                    # csock.send(self.tracking_result[camera_id][1])
                    csock.send(json.dumps(self.tracking_result[camera_id][0]).encode('utf-8'))

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