import socket, sys, json, redis
import threading, time

from api_example import api_retriever
from module_package.tracking_controller import tracking_controller
import myqueue
import parameters
class camera_controller:
    def __init__(self):
        self.controlThread_is_running = False
        self.getImages_is_running = False
        self.updatingPosition_is_running = False
        self.redis_upload_enabled = False
        self.redis_database = redis.StrictRedis(host='osense.redis.cache.windows.net', port=6380, db=0,
                                                password='9PKSoAHrzMNb4fBIH4Inah81XUj9eh2UUGWuJuUibFM=', ssl=True)
        self.server_socket = None
        self.control_socket = None
        self.control_socket_list = []

        self.image_queues = dict()
        self.raw_queues = dict()
        for camera in parameters.startup_parameters:
            image_queue = myqueue.Queue(1) # only the newest photo
            self.image_queues[camera["index"]] = image_queue 
            raw_queue = myqueue.Queue(10)
            self.raw_queues[camera["index"]] = raw_queue 

        self.tracking_controller = tracking_controller(self.image_queues,self.raw_queues, 
            parameters.startup_parameters, parameters.replay_parameters, self.redis_upload_enabled)
        self.api = api_retriever()
        self.api.retrieve_game_id()
        self.defend_players, self.attack_players = self.api.retrieve_player_information()



    def run(self):
        print("===============Server Started=================")
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)  # reuse tcp
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # if reuse port is not ok
        self.server_socket.bind(('127.0.0.1', 8050))
        self.server_socket.listen(5)
        print("[*] Server Listening on %s:%d " % ('127.0.0.1', 8050))

        while True:
            csock, addr = self.server_socket.accept()
            csock.settimeout(5)
            print('Connected by ', addr)
            try:
                msg = csock.recv(1024)
                receive_dict = msg.decode("utf-8")
                receive_dict = json.loads(receive_dict)
                if receive_dict["message"] == 'start':
                    if self.controlThread_is_running:
                        print('control thread is already started')
                        data_to_send = dict()
                        data_to_send["success"] = True
                        csock.send(json.dumps(data_to_send).encode('utf-8'))
                    else:
                        print('Start control thread')
                        self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        # self.control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)  # reuse tcp
                        self.control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # beacuse py3.6 cannot use reuseport
                        self.control_socket.bind(('127.0.0.1', 8051))
                        self.control_socket.setblocking(False) # set socket to non-blocking
                        self.control_socket.listen(5)
                        print("[*] control thread Listening on %s:%d " % ('127.0.0.1', 8051))

                        self.controlThread_is_running = True
                        data_to_send = dict()
                        data_to_send["success"] = True
                        csock.send(json.dumps(data_to_send).encode('utf-8'))
                        control_thread = threading.Thread(target=self.controlThread)
                        control_thread.start()

                        # get images and position updating
                        self.getImages_is_running = True
                        self.updatingPosition_is_running = True
                        self.start_getJpg_thread()
                        self.start_getRaw_thread()
                        self.start_updatingPosition_thread()
                        self.tracking_controller.run()

                elif receive_dict["message"] == 'stop':
                    if self.controlThread_is_running == False:
                        print('control thread is already stopped')
                        data_to_send = dict()
                        data_to_send["success"] = True
                        csock.send(json.dumps(data_to_send).encode('utf-8'))
                    else:
                        print('Stop control thread')
                        self.controlThread_is_running = False
                        self.getImages_is_running = False
                        self.updatingPosition_is_running = False
                        self.erase_redis()
                        self.tracking_controller.stop()
                        self.control_socket.close()
                        data_to_send = dict()
                        data_to_send["success"] = True
                        csock.send(json.dumps(data_to_send).encode('utf-8'))
                else:
                    data_to_send = dict()
                    data_to_send["success"] = False
                    csock.send("fail".encode('utf-8'))
                
            except ConnectionResetError as e:
                print("#main_system socket encounter error :", e)

            csock.close()

    def start_getJpg_thread(self):
        for camera in parameters.startup_parameters:
            if camera["detection_activate"] or camera["tracking_activate"] or camera["replay_activate"]:
                getImages = threading.Thread(target=self._getJpg_thread, args=(camera["index"], camera["file_port"], camera["mount_root"],))
                getImages.start()

    def _getJpg_thread(self, index, port, mount_root):
        print(f"========== camera {index} start getting jpgs ==========")
        serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        serversocket.bind(("127.0.0.1", port))
        serversocket.setblocking(False) # set socket to non-blocking
        serversocket.listen(5)
        while self.getImages_is_running:
            try:
                csock, addr = serversocket.accept()
                msg = csock.recv(1024).decode("utf-8")
                # print('Connected by ', addr)
                filename = msg.split("/")[-1]
                print(f'camera {index} get {filename}')
                self.image_queues[index].put(mount_root + filename)
                csock.close()
                # print(self.image_queues[index].get())
            except BlockingIOError:
                pass
        serversocket.close()

    def start_getRaw_thread(self):
        for camera in parameters.startup_parameters:
            if camera["baseball_tracking_activate"]:
                getRaw = threading.Thread(target=self._getRaw_thread, args=(camera["index"], camera["file_port"], camera["mount_root"],))
                getRaw.start()

    def _getRaw_thread(self, index, port, mount_root):
        print(f"========== camera {index} start getting raws ==========")
        serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        serversocket.bind(("127.0.0.1", port))
        serversocket.setblocking(False) # set socket to non-blocking
        serversocket.listen(5)
        while self.getImages_is_running:
            try:
                csock, addr = serversocket.accept()
                msg = csock.recv(1024).decode("utf-8")
                # print('Connected by ', addr)
                filename = msg.split("/")[-1]
                print(f'camera {index} get {filename}')
                self.raw_queues[index].put(mount_root + filename)
                csock.close()
                # print(self.image_queues[index].get())
            except BlockingIOError:
                pass
        serversocket.close()



    def start_updatingPosition_thread(self):
        for camera in parameters.startup_parameters:
            if camera["tracking_activate"]:
                position_thread = threading.Thread(target=self._updatingPosition_thread, args=(camera["index"], camera["mount_root"],))
                position_thread.start()

    def _updatingPosition_thread(self, index, mount_root):
        print(f"========== camera {index} start updating position ==========")
        while self.updatingPosition_is_running:
            pass
            # TODO


    def controlThread(self):
        while self.controlThread_is_running:
            try: 
                csock, addr = self.control_socket.accept()
                csock.settimeout(5)
                self.control_socket_list.append(csock)
                print('Connected by ', addr)
            except BlockingIOError:
                pass
            
            for client in self.control_socket_list:
                try:    
                    msg = client.recv(1024)
                    receive_dict = msg.decode("utf-8")
                    receive_dict = json.loads(receive_dict)
                    if receive_dict["message"] == 'player_positions':
                        data_to_send = dict()
                        data_to_send["success"] = True
                        # self.sockettrackerListMutex.acquire()
                        # result = self.tracking_controller.get_player_position()  # steven total result here
                        # self.sockettrackerListMutex.release()
                        # data_to_send["datas"] = result
                        client.send(json.dumps(data_to_send).encode('utf-8'))
                    elif receive_dict["message"] == 'player_list':
                        data_to_send = dict()
                        data_to_send["success"] = True
                        result = {'defend': self.defend_players, 'attack': self.attack_players}
                        data_to_send["datas"] = result
                        client.send(json.dumps(data_to_send).encode('utf-8'))
                    elif receive_dict["message"] == "redis_start":
                        print('#main_system receive redis start')  
                        self.redis_upload_enabled = True
                        self.tracking_controller.enable_redis()
                        data_to_send = dict()
                        data_to_send["success"] = True
                        client.send(json.dumps(data_to_send).encode('utf-8'))
                    elif receive_dict["message"] == "redis_stop":
                        print('#main_system receive redis stop')
                        self.redis_upload_enabled = False
                        self.tracking_controller.disable_redis()
                        time.sleep(0.2)
                        self.erase_redis()
                        data_to_send = dict()
                        data_to_send["success"] = True
                        client.send(json.dumps(data_to_send).encode('utf-8'))
                    else:
                        data_to_send = dict()
                        data_to_send["success"] = False
                        client.send(json.dumps(data_to_send).encode('utf-8'))
                except ConnectionResetError as e:
                    print("#main_system socket encounter error :", e)
                except:
                    print("other error occurs")
                client.close()
                self.control_socket_list.remove(client)
        self.control_socket.close()
        print('close socket')

    def erase_redis(self):
        pass
        # TODO
        # for key in self.redis_database.keys():
        #     self.redis_database.delete(key)    

if __name__ == '__main__':
    controller = camera_controller()
    controller.run()
