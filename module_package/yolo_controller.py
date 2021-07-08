import os, threading, timeit
from .yolov3_package import yolov3_predict_gpus_cv as YOLO
import cv2
import time

class yolo_controller:
    def __init__(self, gpu_id, mount_path, image_queue, result_queue, track_queue, replay_queue):
        self.mount_path = mount_path
        self.result_queue = result_queue
        self.track_queue = track_queue
        self.replay_queue = replay_queue
        self.image_queue = image_queue
        # self.yolov3 = YOLO.initial_model('yolov3', gpu_id)    # chris try no gpu_id 
        self.yolov3 = YOLO.initial_model('yolov3')
        self.thread_is_running = False

    def run(self):
        self.thread_is_running = True
        new_thread = threading.Thread(target=self._predict_thread)
        new_thread.start()

    def _predict_thread(self):
        while self.thread_is_running:
            # print("get image from queue")
            image_path = self.image_queue.get()
            # print('image_path:', image_path)
            # image = cv2.imread(os.path.join(self.mount_path, image_path))
            image = cv2.imread(image_path)
            try:
                start = time.time()
                yolo_result = YOLO.detect_img(self.yolov3, image.copy())
                print('yolo detect time:', time.time() - start)
                # print('yolo detect_img result: ', yolo_result)
            except:
                print('fail in detect')
                continue
            if yolo_result == None:
                print('yolo_result == None')
                continue
            result = dict()
            result["path"] = image_path
            result["image"] = image
            result["result"] = yolo_result
            # result["pause_start"] = True  # wait Lily's pause_start
            self.result_queue.put(result)
            if self.replay_queue != None:
                self.replay_queue.put(result)
            if self.track_queue != None:
                self.track_queue.put(result)

    def stop(self):
        self.thread_is_running = False