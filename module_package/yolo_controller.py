import os, threading, timeit
from .yolov3_package import yolov3_predict_gpus_cv as YOLO
import cv2

class yolo_controller:
    def __init__(self, gpu_id, mount_path, image_queue, result_queue, replay_queue = None):
        self.mount_path = mount_path
        self.result_queue = result_queue
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
            # image = cv2.imread(os.path.join(self.mount_path, image_path))
            image = cv2.imread(image_path)
            try:
                yolo_result = YOLO.detect_img(self.yolov3, image.copy())
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
            self.result_queue.put(result)
            if self.replay_queue != None:
                self.replay_queue.put(result)
            # print("put yolo result into queue")

    def stop(self):
        self.thread_is_running = False