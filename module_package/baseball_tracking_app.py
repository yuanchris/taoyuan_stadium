# from numpy.core.shape_base import block
# from numpy.lib.function_base import select
# from core_v4 import Files_Loader,Baseball_Tracking
import threading


class PitecedBallTrackingApp:
    def __init__(self,fileNameQueue,doDebug=False):
        self.doDebug = doDebug
        self.fileNameQueue = fileNameQueue
        # self.BT = Baseball_Tracking(debugMode=self.doDebug)
        self.start()
      
    def threading_controller(self):
        detection_handler = threading.Thread(target=self.detection_thread, args=())
        detection_handler.start()

        # generation_handler = threading.Thread(target=self.BT.generation_thread, args=())
        # generation_handler.start()
        
        # detection_handler.join()
        # generation_handler.join()

    def start(self):
        self.threading_controller()

    def detection_thread(self):
        count = 0
        while True:
            count+=1
            print ('tracking_ball >> current_size::',len(self.fileNameQueue.queue))
            imgPath = self.fileNameQueue.get(block=True)
            # f_time = time.time()
            # imgInfo = self.FL.convert_raw_to_ImgArray(imgPath)
            # print ('{} - <convert: {}>'.format(str(count),time.time()-f_time))
            # self.BT.receive(imgInfo)
            print ('tracking_ball_info::',imgPath)
            print ('\n')
            
if __name__ == "__main__":
    imgNames_queue = None
    app = PitecedBallTrackingApp(imgNames_queue)

