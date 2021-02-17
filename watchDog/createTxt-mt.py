import time
import threading
import queue

class MyThread(threading.Thread):
  def __init__(self,queue, num, lock):
    threading.Thread.__init__(self)
    self.queue = queue
    self.num = num
    self.lock = lock

  def run(self):
    while self.queue.qsize() > 0:
      n = self.queue.get()
      # self.lock.acquire()
      f = open("./data/file{n}.txt".format(n=n), "w")
      print("Thread", self.num, ": ", n)
      # self.lock.release()
      f.close()


      time.sleep(0.002)

start = time.time()

my_queue = queue.Queue()
for i in range(50000):
  my_queue.put(i)

# 建立 5 個子執行緒
threads = []
lock = threading.Lock()
for i in range(1):
  threads.append(MyThread(my_queue,i,lock))
  threads[i].start()    

# 等待所有子執行緒結束
for i in range(1):
  threads[i].join()

print("Done.")

end = time.time()
print(end-start)