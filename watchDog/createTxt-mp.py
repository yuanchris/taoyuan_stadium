import time
import threading
import multiprocessing

n = 0
# while True:
#   f = open(f"./data/file{n}.txt", "w")
#   # f.write("Now the file has more content!")
#   f.close()
#   # print(f"file{n}.txt created")
#   n += 1
#   # time.sleep(1)

class My(multiprocessing.Process):
  def __init__(self, num):
    multiprocessing.Process.__init__(self)
    global n
    self.n = n
    self.num = num

  def run(self):
    while self.n < 10000:
      f = open(f"./data/file{self.n}.txt", "w")
      print("Thread", self.num, ": ", self.n)
      f.close()
      self.n += 1
      # time.sleep(1)

start = time.time()

# 建立 5 個子執行緒
process = []
for i in range(5):
  process.append(My(i))
  process[i].start()    

# 等待所有子執行緒結束
for i in range(5):
  process[i].join()

print("Done.")

end = time.time()
print(end-start)