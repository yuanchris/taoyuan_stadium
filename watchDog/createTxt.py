import time
import threading
import queue, random
import os,sys


start = time.time()
n = 0
string1 = os.urandom(388*1024)
while n < 8000*60*3:
  print(n)
  with open("./8902/file{n}.txt".format(n=n), "wb") as f:
    pass
    # f.write(string1)

  n += 1
  time.sleep(1)

print("Done.")

end = time.time()
print(end-start)