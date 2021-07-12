Strategy_board
============

Note 

> main.py and field2.png are useless, it is just for testing for me.


Using
------------

Declare
~~~python
import status as ST
PS     = ST.player_status()
~~~

Set input

~~~python
is_start = set_pause(HCJ, system_start, servitor_start)
'''
    HCJ is bool which means hit and catcher and judge is ready or not
    system_start is bool from yolo
    servitor_start is someone operating the panel and needs [is_start_or_not, [1h,2h,3h are avaliable?]]
'''
~~~

Run script

~~~python
if is_start:
  PS.init_position(west_img, east_img,west_roi , east_roi)
'''
   img is np array
   roi is result of yolo
'''
~~~

Get result

~~~python
if is_start:
  result = get_result_x_y_pause()
'''
   the result form is {'P':[300,300,False], ...}
   ['P','C','H','1B','2B','3B','SS','LF','CF','RF','1H','2H','3H']
'''
~~~
