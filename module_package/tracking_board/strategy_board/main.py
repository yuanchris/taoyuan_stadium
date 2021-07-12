import status as ST
import cv2, os
import numpy as np
from constant import cfg
import json
# step 0 , initialise , system status as SS  , player status as PS , location tools as LT
PS     = ST.player_status()
WEST   = 1
EAST   = 0
FILE_NAME_LEN   =   29

def get_sec(name):
    return  float(name[9:11])*3600+float(name[12:14])*60 + float(name[15:24].replace('_','.'))

def draw_img_from_result(result_list,img):
    # print(result_list)
    for i in result_list:
        cv2.circle(img,tuple(i[:2]), 5, (0, 255, 255), -1)
    return img
def rank(aa,bb,name,img):
    aa = json.loads(aa)
    res = []
    for a, b in zip(aa.items(),bb.items()):
        if abs(a[1][0] - b[1][0]) > 2 or abs(a[1][1] - b[1][1]) > 2:
            res.append([a[0],a[1][:2],b[1][:2]])
    if len(res) >0:
        # print(res)
        for i in res:
            color = (0,255,0)
            cv2.circle(img,tuple(i[1][:2]), 5, color, -1)
            cv2.putText(img, str(i[0]), tuple(i[1][:2]), cv2.FONT_HERSHEY_SIMPLEX,0.7,color, 1, cv2.LINE_AA)
            cv2.imwrite('image/{}'.format(name),img)
        return False
    return True
def draw_img_from_status(name,img,rr,west_name):
    # result_list = PS.players
    result_list = PS.get_result_x_y()
    res = json.dumps(result_list)
    f = open('res.txt','a')
    f.write(name+'\t'+res+'\n')
    f.close()
    # if PS.players['P'][0] - PS.players['P'][1] <PS.players['C'][0] - PS.players['C'][1]:
    #     print(west_name,'P<C',PS.players['P'][0])
    for key , v in result_list.items():
        i = cfg.always_in_ground.index(key)
        if not PS.visible[i]:
            continue
        # if key =='1H' or key =='2B':
        #     print(key,v)
        if 'H' in key :
            # print(v)
            color = (0,0,255)
            # print(key,v)
            # if v[0] > 130 or v[1] < 270 and(not v[:2] == [-1,-1]):
            #     print(west_name,key,v)
            # if v[0] > 23 and v[0] < 104 and v[1]>297 and v[1] < 378: #375,101  26,300
            #     print(west_name,key,v)
            # if key =='1H':
            #     if v[1] > 390:
            #         print(west_name,key,v)
        else:
            
            color = (255, 0 ,0)
        cv2.circle(img,tuple(v[:2]), 5, color, -1)
        cv2.putText(img, str(key), tuple(v[:2]), cv2.FONT_HERSHEY_SIMPLEX,0.7,color, 1, cv2.LINE_AA)
        # cv2.putText(img, str(key)+','+str(round(v[2],2)), tuple(v[:2]), cv2.FONT_HERSHEY_SIMPLEX,0.7,color, 1, cv2.LINE_AA)
        cv2.putText(img, west_name, (410,40), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2, cv2.LINE_AA)

    # if not rank(rr, result_list,name,img.copy()):
    #     print(name)
    #     pass
    return img

def handle_puase_and_start(west,east):
    name = west if '0418' in west else east
    f = open('./pause_{}.txt'.format(name[:8]))
    pause_time = f.readlines()
    pause_time = [ x.replace('\n','') for x in pause_time ]
    f.close()
    f = open('./start_{}.txt'.format(name[:8]))
    start_time = f.readlines()
    start_time = [ x.replace('\n','').split(' ') for x in start_time ]
    f.close()
    for i in pause_time:
        if i in name:
            PS.SS.is_pause = True
            PS.SS.set_base(False,False,False)
    for i in start_time:
        if i[0] in name:
            # print(name)
            PS.SS.is_pause = False
            PS.LT.is_init = False
            b1,b2,b3 = False,False,False
            base = i[1]
            if '1' in base:
                b1 = True
            else:
                PS.players['1H'][:2] = [-1,-1]
            if '2' in base:
                b2 = True
            else:
                PS.players['2H'][:2] = [-1,-1]
            if '3' in base:
                b3 = True
            else:
                PS.players['3H'][:2] = [-1,-1]
            PS.SS.set_base(b1,b2,b3)
            return True
    return False

if not os.path.exists('image/'):
    os.makedirs('image/')

# txt_path = '/media/osense/4a89f8e5-39c7-4714-9bbd-74f3fe1541f9/lucid_image/yolov3_package_v2/'
date = '0417'
txt_path = './'
f = open(txt_path + 't28_{}.txt'.format(date))
west = f.readlines()
west = [ w.replace('\n','') for w in west]
f.close()
f = open(txt_path + 't30_{}.txt'.format(date))
east = f.readlines()
east = [ w.replace('\n','') for w in east]
f.close 
count = 0
root_path = '/media/osense/4a89f8e5-39c7-4714-9bbd-74f3fe1541f9/lucid_image/jpg/2021_{}_{}/'.format(date[:2],date[2:])
img_path = [root_path + '30/',root_path + '28/']
out = cv2.VideoWriter('./result_0417.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 5, (1158+157,400+92))
field_jpg = cv2.imread('./result.png')
skip = 0
if os.path.exists('res.txt'):
    os.remove('res.txt')
f = open('res_'+date+'.txt')
res = f.readlines()
f.close()
res = [ w.replace('\n','') for w in res]
res = [w.split('\t')[1] for w in res ]
res_count =0
rr = []
more = ''
for i in east:
    west_roi = west[count]
    east_roi = i
    east_name = east_roi[:FILE_NAME_LEN -1]
    west_name = west_roi[:FILE_NAME_LEN -1]
    if -get_sec(east_name) + get_sec(west_name) > 0.2:
        continue
    elif get_sec(east_name) - get_sec(west_name) > 0.2:
        count+=1
    count += 1
    if not os.path.exists('imgs_{}{}'.format(east_name[4:11],more)):
        os.makedirs('imgs_{}{}'.format(east_name[4:11],more))


    if i < '20210417_18-51-50':
        continue
    rr = res[res_count]
    res_count+=1

    # if i <'20210417_19-49-17':
    #     continue
    # if i > '20210418_14-06-29':
    #     break
    print(east_name , west_name)


    flag = os.path.exists('imgs_{}{}/'.format(east_name[4:11],more)+east_name)
    img = field_jpg.copy()
    cv2.putText(img, east_name, (10,40), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2, cv2.LINE_AA)
    if not flag:
        east_img = cv2.imread(img_path[EAST] + east_name)
    else:
        east_img = cv2.imread('imgs_{}{}/'.format(east_name[4:11],more)+east_name)
    if skip > 0:
        skip -= 1
        tmp = cv2.resize(east_img , (758,400+92))
        if not flag: cv2.imwrite('imgs_{}{}/'.format(east_name[4:11],more)+east_name,tmp)
        img = np.hstack((img,tmp))
        img = draw_img_from_status (east_name,img,rr,west_name)
        out.write(  img  )
        continue
    if handle_puase_and_start(west_name,east_name):
        skip += 25
        tmp = cv2.resize(east_img , (758,400+92))
        if not flag: cv2.imwrite('imgs_{}{}/'.format(east_name[4:11],more)+east_name,tmp)
        img = np.hstack((img,tmp))
        img = draw_img_from_status (east_name,img,rr,west_name)
        cv2.putText(img, "PAUSING", (150,200), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2, cv2.LINE_AA)
        out.write(  img  )
        continue
    if PS.SS.is_pause :
        PS.reset()
        tmp = cv2.resize(east_img , (758,400+92))
        if not flag: cv2.imwrite('imgs_{}{}/'.format(east_name[4:11],more)+east_name,tmp)
        img = np.hstack((img,tmp))
        img = draw_img_from_status (east_name,img,rr,west_name)
        cv2.putText(img, "PAUSING", (150,200), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2, cv2.LINE_AA)
        out.write(  img  )
        continue
    west_img = []
    if not flag: west_img = cv2.imread(img_path[WEST] + west_name)

    merge_poi = PS.init_position(west_img, east_img,west_roi , east_roi,west_name, east_name,flag)
    # a,b = PS.init_position(west_img, east_img,west_roi , east_roi)
    # PS.LT.is_init = True
    tmp = cv2.resize(east_img , (758,400+92))
    if not flag: cv2.imwrite('imgs_{}{}/'.format(east_name[4:11],more)+east_name,tmp)
    img = np.hstack((img,tmp))
    # img = draw_img_from_result(merge_poi,img )
    img = draw_img_from_status (east_name,img,rr,west_name)

    # cv2.putText(img, east_name, (10,40), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2, cv2.LINE_AA)
    out.write(  img  )
    # del img
    # del east_img
    # break

    