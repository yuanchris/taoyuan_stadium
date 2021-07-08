import threading, os, timeit, time, requests
import socket, json
data =  {"Code":"E000","Message":"查詢成功!",
            "GameNo":104,"CurrentInning":0,"GameState":1, 'pause_start': True, 
            "Atk_H":"14 林智勝","Atk_1H":"11 林晨樺","Atk_2H":"","Atk_3H":"",
            "Def_P":"01 蘇俊璋","Def_C":"15 劉昱言","Def_1B":"12 蔡豐安","Def_2B":"30 林泓育",
            "Def_3B":"15 陳志遠","Def_SS":"17 Claire","Def_LF":"35 成晉","Def_CF":"18 Matt",
            "Def_RF":"19 Lily"}


print(data)
r = requests.post('http://127.0.0.1:5000/hugo_game_information', data = data)
print(r.status_code)
result = json.loads(r.text) 
print(result)