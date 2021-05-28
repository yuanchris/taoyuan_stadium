import threading, os, timeit, time, requests
import socket, json
data =  {"Code":"E000","Message":"查詢成功!",
    "GameNo":104,"CurrentInning":0,"GameState":1, 'pause_start': 'true', 
    "CurrentBat":"11 林晨樺","Run1B":"12 江國豪","Run2B":"12 江國豪","Run3B":"12 江國豪",
    "Def_P":"00 蘇俊璋","Def_C":"15 劉昱言","Def_1B":"15 Finn","Def_2B":"11 林泓育",
    "Def_3B":"15 Chris","Def_SS":"17 Claire","Def_LF":"35 成晉","Def_CF":"17 Matt",
    "Def_RF":"17 Lily"}

print(data)
r = requests.post('http://127.0.0.1:5000/hugo_game_information', data = data)
print(r.status_code)
result = json.loads(r.text) 
print(result)