startup_parameters = [
    # {
    #     "index"   : "1",
    #     "ip"      : "169.254.100.21",
    #     "port"    : "5588",
    #     "start_message" : "{\"code\":\"resume\"}\0",
    #     "stop_message" : "{\"code\":\"pause\"}\0",
    #     "detection_activate" : False,
    #     "tracking_activate" : False,
    #     "replay_activate" : False,
    #     "file_port" : 8901,
    #     "mount_root" : "/home/chris/Desktop/lucid/chris/gui/imgs/",
    #     "replay_root" : "/home/kma4631452/Desktop/mnts/NAS_01/raw/",
    #     "gpu_id" : 0
    # },
    {
        "index"   : "128",
        "ip"      : "169.254.100.22",
        "port"    : "5588",
        "start_message" : "{\"code\":\"resume\"}\0",
        "stop_message" : "{\"code\":\"pause\"}\0",
        "detection_activate" : True,
        "tracking_activate" : True,
        "replay_activate" : False,
        "baseball_tracking_activate" : False,
        "file_port" : 8902,
        "mount_root" : "/home/chris/Desktop/taoyuan/taoyuan_img/test_210417_19-02-24_19-03-15_128_west/",
        "replay_root" : "/home/chris/Desktop/taoyuan/taoyuan_img/test_210417_19-02-24_19-03-15_128_west/",
        "gpu_id" : 1
    },
    {
        "index"   : "130",
        "ip"      : "169.254.100.23",
        "port"    : "5588",
        "start_message" : "{\"code\":\"resume\"}\0",
        "stop_message" : "{\"code\":\"pause\"}\0",
        "detection_activate" : True,
        "tracking_activate" : True,
        "replay_activate" : False,
        "baseball_tracking_activate" : False,
        "file_port" : 8903,
        "mount_root" : "/home/chris/Desktop/taoyuan/taoyuan_img/test_210417_19-02-24_19-03-15_130_east/",
        "replay_root" : "/home/chris/Desktop/taoyuan/taoyuan_img/test_210417_19-02-24_19-03-15_130_east/",
        "gpu_id" : 2
    },
    # {
    #     "index"   : "4",
    #     "ip"      : "169.254.100.24",
    #     "port"    : "5588",
    #     "start_message" : "{\"code\":\"resume\"}\0",
    #     "stop_message" : "{\"code\":\"pause\"}\0",
    #     "detection_activate" : False,
    #     "tracking_activate" : False,
    #     "replay_activate" : False,
    #     "file_port" : 8904,
    #     "mount_root" : "/home/osense/Desktop/mnts/NAS_04/jpg/",
    #     "replay_root" : "/home/kma4631452/Desktop/mnts/NAS_04/raw/",
    #     "gpu_id" : 3
    # },
    # {
    #     "index"   : "5",
    #     "ip"      : "169.254.100.25",
    #     "port"    : "5588",
    #     "start_message" : "{\"code\":\"resume\"}\0",
    #     "stop_message" : "{\"code\":\"pause\"}\0",
    #     "detection_activate" : False,
    #     "tracking_activate" : False,
    #     "replay_activate" : False,
    #     "file_port" : 8905,
    #     "mount_root" : "/home/osense/Desktop/mnts/NAS_05/jpg/",
    #     "replay_root" : "/home/kma4631452/Desktop/mnts/NAS_05/raw/",
    #     "gpu_id" : 4
    # },
    # {
    #     # teacher 4
    #     "index"   : "6",
    #     "ip"      : "169.254.100.26",
    #     "port"    : "5588",
    #     "start_message" : "{\"code\":\"resume\"}\0",
    #     "stop_message" : "{\"code\":\"pause\"}\0",
    #     "detection_activate" : False,
    #     "tracking_activate" : False,
    #     "replay_activate" : False,
    #     "file_port" : 8906,
    #     "mount_root" : "/home/osense/Desktop/mnts/NAS_06/raw/",
    #     "replay_root" : "/home/kma4631452/Desktop/mnts/NAS_06/raw/",
    #     "gpu_id" : -1
    # },
    # {
    #     "index"   : "7",
    #     "ip"      : "169.254.100.27",
    #     "port"    : "5588",
    #     "start_message" : "{\"code\":\"resume\"}\0",
    #     "stop_message" : "{\"code\":\"pause\"}\0",
    #     "detection_activate" : False,
    #     "tracking_activate" : False,
    #     "replay_activate" : False,
    #     "file_port" : 8907,
    #     "mount_root" : "/home/osense/Desktop/mnts/NAS_07/jpg/",
    #     "replay_root" : "/home/kma4631452/Desktop/mnts/NAS_07/raw/",
    #     "gpu_id" : 5
    # },
    # {
    #     "index"   : "8",
    #     "ip"      : "169.254.100.28",
    #     "port"    : "5588",
    #     "start_message" : "{\"code\":\"resume\"}\0",
    #     "stop_message" : "{\"code\":\"pause\"}\0",
    #     "detection_activate" : False,
    #     "tracking_activate" : False,
    #     "replay_activate" : False,
    #     "file_port" : 8908,
    #     "mount_root" : "/home/osense/Desktop/mnts/NAS_08/jpg/",
    #     "replay_root" : "/home/kma4631452/Desktop/mnts/NAS_08/raw/",
    #     "gpu_id" : 6
    # },
    # {
    #     # teacher 1
    #     "index"   : "9",
    #     "ip"      : "169.254.100.29",
    #     "port"    : "5588",
    #     "start_message" : "{\"code\":\"resume\"}\0",
    #     "stop_message" : "{\"code\":\"pause\"}\0",
    #     "detection_activate" : False,
    #     "tracking_activate" : False,
    #     "replay_activate" : False,
    #     "file_port" : 8909,
    #     "mount_root" : "/home/osense/Desktop/mnts/NAS_09/raw/",
    #     "replay_root" : "/home/kma4631452/Desktop/mnts/NAS_09/raw/",
    #     "gpu_id" : -1
    # },
    # {
    #     # teacher 2
    #     "index"   : "10",
    #     "ip"      : "169.254.100.30",
    #     "port"    : "5588",
    #     "start_message" : "{\"code\":\"resume\"}\0",
    #     "stop_message" : "{\"code\":\"pause\"}\0",
    #     "detection_activate" : False,
    #     "tracking_activate" : False,
    #     "replay_activate" : False,
    #     "file_port" : 8910,
    #     "mount_root" : "/home/osense/Desktop/mnts/NAS_10/raw/",
    #     "replay_root" : "/home/kma4631452/Desktop/mnts/NAS_10/raw/",
    #     "gpu_id" : -1
    # },
    # {
    #     # teacher 3
    #     "index"   : "11",
    #     "ip"      : "169.254.100.31",
    #     "port"    : "5588",
    #     "start_message" : "{\"code\":\"resume\"}\0",
    #     "stop_message" : "{\"code\":\"pause\"}\0",
    #     "detection_activate" : False,
    #     "tracking_activate" : False,
    #     "replay_activate" : False,
    #     "file_port" : 8911,
    #     "mount_root" : "/home/osense/Desktop/mnts/NAS_11/raw/",
    #     "replay_root" : "/home/kma4631452/Desktop/mnts/NAS_11/raw/",
    #     "gpu_id" : -1
    # },
    # {
    #     "index"   : "12",
    #     "ip"      : "169.254.100.32",
    #     "port"    : "5588",
    #     "start_message" : "{\"code\":\"resume\"}\0",
    #     "stop_message" : "{\"code\":\"pause\"}\0",
    #     "detection_activate" : False,
    #     "tracking_activate" : False,
    #     "replay_activate" : False,
    #     "file_port" : 8912,
    #     "mount_root" : "/home/osense/Desktop/mnts/NAS_12/jpg/",
    #     "replay_root" : "/home/kma4631452/Desktop/mnts/NAS_12/raw/",
    #     "gpu_id" : 7
    # },
    # {
    #     # teacher 5
    #     "index"   : "13",
    #     "ip"      : "169.254.100.33",
    #     "port"    : "5588",
    #     "start_message" : "{\"code\":\"resume\"}\0",
    #     "stop_message" : "{\"code\":\"pause\"}\0",
    #     "detection_activate" : False,
    #     "tracking_activate" : False,
    #     "replay_activate" : False,
    #     "file_port" : 8913,
    #     "mount_root" : "/home/osense/Desktop/mnts/NAS_13/raw/",
    #     "replay_root" : "/home/kma4631452/Desktop/mnts/NAS_13/raw/",
    #     "gpu_id" : -1
    # }
]

player_position_dict = {
    "0" : "C",
    "1" : "1B",
    "2" : "2B",
    "3" : "SS",
    "4" : "3B",
    "5" : "RF",
    "6" : "CF",
    "7" : "LF",
    "8" : "P"
}

replay_parameters = {
    "130" : "1B",
    "4" : "2B",
    "3" : "HB",
    "12" : "3B"
}
