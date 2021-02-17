import requests, json
import threading

class api_retriever:
    def __init__(self):
        self.player_information_lock = threading.Lock()
        self.current_hitters_lock = threading.Lock()
        self.current_hitters = list()
        self.player_information = dict()

    def retrieve_game_id(self):
        return_data = self._call_api("https://osense.azurewebsites.net/taichungbs/app/GameTime.php")
        self.game_id = return_data['Games'][-1]['Game_ID']
        # return (return_data['Games'][0]['Game_ID'])

    def get_player_information(self):
        self.player_information_lock.acquire()
        # TODO : return player informations here
        self.player_information_lock.release()

    def retrieve_player_information(self):
        # * api 14 example
        guest_team_player = '[{"UniformNumber":41,"Player_Name":"鄭佳彥","Player_AVG":0,"Player_Position":"P"},{"UniformNumber":65,"Player_Name":"高宇杰","Player_AVG":0.26,"Player_Position":"C"},{"UniformNumber":96,"Player_Name":"蘇緯達","Player_AVG":0.263,"Player_Position":"1B"},{"UniformNumber":61,"Player_Name":"吳東融","Player_AVG":0.279,"Player_Position":"2B"},{"UniformNumber":9,"Player_Name":"王威晨","Player_AVG":0.316,"Player_Position":"3B"},{"UniformNumber":98,"Player_Name":"岳東華","Player_AVG":0.234,"Player_Position":"SS"},{"UniformNumber":43,"Player_Name":"林書逸","Player_AVG":0.267,"Player_Position":"LF"},{"UniformNumber":39,"Player_Name":"詹子賢","Player_AVG":0.318,"Player_Position":"CF"},{"UniformNumber":1,"Player_Name":"陳子豪","Player_AVG":0.306,"Player_Position":"RF"},{"UniformNumber":23,"Player_Name":"彭政閔","Player_AVG":0.333,"Player_Position":"DH"}]'
        local_team_player = '[{"UniformNumber":6,"Player_Name":"張正偉","Player_AVG":0.324,"Hitter_Order":"棒次1"},{"UniformNumber":23,"Player_Name":"楊晉豪","Player_AVG":0.216,"Hitter_Order":"棒次2"},{"UniformNumber":15,"Player_Name":"胡金龍","Player_AVG":0.354,"Hitter_Order":"棒次3"},{"UniformNumber":9,"Player_Name":"林益全","Player_AVG":0.344,"Hitter_Order":"棒次4"},{"UniformNumber":1,"Player_Name":"林哲瑄","Player_AVG":0.307,"Hitter_Order":"棒次5"},{"UniformNumber":53,"Player_Name":"王正棠","Player_AVG":0.298,"Hitter_Order":"棒次6"},{"UniformNumber":98,"Player_Name":"高國麟","Player_AVG":0.273,"Hitter_Order":"棒次7"},{"UniformNumber":95,"Player_Name":"戴培峰","Player_AVG":0.256,"Hitter_Order":"棒次8"},{"UniformNumber":22,"Player_Name":"李宗賢","Player_AVG":0.27,"Hitter_Order":"棒次9"}]'
        guest_team_players = json.loads(guest_team_player)
        local_team_players = json.loads(local_team_player)
        return_data = dict()
        return_data["GuestTeamPlayers"] = guest_team_players
        return_data["LocalTeamPlayers"] = local_team_players


        # return_data = self._call_api('https://osense.azurewebsites.net/taichungbs/app/Player.php', data = {"game_id":self.game_id})
        defender = list()
        attacker = list()
        if 'Player_Position' in return_data["GuestTeamPlayers"][0]:
            defender = return_data["GuestTeamPlayers"]
            attacker = return_data["LocalTeamPlayers"]
        else:
            defender = return_data["LocalTeamPlayers"]
            attacker = return_data["GuestTeamPlayers"]
        defender_result = dict()
        attacker_result = list()
        for player in defender:
            defender_result[player["Player_Position"]] = " ".join([str(player["UniformNumber"]), player["Player_Name"]])
        for player in attacker:
            attacker_result.append(" ".join([str(player["UniformNumber"]), player["Player_Name"]]))
        self.player_information_lock.acquire()
        # TODO : put player informations into class variable
        self.player_information_lock.release()
        # print(defender_result)
        # print(attacker_result)
        return defender_result, attacker_result

    def get_current_hitters(self):
        # TODO : return current hitters here
        # * add lock
        pass

    def retrieve_current_hitters(self):
        # * api 13
        # TODO : put current hitters into class variable
        # * add lock
        pass

    def _call_api(self, url, data=None):
        if data:
            r = requests.post(url, data = data)
        else:
            r = requests.post(url)
        result = json.loads(r.text)
        # print(r.text)
        return result

if __name__ == '__main__':
    api = api_retriever()
    api.retrieve_game_id()
    api.retrieve_player_information()
