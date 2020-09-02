import requests
import dateutil.parser as dt
import datetime
import json
from train_model.input_vector import Input
import numpy as np
from utils.artifact_manager import ChampManager
from constants import game_constants, app_constants
from retrying import retry
import math
from train_model.model import WinPredModel
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import requests

class ScrapeEsportsData:

    def __init__(self):
        self.reset()

    def reset(self):
        self.elder_start_timer = None
        self.baron_start_timer = None
        self.baron_active = [0, 0]
        self.prev_baron_active_raw = [0, 0]
        self.elder_active = [0, 0]
        self.prev_elder_active_raw = [0, 0]
        self.prev_blue_team_dragons = None
        self.prev_red_team_dragons = None
        self.dragon2cap = {"cloud": "AIR_DRAGON", "mountain": "EARTH_DRAGON", "infernal": "FIRE_DRAGON",
                           "ocean": "WATER_DRAGON"}


    @staticmethod
    def generate_live_feed(match_id):
        seconds_inc = 10
        url = f"https://feed.lolesports.com/livestats/v1/window/{match_id}"
        starting_time = dt.parse(ScrapeEsportsData.fetch_payload(url)['frames'][0]['rfc460Timestamp'])
        secs = starting_time.second
        starting_time = starting_time.replace(second=0, microsecond=0)
        starting_time += datetime.timedelta(seconds=10 * math.ceil(secs / 10))
        duration = 0
        prev_timestamp = None
        tolerance = 0
        # counter = 0
        while True:
            # counter += 1
            # if counter > 10:
            #     break
            time_it = starting_time + datetime.timedelta(seconds=duration)
            duration += seconds_inc
            time_it_s = time_it.isoformat().replace('+00:00', 'Z')
            url = f"https://feed.lolesports.com/livestats/v1/window/{match_id}?startingTime={time_it_s}"
            while True:
                try:
                    payload = ScrapeEsportsData.fetch_payload(url)
                    current_timestamp = payload['frames'][0]['rfc460Timestamp']
                    break
                except KeyError:
                    time.sleep(1)
            if current_timestamp == prev_timestamp:
                if tolerance > 15:
                    break
                else:
                    tolerance += 1
                    continue
            tolerance = 0
            yield payload
            prev_timestamp = current_timestamp

    @staticmethod
    @retry(wait_exponential_multiplier=1000, wait_exponential_max=10000)
    def fetch_payload(url):
        print(f"attempting to fetch {url}")
        payload_raw = requests.get(url, timeout=5).text
        payload = json.loads(payload_raw)
        return payload


    def snapshot2vec(self, data):
        champ_names = [participant['championId'] for participant in data['gameMetadata']['blueTeamMetadata'][
            'participantMetadata']] + [participant['championId'] for participant in data['gameMetadata'][
            'redTeamMetadata']['participantMetadata']]

        for frame in data['frames']:
            frame_timestamp = dt.parse(frame['rfc460Timestamp'])
            total_gold = [participant["totalGold"] for participant in frame['blueTeam']['participants'] + frame['redTeam'][
                'participants']]
            kda = [[participant["kills"],participant["deaths"],participant["assists"]] for participant in frame[
                'blueTeam']['participants'] + frame['redTeam']['participants']]
            cs = [participant['creepScore'] for participant in frame[
                'blueTeam']['participants'] + frame['redTeam']['participants']]
            lvl = [participant['level'] for participant in frame[
                'blueTeam']['participants'] + frame['redTeam']['participants']]
            towers = [frame['blueTeam']['towers'],frame['redTeam']['towers']]
            dragon_soul_blue = [0,0,0,0]
            dragon_soul_red = [0,0,0,0]

            if (self.baron_active[0] or self.baron_active[1]) and frame_timestamp > \
                    self.baron_start_timer + datetime.timedelta(seconds=game_constants.BARON_DURATION):
                self.baron_active = [0, 0]
            if (self.elder_active[0] or self.elder_active[1]) and frame_timestamp > \
                    self.elder_start_timer + datetime.timedelta(seconds=game_constants.ELDER_DURATION):
                self.elder_active = [0, 0]

            dragons_killed = np.zeros((2,4))
            blue_team_dragons = frame['blueTeam']['dragons']
            red_team_dragons = frame['redTeam']['dragons']

            if "elder" in set(blue_team_dragons + red_team_dragons):
                if blue_team_dragons != self.prev_blue_team_dragons:
                    self.elder_active = [1, 0]
                    self.elder_start_timer = frame_timestamp
                elif red_team_dragons != self.prev_red_team_dragons:
                    self.elder_active = [0, 1]
                    self.elder_start_timer = frame_timestamp
                self.prev_blue_team_dragons = blue_team_dragons
                self.prev_red_team_dragons = red_team_dragons
                blue_team_dragons = list(filter(lambda a: a != "elder", blue_team_dragons))
                red_team_dragons = list(filter(lambda a: a != "elder", red_team_dragons))
            else:
                self.prev_blue_team_dragons = blue_team_dragons
                self.prev_red_team_dragons = red_team_dragons

            for dragon in blue_team_dragons:
                dragons_killed[0][game_constants.dragon2index[self.dragon2cap[dragon]]] += 1
            for dragon in red_team_dragons:
                dragons_killed[1][game_constants.dragon2index[self.dragon2cap[dragon]]] += 1

            if len(blue_team_dragons) >= 4:
                dragon_soul_blue[game_constants.dragon2index[self.dragon2cap[blue_team_dragons[3]]]] = 1
            elif len(red_team_dragons) >= 4:
                dragon_soul_red[game_constants.dragon2index[self.dragon2cap[red_team_dragons[3]]]] = 1

            baron_active = [frame['blueTeam']['barons'], frame['redTeam']['barons']]
            if baron_active != self.prev_baron_active_raw:
                new_baron = np.array(baron_active) - np.array(self.prev_baron_active_raw)
                self.baron_active = new_baron
                self.baron_start_timer = frame_timestamp

            self.prev_baron_active_raw = baron_active

            champs = [ChampManager().lookup_by("name", chname)["int"] for chname in champ_names]

            kda = np.ravel(kda)
            input_ = {
                          "gameid": int(data['esportsGameId']),
                          "champs": champs,
                          "total_gold": total_gold,
                          "blue_side": [1, 0],
                          "cs": cs,
                          "lvl": lvl,
                          "kills": kda[0::3],
                          "deaths": kda[1::3],
                          "assists": kda[2::3],
                          "baron": self.baron_active,
                          "elder": self.elder_active,
                          "dragons_killed": np.ravel(dragons_killed),
                          "dragon_soul_type": dragon_soul_blue + dragon_soul_red,
                          "turrets_destroyed": towers}
            try:
                x = Input.dict2vec(input_)
            except AssertionError:
                print("some vals were negative")
                raise

            yield x


    def game2vec(self, gameId):
        snapshots = list(s.generate_live_feed(gameId))
        for snapshot in snapshots:
            yield self.snapshot2vec(snapshot)


    def live_game2odds(self, gameId):
        snapshots = s.generate_live_feed(gameId)

        gauss_noise = {"kills": 0.5,
                       "deaths": 0.5,
                       "assists": 0.5,
                       "total_gold": 125,
                       "cs": 10,
                       "lvl": 0.4,
                       "dragons_killed": 0.2,
                       "baron": 0.05,
                       "elder": 0.05,
                       "blue_side": 0.2,
                       "champs": 0.1,
                       "turrets_destroyed": 0.4,
                       "current_gold": 125}
        network_config = {
            "learning_rate": 0.001,
            "stats_dropout": 1.0,
            "champ_dropout": 1.0,
            "noise": gauss_noise
        }

        driver = webdriver.Firefox()

        driver.get('https://www.pinnacle.com/en/soccer/france-national/fc-bastia-borgo-vs-bastia/1164026681')
        # driver.get("https://www.bet365.com/#/IP/EV204480595520980842C151")

        p = []


        model = WinPredModel("standard", network_config=network_config)
        model.load_model()
        first_turret = False
        for snapshot in snapshots:
            x = np.array(list(self.snapshot2vec(snapshot)))[-1:]
            turrets = x[0][Input.indices["start"]["turrets_destroyed"]:Input.indices["end"]["turrets_destroyed"]]
            if not first_turret and sum(turrets==1):
                first_turret = True
                print("FIRST TURRET DESTROYED")

            p = []
            while p == []:
                p = driver.find_elements_by_class_name("price")
            live_odds_blue = float(p[0].text)
            live_odds_red = float(p[1].text)

            p_odds_blue = (1/live_odds_blue)/(1/live_odds_blue + 1/live_odds_red)
            p_odds_red = (1 / live_odds_red) / (1 / live_odds_blue + 1 / live_odds_red)
            pred_odds_blue = model.bayes_predict_sym(x, 128)[0]
            print("  Betting blue team win: "+str(p_odds_blue))
            print("Predicted blue team win: " + str(pred_odds_blue))
            if pred_odds_blue > p_odds_blue:
                print("You should bet on BLUE. Diff: "+str(p_odds_blue - pred_odds_blue))
            else:
                print("You should bet on RED. Diff: " + str(p_odds_blue - pred_odds_blue))
            print("\n\n")




    def tournament2vec(self, gameIds, game_results):
        x = np.empty(shape=(0,Input.len), dtype=np.uint64)

        for gameId, game_result in zip(gameIds, game_results):
            self.reset()
            data_x = self.game2vec(gameId)
            data_x = np.array([list(frame_gen) for ss_gen in data_x for frame_gen in ss_gen], dtype=np.uint64)
            data_x = np.reshape(data_x, (-1, Input.len))
            if game_result == 0:
                data_x = Input.flip_teams(data_x)
            x = np.concatenate([x, data_x], axis=0)



        out_dir = app_constants.train_paths["win_pred"]
        filename = out_dir + "test_x.npz"
        with open(filename, "wb") as writer:
            np.savez_compressed(writer, x)



#
#
#
s = ScrapeEsportsData()
# gameIds = [104242514815381917, 104242514815381915, 104242514815381913, 104242514815381919, 104242514815381911,
#                    104242514815381921, 104242514816168363, 104242514816168357, 104242514816168355, 104242514816168365,
#                    104242514816168361, 104242514816168359, 104251966834409897, 104252200467817589, 104242514817020336,
#                    104242514817020337, 104242514817020338, 104242514817020339, 104242514817020342, 104242514817020343,
#                    104242514817020344, 104242514817020348, 104242514817020349, 104242514817020350, 104242514817020351]
# game_results = [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
gameIds = [
    #LCS
    104174992730350781, 104174992730350782, 104174992730350783,
           104174992730350775, 104174992730350776, 104174992730350777, 104174992730350778, 104174992730350779,
           104174992730350787, 104174992730350788, 104174992730350789,
           104174992730350793, 104174992730350794, 104174992730350795,
           104174992730350805, 104174992730350806, 104174992730350807, 104174992730350808,
104174992730350799, 104174992730350800, 104174992730350801,
104174992730350817,104174992730350818,104174992730350819,
104174992730350811, 104174992730350812, 104174992730350813, 104174992730350814, 104174992730350815,
104174992730350829,104174992730350830,104174992730350831,104174992730350832,
104174992730350823,104174992730350824,104174992730350825,104174992730350826,104174992730350827,
#LCK
104174613333860706, 104174613333860707,
104174613333926330, 104174613333926331, 104174613333926332,
104174613333860666,104174613333860667,
104174613333860746, 104174613333860747,
104174613353718215,104174613353783752,104174613353783753,
104174613353783755,104174613353783756,104174613353783757,

#LPL
#does not have live stats
# 104282610721221959,104282610721221960,104282610721221961,104282610721221962,
# 104282610721221965,104282610721221966,104282610721221967,104282610721221968,
# 104282610721221971,104282610721221972,104282610721221973,104282610721221974,
# 104282610721221977,104282610721221978,104282610721221979,
#LEC
104169295295132788,104169295295132789,104169295295132790,
104169295295132800, 104169295295132801,104169295295132802,104169295295132803,
104169295295132794,104169295295132795,104169295295132796,
104169295295198348,104169295295198349,104169295295198350,104169295295198351,
104169295295132806,104169295295132807,104169295295198344,104169295295198345,104169295295198346,
104169295295198354,104169295295198355,104169295295198356
           ]
game_results = [
    #LCS
    0,0,1,
                1,0,1,0,1,
                0,0,1,
                1,0,1,
                0,1,1,0,
1,1,1,
1,0,0,
1,1,1,1,1,
0,0,0,1,
0,1,1,1,1,
#LCK
1,0,
0,0,1,
0,1,
0,1,
0,1,0,
1,1,1,


#LPL
# 1,1,0,0,
# 1,1,0,1,
# 0,1,0,1,
# 0,0,1,
#LEC
0,0,1,
1,0,1,1,
0,0,0,
0,1,0,0,
0,1,0,1,0,
1,0,0,
]
# gameIds = [104242514815381919]
# game_results = [0]
# s.live_game2odds(104174992730350827)
s.tournament2vec(gameIds, game_results)
# 104169295295132807
# 104169295295198344
# 104169295295198345
# 104169295295198346
#

# X = np.load("training_data/win_pred/test_x.npz")['arr_0']
# Y = np.load("training_data/win_pred/test_y.npz")['arr_0']
# result = np.empty((Input.len,))
# for x,y in zip(X,Y):
#     if y == 0:
#         x = s.trainer.flip_teams(np.array([x]))[0]
#     result = np.concatenate([result, x])
#
# with open("training_data/win_pred/test.npz", "wb") as writer:
#     np.savez_compressed(writer, result)



# for data in data_x:
#     print(list(data))
# for data in data_x_flipped:
#     print(data)
# s.game2vec(104242514815381917, "2020-05-28T07:20:10.000Z")
