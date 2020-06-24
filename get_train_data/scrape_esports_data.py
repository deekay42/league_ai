import requests
import dateutil.parser as dt
import datetime
import json
from train_model.input_vector import Input
from train_model.model import WinPredModel
from train_model.train import WinPredTrainer
import numpy as np
from utils.artifact_manager import ChampManager
from constants import game_constants, app_constants
from retrying import retry
import math


class ScrapeEsportsData:

    def __init__(self):
        self.elder_start_timer = None
        self.baron_start_timer = None
        self.baron_active = [0,0]
        self.prev_baron_active_raw = [0,0]
        self.elder_active = [0,0]
        self.prev_elder_active_raw = [0,0]
        self.prev_blue_team_dragons = None
        self.prev_red_team_dragons = None
        self.dragon2index = {"cloud": "AIR_DRAGON", "mountain": "EARTH_DRAGON", "infernal":"FIRE_DRAGON",
                             "ocean":"WATER_DRAGON"}

        self.trainer = WinPredTrainer()
        self.model = WinPredModel(type="standard")
        # self.model.load_model()

    @staticmethod
    def generate_live_feed(match_id):
        seconds_inc = 10
        url = f"https://feed.lolesports.com/livestats/v1/window/{match_id}"
        starting_time = dt.parse(ScrapeEsportsData.fetch_payload(url)['frames'][0]['rfc460Timestamp'])
        secs = starting_time.second
        starting_time = starting_time.replace(second=0, microsecond=0)
        starting_time += datetime.timedelta(seconds=10*math.ceil(secs/10))
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
            payload = ScrapeEsportsData.fetch_payload(url)
            current_timestamp = payload['frames'][0]['rfc460Timestamp']
            print(current_timestamp)
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
        print(f"success\n\n")
        return payload

# lol = generate_live_feed(104242514815381917, "2020-05-28T07:20:10.000Z")
# print(list(lol))


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
            dragon_soul_blue = "NONE"
            dragon_soul_red = "NONE"

            if (self.baron_active[0] or self.baron_active[1]) and frame_timestamp > \
                    self.baron_start_timer + datetime.timedelta(seconds=game_constants.BARON_DURATION):
                self.baron_active = [0, 0]
            if (self.elder_active[0] or self.elder_active[1]) and frame_timestamp > \
                    self.elder_start_timer + datetime.timedelta(seconds=game_constants.ELDER_DURATION):
                self.elder_active = [0, 0]

            dragons = {"AIR_DRAGON": [0, 0], "EARTH_DRAGON": [0, 0], "FIRE_DRAGON": [0, 0], "WATER_DRAGON": [0, 0]}
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
                dragons[self.dragon2index[dragon]][0] += 1
            for dragon in red_team_dragons:
                dragons[self.dragon2index[dragon]][1] += 1

            if len(blue_team_dragons) >= 4:
                dragon_soul_blue = self.dragon2index[blue_team_dragons[3]]
            elif len(red_team_dragons) >= 4:
                dragon_soul_red = self.dragon2index[red_team_dragons[3]]

            baron_active = [frame['blueTeam']['barons'], frame['redTeam']['barons']]
            if baron_active != self.prev_baron_active_raw:
                new_baron = np.array(baron_active) - np.array(self.prev_baron_active_raw)
                self.baron_active = new_baron
                self.baron_start_timer = frame_timestamp

            self.prev_baron_active_raw = baron_active

            input_ = {"scale_inputs": False, "gameId": data['esportsGameId'],"champs_str": champ_names,
                          "total_gold": total_gold,
                          "first_team_blue_start": 1,
                          "cs": cs,
                          "lvl": lvl,
                          "kda": kda,
                          "baron_active": self.baron_active,
                          "elder_active": self.elder_active,
                          "dragons": dragons,
                          "dragon_soul_type": [dragon_soul_blue, dragon_soul_red],
                          "turrets": towers}

            x = self.model.transform_inputs(**input_)
            yield x


    def game2vec(self, gameId):
        snapshots = list(s.generate_live_feed(gameId))
        for snapshot in snapshots:
            yield self.snapshot2vec(snapshot)


    def tournament2vec(self, gameIds, game_results):
        x = np.empty(shape=(0,Input.len))

        for gameId, game_result in zip(gameIds, game_results):
            data_x = self.game2vec(gameId)
            data_x = np.array([list(frame_gen) for ss_gen in data_x for frame_gen in ss_gen])
            data_x = np.reshape(data_x, (-1, Input.len))
            if game_result == 0:
                data_x = self.trainer.flip_teams(data_x)
            x = np.concatenate([x, data_x], axis=0)



        out_dir = app_constants.train_paths["win_pred"]
        filename = out_dir + "test_x.npz"
        with open(filename, "wb") as writer:
            np.savez_compressed(writer, x)


#
#
#
s = ScrapeEsportsData()
gameIds = [104242514815381917, 104242514815381915, 104242514815381913, 104242514815381919, 104242514815381911,
                   104242514815381921, 104242514816168363, 104242514816168357, 104242514816168355, 104242514816168365,
                   104242514816168361, 104242514816168359, 104251966834409897, 104252200467817589, 104242514817020336,
                   104242514817020337, 104242514817020338, 104242514817020339, 104242514817020342, 104242514817020343,
                   104242514817020344, 104242514817020348, 104242514817020349, 104242514817020350, 104242514817020351]
game_results = [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
# gameIds = [104242514815381919]
# game_results = [0]
s.tournament2vec(gameIds, game_results)
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
