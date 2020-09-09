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
import traceback
import asyncio
import logging
from aiohttp_retry import RetryClient
from aiohttp import ClientSession
import aiohttp
import sys


logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    level=logging.DEBUG,
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger("areq")
logging.getLogger("chardet.charsetprober").disabled = True

class GameNotStarted(Exception):
    pass

class ScrapeEsportsData:

    def __init__(self):
        self.reset()
        self.base_url = "https://feed.lolesports.com/livestats/v1/window/"
        self.invalid_statuses = {x for x in range(100, 600)}
        self.invalid_statuses.remove(200)
        self.invalid_statuses.remove(429)
        self.retry_client = RetryClient()


    def __del__(self):
        self.retry_client.close()


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


    def generate_live_feed(self, match_id):
        seconds_inc = 10
        url = f"https://feed.lolesports.com/livestats/v1/window/{match_id}"

        try:
            init_payload = ScrapeEsportsData.fetch_payload_no_retry(url)
            starting_time = dt.parse(init_payload['frames'][0]['rfc460Timestamp'])
        except Exception:
            raise GameNotStarted()

        secs = starting_time.second
        starting_time = starting_time.replace(second=0, microsecond=0)
        starting_time += datetime.timedelta(seconds=10 * math.ceil(secs / 10))

        duration = 0
        game_finished = False
        # counter = 0
        while not game_finished:
            # counter += 1
            # if counter > 10:
            #     break

            time_it = starting_time + datetime.timedelta(seconds=duration)
            duration += seconds_inc
            time_it_s = time_it.isoformat().replace('+00:00', 'Z')
            url = self.base_url + str(match_id) + f"?startingTime={time_it_s}"
            while True:
                try:
                    payload = ScrapeEsportsData.fetch_payload(url)
                    game_finished = payload['frames'][-1] != 'finished'
                    # current_timestamp = payload['frames'][0]['rfc460Timestamp']
                    break
                except KeyError:
                    time.sleep(1)
            # if current_timestamp == prev_timestamp:
            #     if tolerance > max_tolerance:
            #         break
            #     else:
            #         tolerance += 1
            #         continue
            # tolerance = 0

            yield payload
            # prev_timestamp = current_timestamp


    async def fetch_html(self, url: str, headers=None) -> str:
        """GET request wrapper to fetch page HTML.

        kwargs are passed to `session.request()`.
        """
        try:
            async with self.retry_client.get(url, retry_attempts=10,
                                    retry_for_statuses=self.invalid_statuses, headers=headers if headers is not None \
                else {}) as response:
                return await response.text()
        except (
            aiohttp.ClientError,
            aiohttp.http_exceptions.HttpProcessingError,
        ) as e:
            logger.error(
                "aiohttp exception for %s [%s]: %s",
                url,
                getattr(e, "status", None),
                getattr(e, "message", None),
            )
            raise e
        except Exception as e:
            logger.exception(
                "Non-aiohttp exception occured:  %s", getattr(e, "__dict__", {})
            )
            raise e



    async def download_finished_game(self, match_id):
        seconds_inc = 10
        url = f"https://feed.lolesports.com/livestats/v1/window/{match_id}"
        init_payload = ScrapeEsportsData.fetch_payload_no_retry(url)
        starting_time = dt.parse(init_payload['frames'][0]['rfc460Timestamp'])
        secs = starting_time.second
        starting_time = starting_time.replace(second=0, microsecond=0)
        starting_time += datetime.timedelta(seconds=10 * math.ceil(secs / 10))

        ending_time = starting_time + datetime.timedelta(hours=5)
        end_url = self.base_url + str(match_id) + f"?startingTime={ending_time}"
        final_payload = ScrapeEsportsData.fetch_payload(end_url)
        is_finished = final_payload['frames'][-1]['gameState'] == 'finished'
        if is_finished:
            ending_time = dt.parse(init_payload['frames'][-1]['rfc460Timestamp'])

        urls = []
        duration = 0
        time_it = starting_time
        while time_it != ending_time:
            time_it = starting_time + datetime.timedelta(seconds=duration)
            duration += seconds_inc
            time_it_s = time_it.isoformat().replace('+00:00', 'Z')
            urls.append(self.base_url + str(match_id) + f"?startingTime={time_it_s}")

        frames = [self.fetch_html(url=url) for url in urls]
        return await asyncio.gather(*frames)


    @staticmethod
    @retry(wait_exponential_multiplier=1000, wait_exponential_max=10000)
    def fetch_payload(url, headers=None):
        print(f"attempting to fetch {url}")
        payload_raw = requests.get(url, timeout=5, headers=headers if headers else {}).text
        payload = json.loads(payload_raw)
        return payload


    async def eventids2gameids(self, event_ids):
        url = "https://esports-api.lolesports.com/persisted/gw/getEventDetails?hl=en-US&id="
        futures = [self.fetch_html(url + str(event_id), headers={"x-api-key":
                                                                         "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"})
                   for event_id in event_ids]
        payloads = await asyncio.gather(*futures)
        game_ids = []
        for payload in payloads:
            for game in payload['data']['event']['match']['games']:
                if game['state'] not in {"completed", "unneeded"}:
                    print("ERRRRRRRRRRRROR")
                    raise Exception()
                game_ids.append(game['id'])
        return game_ids



    async def scrape_games(self):
        leagues = ["LEC", "LCK", "LCS"]
        outcomes_file_prefix = "outcomes_"
        event_ids_file_prefix = "event_ids_"
        outcomes = []
        blue_team_champs = []
        red_team_champs = []
        event_ids = []
        game_ids = []

        for league in leagues:
            with open(app_constants.train_paths["win_pred"] + outcomes_file_prefix + league + ".json") as f:
                print(league)
                contents = np.array(json.load(f))
                outcomes.extend(contents[0::3])
                blue_team_champs.extend(contents[1::3])
                red_team_champs.extend(contents[2::3])
            with open(app_constants.train_paths["win_pred"] + event_ids_file_prefix + league + ".json") as f:
                event_ids.extend(json.load(f))
            game_ids = []
            for event_id in event_ids:
                game_ids.extend(list(await self.eventids2gameids([event_id])))

        champs = []
        for b, r in zip(blue_team_champs, red_team_champs):
            blue_champs = [ChampManager().lookup_by("name", c)["int"] for c in b]
            red_champs = [ChampManager().lookup_by("name", c)["int"] for c in r]
            champs.append(blue_champs + red_champs)


        assert len(game_ids) == len(outcomes)
        print("-"*50 + "  Got all game IDs  "+"-"*50)

        x = np.empty((0,Input.len))
        async with ClientSession() as client:
            retry_client = RetryClient(client)
            with open(app_constants.train_paths["win_pred"] + "train.npz", "wb") as result_file:
                try:
                    for game_id, team_comps, outcome in zip(game_ids, champs, outcomes):
                        print(f"Getting gameId: {game_id}")

                        vec = self.gameids2vec(game_ids, outcomes)
                        x = np.concatenate([x, vec], axis=0)
                except Exception:
                    print(traceback.print_exc())
                    np.savez_compressed(result_file, x)
                    await retry_client.close()
                    raise
            await retry_client.close()
        np.savez_compressed(result_file, x)
        print("done")


    def fetch_payload_no_retry(url):
        print(f"INIT attempting to fetch {url}")
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
        snapshots = s.download_finished_game(gameId)
        for snapshot in snapshots:
            yield self.snapshot2vec(snapshot)


    def live_game2odds(self, gameId):


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

        # update swap_teams
        # update gameid
        #update xpath
        gameId = 104174992730350842
        swap_teams = True
        snapshots = s.generate_live_feed(gameId)

        urls = [
            # 'https://sports.betway.com/en/sports/evt/6329588',
                'https://sports.williamhill.com/betting/en-gb/e-sports/OB_EV18429342/flyquest-vs-team-solomid-bo5']
        css_scrapers = [
            # lambda: drivers[0].find_element_by_xpath("//*")[0],
                       lambda: drivers[0].find_elements_by_xpath("//h2[contains(text(),"
                                                                 "'Map 2 Winner')]/../following-sibling::div//span["
                                                                 "@class='betbutton__odds']")]
        options = webdriver.ChromeOptions()
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        drivers = [webdriver.Chrome(ChromeDriverManager().install(), options=options) for _ in urls]
        converters = [float, float]
        for url, driver in zip(urls,drivers):
            driver.execute_cdp_cmd('Network.setUserAgentOverride', {
                "userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.53 Safari/537.36'})
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            driver.get(url)


        scraped_odds_blue = np.zeros((len(drivers),))
        scraped_odds_red = np.zeros((len(drivers),))
        p_odds_blue = np.zeros((len(drivers),))
        p_odds_red = np.zeros((len(drivers),))
        raw_blue = ["" for _ in range(len(drivers))]
        raw_red = ["" for _ in range(len(drivers))]

        model = WinPredModel("standard", network_config=network_config)
        model.load_model()

        while True:

            try:
                snapshot = next(snapshots)
                x = np.array(list(self.snapshot2vec(snapshot)))[-1:]
                self.check_events(x)
                pred_odds_blue = model.bayes_predict_sym(x, 1024)[0]
            except GameNotStarted:
                print("Game not started yet")
                pred_odds_blue = 0.5
                time.sleep(1)
                snapshots = s.generate_live_feed(gameId)


            print("Predicted blue team win: " + str(pred_odds_blue))

            for i, css_scraper in enumerate(css_scrapers):
                print('\t' + urls[i])
                tmp = []
                for _ in range(3):
                    tmp = css_scraper()
                    if tmp != []:
                        break
                    time.sleep(1)
                else:
                    print("Unable to find odds")
                    continue
                try:
                    raw_blue[i] = tmp[0].text
                    raw_red[i] = tmp[1].text
                    scraped_odds_blue[i] = float(converters[i](tmp[0].text))
                    scraped_odds_red[i] = float(converters[i](tmp[1].text))
                    if swap_teams:
                        scraped_odds_blue[i], scraped_odds_red[i] = scraped_odds_red[i], scraped_odds_blue[i]
                        raw_blue[i], raw_red[i] = raw_red[i], raw_blue[i]
                except:
                    continue
                # p_odds_blue[i] = (1/scraped_odds_blue[i])/(1/scraped_odds_blue[i] + 1/scraped_odds_red[i])
                # p_odds_red[i] = (1 / scraped_odds_red[i]) / (1 / scraped_odds_blue[i] + 1 / scraped_odds_red[i])

                p_odds_blue[i] = (1/scraped_odds_blue[i])
                p_odds_red[i] = (1 / scraped_odds_red[i])

                print("\tRaw odds blue: " + str(raw_blue[i]))
                print("\tRaw odds red: " + str(raw_red[i]))
                print("\tOdds blue: " + str(p_odds_blue[i]))
                print("\tOdds red: " + str(p_odds_red[i]))

                print("\n")

                acc = 0.78
                if scraped_odds_blue[i] * pred_odds_blue > scraped_odds_red[i] * (1-pred_odds_blue):
                    print("\tYou should bet on BLUE. Expected payout: " + str(scraped_odds_blue[i] * pred_odds_blue)
                          + f" scaled payout: {acc * scraped_odds_blue[i] * pred_odds_blue}")
                else:
                    print("\tYou should bet on RED. Expected payout: " + str(scraped_odds_red[i] * (1-pred_odds_blue))
                          + f" scaled payout: {acc * scraped_odds_red[i] * (1-pred_odds_blue)}")
                print("\n")
            print("-" * 100)
            print("\n\n")


    def check_events(self, x):
        # if not hasattr(self, "first_turret"):
        #     self.first_turret = False
        # if not hasattr(self, "first_6"):
        #     self.first_6 = False
        # if not hasattr(self, "first_drag"):
        #     self.first_drag = False
        # if not hasattr(self, "first_kill"):
        #     self.first_kill = False

        first_turret = sum(x[0][Input.indices["start"]["turrets_destroyed"]:Input.indices["end"][
            "turrets_destroyed"]]) >= 1
        first_kill = sum(x[0][Input.indices["start"]["kills"]:Input.indices["end"]["kills"]]) >= 1
        first_drag = sum(x[0][Input.indices["start"]["dragons_killed"]:Input.indices["end"]["dragons_killed"]]) >= 1
        first_lvl6 = np.any(x[0][Input.indices["start"]["lvl"]:Input.indices["end"]["lvl"]] >= 6)
        # if not self.first_turret and first_turret:
        #     self.first_turret = True
        #     print("FIRST TURRET")
        # if not self.first_kill and first_kill:
        #     self.first_kill = True
        #     print("FIRST KILL")
        # if not self.first_drag and first_drag:
        #     self.first_drag = True
        #     print("FIRST DRAG")
        # if not self.first_lvl6 and first_lvl6:
        #     self.first_lvl6 = True
        #     print("FIRST LVL 6")

        if first_turret:
            print("FIRST TURRET")
        if first_kill:
            print("FIRST KILL")
        if first_drag:
            print("FIRST DRAG")
        if first_lvl6:
            print("FIRST LVL 6")



    def gameids2vec(self, gameIds, game_results):
        x = np.empty(shape=(0,Input.len), dtype=np.uint64)

        for gameId, game_result in zip(gameIds, game_results):
            self.reset()
            try:
                data_x = self.game2vec(gameId)
            except Exception as e:
                logger.exception(
                    "Error while downloading game. Skip."
                )
                continue
            data_x = np.array([list(frame_gen) for ss_gen in data_x for frame_gen in ss_gen], dtype=np.uint64)
            data_x = np.reshape(data_x, (-1, Input.len))
            # if champs is not None:
            #     data_x[:,Input.indices["start"]["champs"]:Input.indices["end"]["champs"]] = champs
            if game_result == 0:
                data_x = Input.flip_teams(data_x)
            x = np.concatenate([x, data_x], axis=0)
        return x

async def run_async():
    s = ScrapeEsportsData()
    s.scrape_games()


#
#
#
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
104174992730350835,104174992730350836,104174992730350837,104174992730350838,104174992730350839,
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
104169295295198354,104169295295198355,104169295295198356,
104169295295198360,104169295295198361,104169295295198362,104169295295198363,104169295295198364,
104169295295198366,104169295295198367,104169295295198368,
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
    1,1,0,1,0,
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
1,0,0,0,0,
1,0,0
]

# gameIds = [104242514815381919]
# game_results = [0]

# $x('string(//tr[contains(@class,"multirow-highlighter")])')
# a =$x("//div[@class='EventMatch']/a/@href")
# for(var i=0;i<200;++i)
#     console.log(a[i].textContent)
asyncio.run(run_async())

# s.live_game2odds(104169295295198367)

# x = s.tournament2vec(gameIds, game_results)
# out_dir = app_constants.train_paths["win_pred"]
# filename = out_dir + "test_x.npz"
# with open(filename, "wb") as writer:
#     np.savez_compressed(writer, x)
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



