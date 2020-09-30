import dateutil.parser as dt
import datetime
import json
from train_model.input_vector import Input, InputWinPred
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
from bs4 import BeautifulSoup
import logging
from selenium.webdriver.remote.remote_connection import LOGGER

LOGGER.setLevel(logging.WARNING)

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
        self.schedule_url = {"LCK": "https://esports-api.lolesports.com/persisted/gw/getSchedule?hl=en-US&leagueId"
                                   "=98767991310872058&pageToken={pageToken}",
                             "LEC": "https://esports-api.lolesports.com/persisted/gw/getSchedule?hl=en-US&leagueId"
                                   "=98767991302996019&pageToken={pageToken}",
                             "LCS": "https://esports-api.lolesports.com/persisted/gw/getSchedule?hl=en-US&leagueId"
                                    "=98767991299243165&pageToken={pageToken}"
                             }
        self.invalid_statuses = {x for x in range(100, 600)}
        self.invalid_statuses.remove(200)
        self.invalid_statuses.remove(429)
        connector = aiohttp.TCPConnector(limit=10)
        self.retry_client = RetryClient(connector=connector)


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
                    game_finished = payload['frames'][-1]['gameState'] == 'finished'
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
            print(f"Getting {url}")
            async with self.retry_client.get(url, retry_start_timeout=1, retry_attempts=100, retry_factor=2,
                                        retry_for_statuses=self.invalid_statuses, retry_exceptions={OSError,
                                             Exception}, headers=headers if headers is not None else {}) as response:
                return json.loads(await response.text())
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

        #sometimes overshooting doesnt return the last frame. sometimes it's the first frame.
        #however, no game lasts longer than 2 hours so we fetch all those frames and
        ending_time = starting_time + datetime.timedelta(hours=2)
        ending_time_s = ending_time.isoformat().replace('+00:00', 'Z')
        end_url = self.base_url + str(match_id) + f"?startingTime={ending_time_s}"
        final_payload = ScrapeEsportsData.fetch_payload(end_url)
        is_finished = final_payload['frames'][-1]['gameState'] == 'finished'
        if is_finished:
            ending_time = dt.parse(final_payload['frames'][-1]['rfc460Timestamp'])

        urls = []
        duration = 0
        time_it = starting_time
        while time_it < ending_time:
            time_it = starting_time + datetime.timedelta(seconds=duration)
            duration += seconds_inc
            time_it_s = time_it.isoformat().replace('+00:00', 'Z')
            urls.append(self.base_url + str(match_id) + f"?startingTime={time_it_s}")

        snapshots = [self.fetch_html(url=url) for url in urls]
        snapshots = await asyncio.gather(*snapshots)
        if is_finished:
            return snapshots

        #else we need to find the end
        for end_offset, snapshot in enumerate(snapshots[::-1]):
            for frame in snapshot['frames']:
                if frame['gameState'] == 'finished':
                    break
            else:
                continue
            break
        return snapshots[:-(end_offset + 1)]



    @staticmethod
    @retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_delay=120000)
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
            match_game_ids = []
            for game in payload['data']['event']['match']['games']:
                if game['state'] =="completed":
                    match_game_ids.append(game['id'])
            game_ids.append(match_game_ids)
        return game_ids



    async def scrape_games(self):
        leagues = ["LCK", "LEC", "LCS"]
        outcomes_file_prefix = "outcomes_"
        event_ids_file_prefix = "event_ids_"
        outcomes = []
        blue_team_champs = []
        red_team_champs = []
        event_ids = []

        for league in leagues:
            with open(app_constants.train_paths["win_pred"] + outcomes_file_prefix + league + ".json") as f:
                print(league)
                contents = np.array(json.load(f))
                outcomes.extend(contents[0::3])
                blue_team_champs.extend(contents[1::3])
                red_team_champs.extend(contents[2::3])
            with open(app_constants.train_paths["win_pred"] + event_ids_file_prefix + league + ".json") as f:
                event_ids.extend(json.load(f))
        # outcomes = [0,0]
        # event_ids = [103540364360759369]

        game_ids = await self.eventids2gameids(event_ids)
        game_ids = np.ravel(game_ids).tolist()

        # outcomes = outcomes[:len(game_ids)]
        # blue_team_champs = blue_team_champs[:len(game_ids)]
        # red_team_champs = red_team_champs[:len(game_ids)]

        # champs = []
        # for b, r in zip(blue_team_champs, red_team_champs):
        #     blue_champs = [ChampManager().lookup_by("name", c)["int"] for c in b]
        #     red_champs = [ChampManager().lookup_by("name", c)["int"] for c in r]
        #     champs.append(blue_champs + red_champs)

        assert len(game_ids) == len(outcomes)
        print("-"*50 + f"  Got all game IDs: {len(game_ids)} "+"-"*50)

        x = await self.gameids2vec(game_ids, outcomes)
        with open(app_constants.train_paths["win_pred"] + "train.npz", "wb") as result_file:
            np.savez_compressed(result_file, x)
        print("done")


    def fetch_payload_no_retry(url):
        print(f"INIT attempting to fetch {url}")
        payload_raw = requests.get(url, timeout=5).text
        payload = json.loads(payload_raw)
        return payload


    def snapshot2vec(self, data, odds):
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
                          "turrets_destroyed": towers,
                        "team_odds": odds

            }
            try:
                x = InputWinPred.dict2vec(input_)
            except AssertionError:
                print("some vals were negative")
                raise

            yield x


    async def game2vec(self, gameId):
        game_snapshots = await self.download_finished_game(gameId)
        return [self.snapshot2vec(snapshot) for snapshot in game_snapshots]


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
                       "team_odds": 0.05
                       }

        no_noise = {"kills": 0.0,
                    "deaths": 0.0,
                    "assists": 0.0,
                    "total_gold": 0,
                    "cs": 0,
                    "lvl": 0.0,
                    "dragons_killed": 0.0,
                    "baron": 0.0,
                    "elder": 0.0,
                    "blue_side": 0.0,
                    "champs": 0.0,
                    "turrets_destroyed": 0.0,
                    "team_odds": 0.00}

        network_config = {
            "learning_rate": 0.001,
            "stats_dropout": 1.0,
            "champ_dropout": 1.0,
            "noise": gauss_noise
        }

        network_config2 = {
            "learning_rate": 0.001,
            "stats_dropout": 1.0,
            "champ_dropout": 1.0,
            "noise": no_noise
        }


        network_config["noise"] = InputWinPred.scale_rel(network_config["noise"])

        # update swap_teams
        # update gameid
        #update xpath
        gameId = 104841804589478928
        odds = (1.72, 1.95)
        swap_odds = False
        snapshots = s.generate_live_feed(gameId)

        urls = [
            'https://sports.betway.com/en/sports/evt/6483646',
            # 'https://www.pinnacle.com/en/esports/league-of-legends-world-championship/rainbow7-match-vs-lgd-gaming-match/1181384607',
            #     'https://sports.williamhill.com/betting/en-gb/e-sports/OB_EV18616680/rainbow7-vs-lgd-gaming-bo5'
        ]
        css_scrapers = [
                           lambda: drivers[0].find_elements_by_xpath('//span[contains(text(),'
                                                                     '"Map 3 Winner")]/../../../following-sibling::div//div[@class="odds"]'),
            # lambda: drivers[0].find_elements_by_xpath("//span[contains(text(),'Money Line – Map "
            #                                           "3')]/../following-sibling::div/div/a/span[@class='price']"),
            #            lambda: drivers[1].find_elements_by_xpath("//h2[contains(text(),"
            #                                                      "'Match Betting')]/../following-sibling::div//span["
            #                                                      "@class='betbutton__odds']")],
        ]

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
        model2 = WinPredModel("standard", network_config=network_config2)
        model.load_model()
        model2.load_model()
        logging.getLogger("urllib3").setLevel(logging.CRITICAL)
        logging.getLogger("requests").setLevel(logging.CRITICAL)

        while True:

            try:
                snapshot = next(snapshots)
                x = np.array(list(self.snapshot2vec(snapshot, odds)))

                self.check_events(x)
                x = InputWinPred().scale_inputs(x)
                pred_odds_blue = [model.bayes_predict_sym(np.array([x_]), 1024)[0] for x_ in x]
            except GameNotStarted:
                print("Game not started yet")
                pred_odds_blue = [0.5]
                time.sleep(1)
                snapshots = s.generate_live_feed(gameId)


            print("Predicted blue team win: " + str(pred_odds_blue))
            pred_odds_blue = pred_odds_blue[-1]

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
                    if swap_odds:
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

                acc = 1.0
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

        first_turret = np.sum(x[:,InputWinPred.indices["start"]["turrets_destroyed"]:InputWinPred.indices["end"][
            "turrets_destroyed"]]) >= 1
        first_kill = np.sum(x[:,InputWinPred.indices["start"]["kills"]:InputWinPred.indices["end"]["kills"]]) >= 1
        first_drag = np.sum(x[:,InputWinPred.indices["start"]["dragons_killed"]:InputWinPred.indices["end"][
            "dragons_killed"]]) >= 1
        first_lvl6 = np.any(x[:,InputWinPred.indices["start"]["lvl"]:InputWinPred.indices["end"]["lvl"]] >= 6)
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



    async def gameids2vec(self, gameIds, game_results):
        x = []
        error_matches = set()
        for gameId, game_result in zip(gameIds, game_results):
            self.reset()
            try:
                data_x = await self.game2vec(gameId)
                data_x = np.array([list(frame_gen) for ss_gen in data_x for frame_gen in ss_gen], dtype=np.uint64)
                data_x = np.reshape(data_x, (-1, InputWinPred.len))
                # if champs is not None:
                #     data_x[:,Input.indices["start"]["champs"]:Input.indices["end"]["champs"]] = champs
                if game_result == [0]:
                    data_x = InputWinPred.flip_teams(data_x)
                x.append(data_x)
            except Exception as e:
                logger.exception(
                    f"Error while downloading game {gameId}. Skip."
                )
                error_matches.add(gameId)
                logger.exception(
                    f"Error matches {error_matches}. Skip."
                )
                continue
        print(f"Complete. Error matches: {error_matches}")
        return np.concatenate(x, axis=0)

    def get_schedule(self, region):
        schedule = []
        page_token = None
        while page_token != -1:
            part_schedule = json.loads(requests.get(url=self.schedule_url[region].format(pageToken=page_token),
                                                    headers={"x-api-key":"0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"}).text)
            try:
                page_token = part_schedule["data"]["schedule"]["pages"]["older"]
                if page_token is None:
                    page_token = -1
            except KeyError:
                break
            events = part_schedule["data"]["schedule"]["events"]
            events_parsed = [(event['match']['id'], event["startTime"],event['match']['teams'][0]["name"],event['match'][
                'teams'][1]["name"]) for
                             event in events]
            schedule = events_parsed + schedule
        return schedule

import itertools

class ScrapeChampStats:

    def __init__(self):
        self.url = "https://api.op.lol/mega/?ep=champion2&p=d&v=1&patch=10.{" \
                     "patch}&cid={champ_id}&lane={lane}&tier=master_plus&queue" \
                  "=420&region=all"
        self.lanes = ['top', 'jungle', 'middle', 'bottom', 'support']
        self.counters_data = ["champ1","patch", "champ2", "lane1", "lane2", "num_games", "num_wins", "champ2_wr"]
        self.syn_data = ["champ1","patch", "champ2", "lane1", "lane2", "num_games", "num_wins", "champ2_wr"]
        self.champ_data = ["champ","patch", "lane", "num_games", "won", "num_length0_15","num_length15_20",
                           "num_length20_25","num_length25_30","num_length30_35","num_length35_40","num_length40+",
                           "won_length0_15", "won_length15_20",
                           "won_length20_25", "won_length25_30", "won_length30_35", "won_length35_40", "won_length40+",
                           "top_wr"]
        self.patches = range(1,20)

        self.counters_stats = []
        self.syn_stats = []
        self.champ_stats = []


    def scrape_stats(self, payload, champ_lane, patch, champ_id):
        syn_vs_opp = ["team", "enemy"]
        champ_lane_index = self.lanes.index(champ_lane)
        for lane in self.lanes:
            for so, data_field in zip(syn_vs_opp, [self.syn_stats, self.counters_stats]):
                try:
                    current_lane_data = payload[so+"_"+lane]
                    current_lane_index = self.lanes.index(lane)

                except KeyError:
                    continue
                for data_champ in current_lane_data:
                    data_field.append([champ_id, patch, data_champ[0], champ_lane_index, current_lane_index, data_champ[
                        1], data_champ[2], data_champ[3]])

        stats = [champ_id, patch, champ_lane_index, payload['n'], payload['wr'],0,0,0,0,0,0,0,0,0,0,0,0,0,0, \
                payload['topStats']['topwin']]

        for i in range(1,8):
            try:
                stats[5 + i] = payload['time'][str(i)]
            except (KeyError, TypeError):
                stats[5 + i] = 0
            try:
                stats[12 + i] = payload['timeWin'][str(i)]
            except (KeyError, TypeError):
                stats[12 + i] = 0


        self.champ_stats.append(stats)



    def scrape_all_stats(self):
        # with open("matchups.json") as f:
        #     games = json.loads(f.read())
        # headers = ['game_id', 'team1','team2','patch', 'players', 'champs']
        # for game in games:
        #     champs = game[12:]
        #     for lane, champ in enumerate(champs):
        #         self.scrape_stats(champ, lane % 5, game[0])
        urls = []
        for champ in ChampManager().get_ints().values():
            if champ['id'] == '0':
                continue
            for lane in self.lanes:
                for patch in self.patches:
                    urls.append([self.url.format(champ_id = champ['id'], patch=patch, lane=lane), patch,lane,
                                 champ['id']])
        for i,(url,patch,lane,champ_id) in enumerate(urls):
            print(f"{round(i/len(urls)*100)} %")
            payload = json.loads(requests.get(url=url).text)
            try:
                self.scrape_stats(payload, champ_lane=lane, patch=patch, champ_id=int(champ_id))
            except Exception as e:
                print("ERROR")
                print(e)
                print(traceback.print_exc())
                continue
        with open(f'champ_stats.npz', 'wb') as fp:
            np.savez_compressed(fp, np.array(self.champ_stats))
        with open(f'counter_stats.npz', 'wb') as fp:
            np.savez_compressed(fp, np.array(self.counters_stats))
        with open(f'syn_stats.npz', 'wb') as fp:
            np.savez_compressed(fp, np.array(self.syn_stats))


    async def scrape_extra_esports_stats(self):
        leagues = ["LCK", "LCS", "LEC"]
        event_ids_file_prefix = "event_ids_"
        s = scrape_esports_data.ScrapeEsportsData()
        result = []
        for league in leagues:
            with open(app_constants.train_paths["win_pred"] + event_ids_file_prefix + league + ".json") as f:
                event_ids = json.load(f)
            game_ids = await s.eventids2gameids(event_ids)
            for game_id in game_ids:
                payload = s.fetch_payload(
                    url=f"https://feed.lolesports.com/livestats/v1/window/{game_id}")
                blue_top_name = payload["gameMetadata"]["blueTeamMetadata"]['participantMetadata'][0]['summonerName']
                red_top_name = payload["gameMetadata"]["redTeamMetadata"]['participantMetadata'][0]['summonerName']
                blue_team_tag = blue_top_name[:blue_top_name.index(' ')].lower()
                red_team_tag = red_top_name[:red_top_name.index(' ')].lower()
                blue_team = scrape_esports_data.team_codes[league][blue_team_tag][0]
                red_team = scrape_esports_data.team_codes[league][red_team_tag][0]
                patch_raw = payload['gameMetadata']['patchVersion']
                first_dot = patch_raw.index('.')
                second_dot = patch_raw[first_dot:].index('.')
                patch = int(patch_raw[first_dot:second_dot])
                blue_players = np.array([(participant['summonerName'],participant['championId']) for participant in \
                    payload[
                    "gameMetadata"][
                    "blueTeamMetadata"][
                    'participantMetadata']])
                red_players = np.array([(participant['summonerName'],participant['championId']) for participant in
                                       payload["gameMetadata"][
                    "redTeamMetadata"][
                    'participantMetadata']])
                row = [game_id, patch, blue_team, red_team, blue_players[:,0].tolist() + red_players[:,0].tolist(),
                       blue_players[:,1].tolist() + red_players[:,1].tolist()]
                result.append(row)
        with open("games_metadata.json", "w") as f:
            json.dump(result, f)


class ScrapePlayerStats:

        def __init__(self):
            self.player_stats_urls = {
                "all":"https://lol.gamepedia.com/index.php?pfRunQueryFormName=TournamentStatistics&title=Special"
                      "%3ARunQuery%2FTournamentStatistics&TS%5Bpreload%5D=PlayerByChampion&TS%5Btournament%5D=&TS"
                      "%5Blink%5D={player_name}&TS%5Bchampion%5D=&TS%5Brole%5D=&TS%5Bteam%5D=&TS%5Bpatch%5D=&TS%5Byear"
                      "%5D=&TS%5Bregion%5D=&TS%5Btournamentlevel%5D=&TS%5Bwhere%5D=&TS%5Bincludelink%5D%5Bis_checkbox"
                      "%5D=true&TS%5Bshownet%5D%5Bis_checkbox%5D=true&wpRunQuery=Run+query&pf_free_text=",

                                      "2020season":
            "https://lol.gamepedia.com/index.php?pfRunQueryFormName=TournamentStatistics&title=Special%3ARunQuery" \
            "%2FTournamentStatistics&TS%5Bpreload%5D=PlayerByChampion&TS%5Btournament%5D=&TS%5Blink%5D={" \
            "player_name}&TS%5Bchampion%5D=&TS%5Brole%5D=&TS%5Bteam%5D=&TS%5Bpatch%5D=&TS%5Byear%5D=2020&TS%5Bregion" \
            "%5D=&TS%5Btournamentlevel%5D=&TS%5Bwhere%5D=&TS%5Bincludelink%5D%5Bis_checkbox%5D=true&TS%5Bshownet%5D%5Bis_checkbox%5D=true&wpRunQuery=Run+query&pf_free_text=",
                                      "2020summer":
                                          "https://lol.gamepedia.com/index.php?pfRunQueryFormName=TournamentStatistics&title=Special%3ARunQuery" \
                "%2FTournamentStatistics&TS%5Bpreload%5D=PlayerByChampion&TS%5Btournament%5D=&TS%5Blink%5D={" \
                "player_name}&TS%5Bchampion%5D=&TS%5Brole%5D=&TS%5Bteam%5D=&TS%5Bpatch%5D=&TS%5Byear%5D=2020&TS" \
                "%5Bregion%5D=&TS%5Btournamentlevel%5D=&TS%5Bwhere%5D=SG.Overviewpage%3D%22{region}"
                                          "%2F2020+Season%2FSummer+Season%22&TS%5Bincludelink%5D%5Bis_checkbox%5D=true&TS%5Bshownet%5D%5Bis_checkbox%5D=true&wpRunQuery=Run+query&pf_free_text="
                                      }

        def parse_row(self, row):
            stats = row
            champ_name = stats[0].contents[0].contents[1].contents[0]


            try:
                games = int(stats[1].a.contents[0])
            except (ValueError, AttributeError):
                games = 1
            try:
                wins = int(stats[2].contents[0])
            except (ValueError, AttributeError):
                wins = 1
            try:
                losses = int(stats[3].contents[0])
            except (ValueError, AttributeError):
                losses = 1
            try:
                wr = float(stats[4].contents[0][:-1]) / 100
            except (ValueError, AttributeError):
                wr = 0.5
            try:
                kills = float(stats[5].contents[0])
            except (ValueError, AttributeError):
                kills = -1
            try:
                deaths = float(stats[6].contents[0])
            except (ValueError, AttributeError):
                deaths = -1
            try:
                assists = float(stats[7].contents[0])
            except (ValueError, AttributeError):
                assists = -1
            try:
                kda = float(stats[8].contents[0])
            except (ValueError, AttributeError):
                kda = -1
            try:
                cs = float(stats[9].contents[0])
            except (ValueError, AttributeError):
                cs = -1
            try:
                csm = float(stats[10].contents[0])
            except (ValueError, AttributeError):
                csm = -1
            try:
                gold = float(stats[11].contents[0].contents[0]) * 1000
            except (ValueError, AttributeError):
                gold = -1
            try:
                gpm = int(stats[12].contents[0][:-1])
            except (ValueError, AttributeError):
                gpm = -1
            try:
                kp = float(stats[13].contents[0][:-1]) / 100
            except (ValueError, AttributeError):
                kp = -1
            try:
                ks = float(stats[14].contents[0][:-1]) / 100
            except (ValueError, AttributeError):
                ks = -1
            try:
                gs = float(stats[15].contents[0][:-1]) / 100
            except (ValueError, AttributeError):
                gs = -1

            return (champ_name, np.array([games, wins, losses, wr, kills, deaths, assists, kda, cs, csm, gold, gpm, kp, ks,
                                   gs], dtype=np.float))

        def scrape_all_stats(self):
            with open("season10_players.json") as f:
                player_names = json.loads(f.read())
            result = dict()
            timeframes = ["all", "2020season", "2020summer"]
            for league in player_names:
                for player_name in player_names[league]:

                    print(player_name)
                    # payload = requests.get(url="https://lol.gamepedia.com/" + player_name).content
                    # content = BeautifulSoup(payload, 'html.parser')
                    # try:
                    #     print([pn.contents[2].attrs['title'] for pn in content.find('ul').contents])
                    # except AttributeError:
                    #     pass
                    result[player_name] = dict()
                    for timeframe in timeframes:
                        try:
                            result[player_name][timeframe] = self.scrape_stats_for_player(player_name, timeframe,
                                                                                          league)
                        except ValueError:
                            #this is so dirty
                            result[player_name]["2020summer"] = result[player_name]['2020season']

            with open(f'player_champ_proficiency.json', 'w') as fp:
                json.dump(result, fp)


        def scrape_stats_for_player(self, player_name, timeframe, region):
            page = requests.get(self.player_stats_urls[timeframe].format(player_name=player_name, region=region))
            results = BeautifulSoup(page.content, 'html.parser')
            results = results.find_all('table', class_='wikitable')
            stats = results[0].find_all('td')
            if stats == []:
                print(f"Error: no results for {player_name} {timeframe}")
                raise ValueError(f"Error: no results for {player_name} {timeframe}")
            stats_row_len = 16
            stats_arr = []
            champs_arr = []
            for i in range(0,len(stats)//stats_row_len):
                parsed_result = self.parse_row(stats[i*stats_row_len: (i+1)*stats_row_len])
                champs_arr.append(parsed_result[0])
                stats_arr.append(parsed_result[1])

            stats_arr = np.reshape(stats_arr, (-1,stats_row_len-1))

            for col_i in range(1,stats_row_len-1):
                column = np.array(stats_arr[:,col_i])
                invalid_indices = np.logical_or(column == -1, np.isnan(column))
                if np.any(invalid_indices):
                    valid_indices = np.logical_not(invalid_indices)
                    avg = np.mean(column[valid_indices])
                    column[invalid_indices] = avg
                    stats_arr[:, col_i] = column

            champs_arr.append("avg")
            stats_arr = np.concatenate([stats_arr, np.expand_dims(np.mean(stats_arr,axis=0), axis=0)],axis=0)
            return dict(zip(champs_arr, stats_arr.tolist()))







async def run_async():
    s = ScrapeEsportsData()
    s.scrape_games()
    # a = s.fetch_payload(url="https://feed.lolesports.com/livestats/v1/window/104174613353783764")

    # x = await s.game2vec(104174613333794954)
    # print(x)
    # s.scrape_stats("bjergsen", "summer")


if __name__ == "__main__":
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

    s = ScrapeEsportsData()
    s.live_game2odds(0)

    # s = ScrapeEsportsData()
    # s.scrape_all_stats()
    print('hi')
    # gameIds = [104242514815381919]
    # game_results = [0]

    # $x('string(//tr[contains(@class,"multirow-highlighter")])')
    # a =$x("//div[@class='EventMatch']/a/@href")
    # for(var i=0;i<200;++i)
    #     console.log(a[i].textContent)


    # s.live_game2odds(0)

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

    # def funcfunc():
    #     outcomes = []
    #     # for league in ["LCK", "LEC", "LCS"]:
    #     with open(app_constants.train_paths["win_pred"] + "outcomes_" + "LCK" + ".json") as f:
    #         contents = np.array(json.load(f))
    #         outcomes.extend(contents[0::3])
    #     X_pro = np.load("training_data/win_pred/train.npz")['arr_0']
    #     new_out = []
    #     current_game_id = X_pro[0,0]
    #     current_index = 0
    #     first_i = 0
    #     second_i = 0
    #     for outcome in outcomes:
    #         if current_game_id==104174613353783767:
    #             if first_i < 2:
    #                 first_i += 1
    #                 continue
    #         if current_game_id==103540364360759371:
    #             if second_i < 1:
    #                 second_i += 1
    #                 continue
    #
    #         while current_index < X_pro.shape[0] and X_pro[current_index,0] == current_game_id:
    #             current_index += 1
    #         if outcome == [0]:
    #             new_game = Input.flip_teams(X_pro[:current_index])
    #         else:
    #             new_game = X_pro[:current_index]
    #         new_out.append(new_game)
    #         current_game_id = X_pro[current_index,0]
    #         X_pro = X_pro[current_index:]
    #         current_index = 0
    #     return new_out
    #


    # asyncio.run(run_async())


    #
    # new_out = funcfunc()
    # new_out = np.concatenate(new_out, axis=0)
    # with open(app_constants.train_paths["win_pred"] + "lulz.npz", "wb") as result_file:
    #     np.savez_compressed(result_file, new_out)
