
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
        self.last_dragon_slain_time = None
        self.last_baron_slain_time = None
        self.last_elder_slain_time = None
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
                if game['state'] == "completed":
                    match_game_ids.append(game['id'])
            game_ids.extend(match_game_ids)
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

        x, meta_x = await self.gameids2vec(game_ids, outcomes)
        with open(app_constants.train_paths["win_pred"] + "train_new.npz", "wb") as result_file:
            np.savez_compressed(result_file, x)
        with open(app_constants.train_paths["win_pred"] + "train_new_meta.npz", "wb") as result_file:
            np.savez_compressed(result_file, meta_x)
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
        patch = data['gameMetadata']['patchVersion']
        dotindex = patch.index('.')
        patch = patch[dotindex+1:]
        try:
            dotindex = patch.index('.')
        except ValueError:
            dotindex = len(patch)
        patch = float(patch[:dotindex])

        for frame in data['frames']:

            total_gold = [participant["totalGold"] for participant in frame['blueTeam']['participants'] + frame['redTeam'][
                'participants']]
            kda = [[participant["kills"],participant["deaths"],participant["assists"]] for participant in frame[
                'blueTeam']['participants'] + frame['redTeam']['participants']]
            cs = [participant['creepScore'] for participant in frame[
                'blueTeam']['participants'] + frame['redTeam']['participants']]
            lvl = [participant['level'] for participant in frame[
                'blueTeam']['participants'] + frame['redTeam']['participants']]
            max_healths = [participant['maxHealth'] for participant in frame[
                'blueTeam']['participants'] + frame['redTeam']['participants']]
            current_healths = [participant['currentHealth'] for participant in frame[
                'blueTeam']['participants'] + frame['redTeam']['participants']]
            towers = [frame['blueTeam']['towers'],frame['redTeam']['towers']]
            frame_timestamp = dt.parse(frame['rfc460Timestamp'])
            frame_timestamp_seconds = frame_timestamp.timestamp()
            if np.sum(max_healths) == 0:
                self.last_baron_slain_time = frame_timestamp_seconds
                self.last_elder_slain_time = frame_timestamp_seconds
                self.last_dragon_slain_time = frame_timestamp_seconds
                self.start_time = frame_timestamp_seconds
            game_time = frame_timestamp_seconds - self.start_time

            dragon_soul_blue = [0,0,0,0]
            dragon_soul_red = [0,0,0,0]

            baron_left = 0
            elder_left = 0
            if self.baron_active[0] or self.baron_active[1]:
                elapsed = frame_timestamp_seconds - self.baron_start_timer
                blue_all_dead = np.sum(current_healths[:5]) == 0
                red_all_dead = np.sum(current_healths[5:]) == 0
                if elapsed > game_constants.BARON_DURATION or \
                    self.baron_active[0] and blue_all_dead or \
                    self.baron_active[1] and red_all_dead:
                    self.baron_active = [0, 0]
                else:
                    baron_left = game_constants.BARON_DURATION - elapsed
            if self.elder_active[0] or self.elder_active[1]:
                elapsed = frame_timestamp_seconds - self.elder_start_timer
                blue_all_dead = np.sum(current_healths[:5]) == 0
                red_all_dead = np.sum(current_healths[5:]) == 0
                if elapsed > game_constants.ELDER_DURATION or \
                    self.elder_active[0] and blue_all_dead or \
                    self.elder_active[1] and red_all_dead:
                    self.elder_active = [0, 0]
                else:
                    elder_left = game_constants.ELDER_DURATION - elapsed

            dragon_countdown = game_constants.DRAGON_RESPAWN_TIMER - (frame_timestamp_seconds -
                                                                      self.last_dragon_slain_time)
            if dragon_countdown < 0:
                dragon_countdown = 0

            if game_time - game_constants.BARON_INIT_SPAWN < 0:
                baron_countdown = game_constants.BARON_INIT_SPAWN - game_time
            else:
                baron_countdown = game_constants.BARON_RESPAWN_TIMER - (frame_timestamp_seconds - self.last_baron_slain_time)

            if game_time - game_constants.ELDER_INIT_SPAWN < 0:
                elder_countdown = game_constants.ELDER_INIT_SPAWN - game_time
            else:
                elder_countdown = game_constants.ELDER_RESPAWN_TIMER - (frame_timestamp_seconds -
                                                                        self.last_elder_slain_time)
            if baron_countdown < 0:
                baron_countdown = 0
            if elder_countdown < 0:
                elder_countdown = 0

            dragons_killed = np.zeros((2,4))
            blue_team_dragons = frame['blueTeam']['dragons']
            red_team_dragons = frame['redTeam']['dragons']

            if "elder" in set(blue_team_dragons + red_team_dragons):
                if blue_team_dragons != self.prev_blue_team_dragons:
                    self.elder_active = [1, 0]
                    self.elder_start_timer = frame_timestamp_seconds
                    self.last_elder_slain_time = frame_timestamp_seconds
                elif red_team_dragons != self.prev_red_team_dragons:
                    self.elder_active = [0, 1]
                    self.elder_start_timer = frame_timestamp_seconds
                    self.last_elder_slain_time = frame_timestamp_seconds
                self.prev_blue_team_dragons = blue_team_dragons
                self.prev_red_team_dragons = red_team_dragons
                blue_team_dragons = list(filter(lambda a: a != "elder", blue_team_dragons))
                red_team_dragons = list(filter(lambda a: a != "elder", red_team_dragons))
            else:
                if blue_team_dragons != self.prev_blue_team_dragons or red_team_dragons != self.prev_red_team_dragons:
                    self.last_dragon_slain_time = frame_timestamp_seconds
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
                self.baron_start_timer = frame_timestamp_seconds
                self.last_baron_slain_time = frame_timestamp_seconds

            self.prev_baron_active_raw = baron_active

            champs = [ChampManager().lookup_by("name", chname)["int"] for chname in champ_names]

            kda = np.ravel(kda)
            input_ = {
                          "champs": champs,
                          "total_gold": total_gold,
                          "blue_side": [1, 0],
                          "cs": cs,
                          "lvl": lvl,
                          "kills": kda[0::3],
                          "deaths": kda[1::3],
                          "assists": kda[2::3],
                          "turrets_destroyed": towers,
                            "team_odds": odds,
                            "max_health": max_healths,
                            "current_health": current_healths,

                            "baron_countdown": baron_countdown,
                            "baron": self.baron_active,
                            "baron_time_left": baron_left,

                            "dragon_countdown": dragon_countdown,
                            "dragons_killed": np.ravel(dragons_killed),
                            "dragon_soul_type": dragon_soul_blue + dragon_soul_red,

                            "elder_countdown": elder_countdown,
                            "elder": self.elder_active,
                            "elder_time_left": elder_left,

            }
            try:
                x = InputWinPred.dict2vec(input_)
            except AssertionError:
                print("some vals were negative")
                raise
            x_meta = np.array([int(data['esportsGameId']), frame_timestamp.timestamp(), patch], dtype=np.float32)
            yield x, x_meta


    async def game2vec(self, gameId):
        # game_snapshots = self.generate_live_feed(gameId)
        game_snapshots = await self.download_finished_game(gameId)
        result = [list(self.snapshot2vec(snapshot, [0, 0])) for snapshot in game_snapshots]
        vec = np.array([y[0] for x in result for y in x], dtype=np.float32)
        meta = np.array([y[1] for x in result for y in x], dtype=np.uint64)
        return vec, meta
        # while True:
        #     try:
        #         snapshot = next(game_snapshots)
        #         vec = list(self.snapshot2vec(snapshot, [0, 0]))
        #         result.append(vec)
        #     except StopIteration:
        #         vec = np.array([y[0] for x in result for y in x], dtype=np.float32)
        #         meta = np.array([y[1] for x in result for y in x], dtype=np.uint64)
        #         return vec,meta


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
        gameId = 104841804589544540
        odds = (1.35, 2.8)
        swap_odds = False
        snapshots = s.generate_live_feed(gameId)

        urls = [
            'https://sports.betway.com/en/sports/evt/6416568',
            'https://www.betfair.com/sport/e-sports/lol-world-championship/top-esports-dragonx/30027771',
            #     'https://sports.williamhill.com/betting/en-gb/e-sports/OB_EV18616680/rainbow7-vs-lgd-gaming-bo5'
        ]
        css_scrapers = [
            lambda: drivers[0].find_elements_by_xpath('//span[contains(text(),'
                                                      '"Match Winner")]/../../../following-sibling::div//div[@class="odds"]'),
            lambda: drivers[1].find_elements_by_xpath('//span[contains(@class, "ui-runner-price")]'),

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
        import pandas as pd
        champ_comp = pd.read_csv('training_data/win_pred/competitive_champs_norm.csv')
        champ_comp = champ_comp.to_numpy()
        champ_comp = {ChampManager().lookup_by('name', c[1])['id']: c[2:] for c in champ_comp}
        champ_wrs = np.load("training_data/win_pred/champ_wr_by_patch.npz")
        champ_early_late = np.load("training_data/win_pred/champ_early_late.npz")
        champ_syns = np.load("training_data/win_pred/syn_stats.npz")
        champ_counters = np.load("training_data/win_pred/counter_stats.npz")

        model = WinPredModel("standard", network_config=network_config)
        model2 = WinPredModel("standard", network_config=network_config2)
        model.load_model()
        model2.load_model()
        logging.getLogger("urllib3").setLevel(logging.CRITICAL)
        logging.getLogger("requests").setLevel(logging.CRITICAL)
        new_data = None
        while True:

            try:
                snapshot = next(snapshots)
                x = list(self.snapshot2vec(snapshot, odds))
                x = np.array([a[0] for a in x])

                self.check_events(x)

                if new_data is None:
                    new_data1 = get_team_stats(x[0], 0, champ_comp, champ_counters, champ_syns, champ_wrs,
                                               champ_early_late, 19)
                    new_data2 = get_team_stats(x[0], 1, champ_comp, champ_counters, champ_syns, champ_wrs,
                                               champ_early_late, 19)
                    new_data = np.concatenate([new_data1, new_data2], axis=0)

                x[:, InputWinPred.indices['start']['champ_wr']:InputWinPred.indices['end']['champ_wr']] = np.tile(new_data, [x.shape[0],1])
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
                    print("\tYou should bet on BLUE. Expected payout: " + str(scraped_odds_blue[i] * pred_odds_blue * 0.95)
                          + f" scaled payout: {acc * scraped_odds_blue[i] * pred_odds_blue}")
                else:
                    print("\tYou should bet on RED. Expected payout: " + str(scraped_odds_red[i] * (1-pred_odds_blue) * 0.95)
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
        meta_x = []
        error_matches = set()
        for gameId, game_result in zip(gameIds, game_results):
            self.reset()
            try:
                data_x,meta = await self.game2vec(gameId)
                # data_x = np.array([list(frame_gen) for ss_gen in data_x for frame_gen in ss_gen], dtype=np.uint64)
                data_x = np.reshape(data_x, (-1, InputWinPred.len))
                # if champs is not None:
                #     data_x[:,Input.indices["start"]["champs"]:Input.indices["end"]["champs"]] = champs
                if game_result == [0]:
                    data_x = InputWinPred.flip_teams(data_x)
                x.append(data_x)
                meta_x.append(meta)
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
        try:
            np.save("tmp1.npz", x)
            np.save("tmp2.npz", meta_x)
        except:
            pass
        x = [a for a in x if a.shape[0]!=0]
        meta_x = [a for a in meta_x if a.shape[0]!=0]
        x = np.concatenate(x, axis=0)
        meta_x = np.concatenate(meta_x, axis=0)
        return x, meta_x

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


def get_team_stats(game_row, side, champ_comp, champ_counters, champ_syns, champ_wrs, champ_early_late, patch):

    start_index = InputWinPred.indices["start"]["champs"]
    end_index = InputWinPred.indices["half"]["champs"]
    opp_start_index = InputWinPred.indices["half"]["champs"]
    opp_end_index = InputWinPred.indices["end"]["champs"]

    new_data = []
    if side == 1:
        start_index, opp_start_index = opp_start_index, start_index
        end_index, opp_end_index = opp_end_index, end_index

    for lane, champ in enumerate(game_row[start_index:end_index]):
        champ_id = int(ChampManager().lookup_by("int", champ)["id"])
        new_data.extend(champ_comp[str(champ_id)])
        try:
            new_data.extend(champ_wrs[str((champ_id, patch, lane))])
        except KeyError:
            new_data.extend([0.0,0.0])
        new_data.extend(champ_early_late[str(champ_id)])
        for team_champ in game_row[start_index:end_index]:
            if team_champ == champ:
                continue
            team_champ_id = int(ChampManager().lookup_by("int", team_champ)["id"])
            new_data.append(float(champ_syns[str((champ_id, team_champ_id))]))
        for opp_lane, opp_champ in enumerate(game_row[opp_start_index:opp_end_index]):
            opp_champ_id = int(ChampManager().lookup_by("int", opp_champ)["id"])
            if opp_lane == lane:
                new_data.append(champ_counters[str((champ_id, opp_champ_id))][0])
            else:
                new_data.append(champ_counters[str((champ_id, opp_champ_id))][1])
    return np.ravel(new_data)


def build_champ_wr_table():

    ["champ", "patch", "lane", "num_games", "won", "num_length0_15", "num_length15_20",
     "num_length20_25", "num_length25_30", "num_length30_35", "num_length35_40", "num_length40+",
     "won_length0_15", "won_length15_20",
     "won_length20_25", "won_length25_30", "won_length30_35", "won_length35_40", "won_length40+",
     "top_wr"]


    champ_i = 0
    lane_i = 2
    patch_i = 1

    champ_early_late = dict()
    champ_wr_by_patch = dict()

    champs_stats = np.load("champ_stats.npz", allow_pickle=True)['arr_0']
    import pandas as pd
    champ_comp_stats = pd.read_csv('training_data/win_pred/competitive_champs.csv')
    for col in ['KDA', 'CSM', 'DPM', 'CSD@15', 'WR']:
        champ_comp_stats[col] = (champ_comp_stats[col] - champ_comp_stats[col].mean()) / champ_comp_stats[col].std()
    champ_comp_stats.to_csv('training_data/win_pred/competitive_champs_norm.csv')

    num_games_i_champ_stats = 3
    num_wins_i_champ_stats = 4
    cum_by_patch = dict()
    champ_total_wr = dict()
    for row in champs_stats:
        champ = row[champ_i]
        if champ not in champ_early_late:
            champ_early_late[champ] = np.array([0]*14)
        champ_early_late[champ] += np.array(row[6:20])
        key = champ,row[patch_i], row[lane_i]
        patch_wins = row[num_wins_i_champ_stats]
        patch_games = row[num_games_i_champ_stats]
        cum_key = champ,row[lane_i]
        if cum_key not in cum_by_patch:
            cum_by_patch[cum_key] = np.array([1,1])
        if champ not in champ_total_wr:
            champ_total_wr[champ] = np.array([0,0])
        if key not in champ_wr_by_patch:
            cum_wr = cum_by_patch[champ, row[lane_i]] + np.array([patch_wins, patch_games])
            cum_wr = cum_wr[0]/cum_wr[1]
            champ_wr = patch_wins/patch_games
            champ_wr_by_patch[key] = [champ_wr, cum_wr]
        champ_total_wr[champ] += np.array([patch_wins,patch_games])
        cum_by_patch[cum_key] += np.array([patch_wins, patch_games])
    champ_wr_by_patch = {str(k):(np.array(v)-0.5)/0.1 for k,v in champ_wr_by_patch.items()}
    champ_total_wr = {k:v[0]/v[1] for k,v in champ_total_wr.items()}


    champ_early_late = build_early_late_stats(champ_early_late, champ_total_wr)

    syn_stats, counter_stats = build_syn_counter_stats(champ_total_wr)


    # champ_early_late_pd = pd.Dataframe.from_dict(champ_early_late)
    # champ_early_late_pd.to_csv('champ_early_late')
    with open(f'champ_early_late.npz', 'wb') as fp:
        np.savez_compressed(fp, **champ_early_late)
    with open(f'syn_stats.npz', 'wb') as fp:
        np.savez_compressed(fp, **syn_stats)
    with open(f'counter_stats.npz', 'wb') as fp:
        np.savez_compressed(fp, **counter_stats)
    with open(f'champ_wr_by_patch.npz', 'wb') as fp:
        np.savez_compressed(fp, **champ_wr_by_patch)
    print('all done')


def build_early_late_stats(champ_early_late, champ_total_wr):
    for champ in champ_early_late:
        wrs = champ_early_late[champ][7:] / champ_early_late[champ][:7]
        wrs_relative = wrs - champ_total_wr[champ]
        champ_early_late[champ] = wrs_relative

    champ_early_late_arr = np.array(list([[k, *v] for k, v in champ_early_late.items()]))
    champ_ids = champ_early_late_arr[:, :1]
    early_late = champ_early_late_arr[:, 1:]
    early_late_mean = np.mean(early_late[:, 1:])
    early_late_std = np.std(early_late[:, 1:])
    early_late = [(v - early_late_mean) / early_late_std for v in early_late]
    champ_early_late = np.concatenate([champ_ids, early_late], axis=1)
    champ_early_late = {str(int(c[0])): c[1:] for c in champ_early_late}
    return champ_early_late


def relative_counter_stats(result_stats, champ_total_wrs):
    result = []
    for champ_id, champ in result_stats.items():
        current_champ_diffs_l = []
        current_champ_diffs_nl = []
        current_champ_opp_ids = []
        current_champ_ids = []
        for opp_id, team_champ in result_stats[champ_id].items():
            try:
                expected_wr = champ_total_wrs[champ_id] / (champ_total_wrs[champ_id] + champ_total_wrs[
                    opp_id])
            except KeyError:
                continue
            actual_wr_lane_opp = result_stats[champ_id][opp_id][0][1] / result_stats[champ_id][
                opp_id][0][0]
            actual_wr_non_lane_opp = result_stats[champ_id][opp_id][1][1] / result_stats[champ_id][
                opp_id][1][0]
            diff_lane_opp = actual_wr_lane_opp - expected_wr
            diff_non_lane_opp = actual_wr_non_lane_opp - expected_wr
            current_champ_diffs_l.append(diff_lane_opp)
            current_champ_diffs_nl.append(diff_non_lane_opp)
            current_champ_opp_ids.append(opp_id)
            current_champ_ids.append(champ_id)

        current_champ_diffs_l = (current_champ_diffs_l - np.mean(current_champ_diffs_l)) / np.std(current_champ_diffs_l)
        current_champ_diffs_nl = (current_champ_diffs_nl - np.mean(current_champ_diffs_nl)) / np.std(
            current_champ_diffs_nl)
        new_data = np.transpose([current_champ_ids, current_champ_opp_ids, current_champ_diffs_l, current_champ_diffs_nl], [1, 0])
        result.append(new_data)

    result = np.concatenate(result, axis=0)
    result_dict = {str((int(my_champ), int(opp_champ))): (wr_lane, wr_nl) for my_champ, opp_champ, wr_lane,
                                                                       wr_nl in result}
    return result_dict


def build_syn_counter_stats(champ_total_wrs):
    syn_stats = np.load("syn_stats.npz", allow_pickle=True)['arr_0']
    counter_stats = np.load("counter_stats.npz", allow_pickle=True)['arr_0']
    syn_stats_aggregated = dict()
    counter_stats_aggregated = dict()
    champ_i = 0
    opp_champ_i = 2
    num_games_i = 5
    num_wins_i = 6
    lane_champ1_i = 3
    lane_champ2_i = 4

    for stats, result_stats in zip([syn_stats, counter_stats], [syn_stats_aggregated, counter_stats_aggregated]):
        for row in stats:
            champ = row[champ_i]
            opp_champ = row[opp_champ_i]
            if champ not in result_stats:
                result_stats[champ] = dict()
            if opp_champ not in result_stats[champ]:
                result_stats[champ][opp_champ] = np.array([[1, 1], [1, 1]])
            if row[lane_champ1_i] == row[lane_champ2_i]:
                result_stats[champ][opp_champ][0] += np.array([row[num_games_i], row[num_wins_i]])
            else:
                result_stats[champ][opp_champ][1] += np.array([row[num_games_i], row[num_wins_i]])
    syn_stats = relative_syn_stats(syn_stats_aggregated, champ_total_wrs)
    counter_stats = relative_counter_stats(counter_stats_aggregated, champ_total_wrs)
    return syn_stats, counter_stats


def relative_syn_stats(result_stats, champ_total_wrs):
    result = []

    for champ_id, champ in result_stats.items():
        current_champ_diffs = []
        current_champ_team_ids = []
        current_champ_ids = []
        for team_champ_id, team_champ in result_stats[champ_id].items():
            try:
                expected_wr = (champ_total_wrs[champ_id] + champ_total_wrs[team_champ_id])/2
            except KeyError:
                continue
            actual_wr = result_stats[champ_id][team_champ_id][1][1] / result_stats[champ_id][team_champ_id][1][0]
            diff = actual_wr - expected_wr
            current_champ_diffs.append(diff)
            current_champ_team_ids.append(team_champ_id)
            current_champ_ids.append(champ_id)

        current_champ_diffs = (current_champ_diffs - np.mean(current_champ_diffs))/np.std(current_champ_diffs)
        new_data = np.transpose([current_champ_ids, current_champ_team_ids, current_champ_diffs], [1,0])
        result.append(new_data)

    result = np.concatenate(result, axis=0)
    result_dict = {str((int(my_champ), int(team_champ))):wr for my_champ, team_champ, wr in result}
    return result_dict


def fold_in_champ_wrs():
    import pandas as pd
    champ_comp = pd.read_csv('training_data/win_pred/competitive_champs_norm.csv')
    champ_comp = champ_comp.to_numpy()
    champ_comp = {ChampManager().lookup_by('name',c[1])['id']:c[2:] for c in champ_comp}
    champ_wrs = np.load("training_data/win_pred/champ_wr_by_patch.npz")
    champ_early_late = np.load("training_data/win_pred/champ_early_late.npz")
    champ_syns = np.load("training_data/win_pred/syn_stats.npz")
    champ_counters = np.load("training_data/win_pred/counter_stats.npz")

    X = np.load("training_data/win_pred/train_new_winpred_odds.npz")['arr_0']
    meta = np.load("training_data/win_pred/train_new_meta.npz")['arr_0']
    gameids = meta[:, 0]
    patches = meta[:,2]
    gameids2champwr = dict()
    new_X = []
    while X.shape[0] > 0:
        game_row = X[0]
        game_id = gameids[0]
        patch = patches[0]
        if str(game_id) in gameids2champwr:
            new_data = gameids2champwr[str(game_id)]
        else:
            new_data1 = get_team_stats(game_row, 0, champ_comp, champ_counters, champ_syns, champ_wrs,
                                       champ_early_late, patch)
            new_data2 = get_team_stats(game_row, 1, champ_comp, champ_counters, champ_syns, champ_wrs,
                                       champ_early_late, patch)
            new_data = np.concatenate([new_data1,new_data2],axis=0)
            gameids2champwr[str(game_id)] = new_data
        # new_row = np.concatenate([game_row,new_data], axis=0)
        # new_X.append(new_row)
        X = X[1:]
        gameids = gameids[1:]
        patches = patches[1:]

    with open("training_data/win_pred/wrs_dict.npz", "wb") as writer:
        np.savez_compressed(writer, **gameids2champwr)




async def run_async():
    s = ScrapeEsportsData()
    await s.scrape_games()
    # a = s.fetch_payload(url="https://feed.lolesports.com/livestats/v1/window/104174613353783764")

    # x = await s.game2vec(104174613333794954)
    # print(x)
    # s.scrape_stats("bjergsen", "summer")


if __name__ == "__main__":


    # build_champ_wr_table()
    # fold_in_champ_wrs()
    s = ScrapeEsportsData()
    # s.scrape_games()
    # s = ScrapePlayerStats()
    # s.scrape_all_stats()

    # s = ScrapeEsportsData()
    # s.scrape_all_stats()
    # print('hi')
    # gameIds = [104242514815381919]
    # game_results = [0]

    # $x('string(//tr[contains(@class,"multirow-highlighter")])')
    # a =$x("//div[@class='EventMatch']/a/@href")
    # for(var i=0;i<200;++i)
    #     console.log(a[i].textContent)


    s.live_game2odds(0)

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
