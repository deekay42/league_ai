import asyncio
import json

import dateutil.parser as dt
import requests
import scrape_esports_data
import cmath
import numpy as np
from constants import app_constants

match_urls = {
    "LCK": "https://oddspedia.com/api/v1/getMatchList?typeOfReturn=list&startDate=2020-01-01T00%3A00%3A00%2B02%3A00&endDate=2020-09-13T23%3A59%3A59%2B02%3A00&sport=league-of-legends&category=world&league=lck&geoCode=DE&inplay=0&popularLeaguesOnly=0&seasonId=68458&language=en",
    "LEC": "https://oddspedia.com/api/v1/getMatchList?typeOfReturn=list&startDate=2020-01-01T00%3A00%3A00%2B02%3A00&endDate=2020-09-13T23%3A59%3A59%2B02%3A00&sport=league-of-legends&category=world&league=lec&geoCode=DE&inplay=0&popularLeaguesOnly=0&seasonId=68458&language=en",
    "LCS": "https://oddspedia.com/api/v1/getMatchList?typeOfReturn=list&startDate=2020-01-01T00%3A00%3A00%2B02%3A00&endDate=2020-09-13T23%3A59%3A59%2B02%3A00&sport=league-of-legends&category=world&league=lcs&geoCode=DE&inplay=0&popularLeaguesOnly=0&seasonId=68458&language=en"
    }
single_odds_url = "https://oddspedia.com/api/v1/getSingleOdds?inplay=0&wettsteuer=0&geoCode=&language=en&matchId={matchId}&ot={mapId}"
matchinfo_url = "https://oddspedia.com/api/v1/getMatchInfo?inplay=0&wettsteuer=0&geoCode=&language=en&id={matchId}"
team_names = {
    "LCK": {"damwon": ["damwon"], "rolster": ["rolster"], "t1": ["t1"], "griffin": ["griffin"], "afreeca": [
        "afreeca"], "prince": ["prince"], "sandbox": ["sandbox"], \
            "hanwha": ["hanwha"], \
            "dynamics": ["dynamics"], "seora": ["seora"],
            "gen": ["gen"], "dragonx": ["dragonx", "drx"]},
    "LCS": {"100": ["100", "thieves"], "cloud": ["cloud"], "dignitas": ["dignitas"], "evil": ["evil",
                                                                                              "genius"],
            "fly": ["fly"], \
            "golden": ["golden"], \
            "liquid": ["liquid"], "tsm": ["tsm", "solomid"],
            "logic": ["logic", "clg"], \
            "immortal": ["immortal",
                         "imt"]},
    "LEC": {"g2": ["g2"], "rogue": ["rogue", "rng"], "fnatic": ["fnatic"], "04": ["04", "schalke"],
            "mad": ["mad"],
            "sk": ["sk"], "vitality": ["vitality"], "misfit": ["misfit"], \
            "excel": ["excel"], "origen": ["origen"]}
}

team_codes = {
    "LCK": {"dwg": ["damwon"], "kt": ["rolster"], "t1": ["t1"], "grf": ["griffin"], "af": [
        "afreeca"], "apk": ["prince"], "sp": ["prince"], "sb": ["sandbox"], \
            "hle": ["hanwha"], \
            "dyn": ["dynamics"], "srb": ["seora"],
            "gen": ["gen"], "drx": ["dragonx"]},
    "LCS": {"100": ["100t"], "c9": ["cloud"], "dig": ["dignitas"], "eg": ["evil"],
            "fly": ["fly"], \
            "gg": ["golden"], \
            "tl": ["liquid"], "tsm": ["tsm"],
            "clg": ["logic"], \
            "imt": ["immortal"]},
    "LEC": {"g2": ["g2"], "rge": ["rogue"], "fnc": ["fnatic"], "s04": ["04"],
            "mad": ["mad"],
            "sk": ["sk"], "vit": ["vitality"], "msf": ["misfit"], \
            "xl": ["excel"], "og": ["origen"]}
}

async def main():
    leagues = ["LCK", "LCS", "LEC"]
    event_ids_file_prefix = "event_ids_"

    event_ids = []
    for league in leagues:
        with open(app_constants.train_paths["win_pred"] + event_ids_file_prefix + league + ".json") as f:
            event_ids = json.load(f)
        game_ids2odds = await get_odds(event_ids, league)
        with open(f'game_ids2odds{league}.json', 'w') as fp:
            json.dump(game_ids2odds, fp)


def gameid2bluered(game_id:str, region:str):
    payload = scrape_esports_data.ScrapeEsportsData.fetch_payload(
        url=f"https://feed.lolesports.com/livestats/v1/window/{game_id}")
    blue_top_name = payload["gameMetadata"]["blueTeamMetadata"]['participantMetadata'][0]['summonerName']
    red_top_name = payload["gameMetadata"]["redTeamMetadata"]['participantMetadata'][0]['summonerName']
    blue_team_tag = blue_top_name[:blue_top_name.index(' ')].lower()
    red_team_tag = red_top_name[:red_top_name.index(' ')].lower()
    return team_codes[region][blue_team_tag][0], team_codes[region][red_team_tag][0]


async def get_odds(event_ids: list, region: str):
    s = scrape_esports_data.ScrapeEsportsData()
    res = await s.eventids2gameids(event_ids)
    match_id2game_ids = dict(zip(event_ids, res))

    schedule = s.get_schedule(region)
    matchid2dateteams = dict()
    for match in schedule:
        timestamp = dt.parse(match[1])
        team1name = match[2]
        team2name = match[3]
        try:
            ht, at = map_team_names(team1name, team2name, region)
        except AssertionError:
            print(f"Error: {team1name} {team2name} {timestamp} {match[0]}")
            raise
        val = (f"{timestamp.year}.{timestamp.month}.{timestamp.day}", ht, at)
        matchid = match[0]
        matchid2dateteams[matchid] = val
    odds_dateteams2matchid = get_dateteams2matchid(region)
    game_ids2odds = dict()
    odds_matchid2odds = get_odds_matchid2odds(odds_dateteams2matchid.values())
    # counter = 0
    for match_id, game_ids in match_id2game_ids.items():
        dateteams = matchid2dateteams[str(match_id)]
        odds_flipped = False
        try:
            try:
                odds_matchid = odds_dateteams2matchid[dateteams]
            except KeyError:
                try:
                    odds_flipped = True
                    odds_matchid = odds_dateteams2matchid[(dateteams[0], dateteams[2], dateteams[1])]
                except KeyError:
                    print(f"ACTUAL ERROR")
                    raise
            game_odds = odds_matchid2odds[odds_matchid]
        except KeyError:
            print(f"Error: Unable to find {dateteams}")
            game_odds = {'201': [(-2, -2)], '216': [(-2, -2)], '217': [(-2, -2)], '218': [(-2, -2)], '219': [(-2, -2)],
                         '220': [(-2, -2)]}
        # match_odds = game_odds['201']
        for game_id, game_odd in zip(game_ids, [game_odds['216'], game_odds['217'], game_odds['218'], game_odds['219'],
                                                game_odds['220']]):

            teams_flipped = False
            try:
                blue_team, red_team = gameid2bluered(game_id, region)
            except Exception:
                #we have to guess here...
                red_team = dateteams[2]
            if dateteams[1] == red_team:
                teams_flipped = True
            #xor
            if odds_flipped != teams_flipped:
                game_odd = [(go[1], go[0]) for go in game_odd]
            game_ids2odds[game_id] = game_odd
        # if counter == 3:
        #     return game_ids2odds
        # counter += 1
    return game_ids2odds


def map_team_names(team1name, team2name, region):
    ht = None
    at = None
    team1name = team1name.lower()
    team2name = team2name.lower()
    for team_name_key in team_names[region]:
        for team_name in team_names[region][team_name_key]:
            if ht is None and team_name.lower() in team1name:
                ht = team_name_key
            elif at is None and team_name.lower() in team2name:
                at = team_name_key
    assert ht is not None and at is not None
    return ht, at

def matchodds2gameodds(team1odds:float, team2odds:float) -> tuple:
    team1winprob = team2odds / (team1odds + team2odds)
    p = team1winprob
    team1winprob_game = (1 / 4) * (-(1 + 1j * cmath.sqrt(3)) * (-2 * p + 2 * cmath.sqrt((p - 1) * p) + 1) ** (1 / 3)
                                   + (1j * (cmath.sqrt(3) + 1j)) / (-2 * p + 2 * cmath.sqrt((p - 1) * p) + 1) ** (1 / 3) + 2)
    team1winprob_game = round(team1winprob_game.real,3)
    return 1/team1winprob_game, 1/(1-team1winprob_game)


def get_dateteams2matchid(region):
    match_ids_raw = json.loads(requests.get(url=match_urls[region]).text)
    dateteams2match_id = dict()
    for match in match_ids_raw["data"]["matchList"]:
        team1name = match['ht'].lower()
        team2name = match['at'].lower()
        timestamp = dt.parse(match['md'])
        try:
            ht, at = map_team_names(team1name, team2name, region)
        except AssertionError:
            print(f"Error in get_dateteams2matchid: {team1name} {team2name} {timestamp} {match['id']}")
            continue
        key = (f"{timestamp.year}.{timestamp.month}.{timestamp.day}", ht, at)
        dateteams2match_id[key] = match["id"]
    return dateteams2match_id


def get_odds_matchid2odds(match_ids: list) -> dict:
    matchid2odds = dict()
    counter = 0
    for match_id in match_ids:
        odds = dict()
        match_odds = json.loads(requests.get(url=single_odds_url.format(matchId=match_id, mapId="201")).text)
        try:
            match_odds = [(float(bookie['o1']), float(bookie['o2'])) for bookie in list(match_odds['data'].values())]
        except KeyError:
            match_odds = [(-1, -1)]
        # odds['201'] = match_odds
        for map_id in ["216", "217", "218", "219", "220"]:
            raw = requests.get(url=single_odds_url.format(matchId=match_id, mapId=map_id)).text
            payload = json.loads(raw)
            try:
                odds_by_map = [(float(bookie['o1']), float(bookie['o2'])) for bookie in list(payload['data'].values())]
            except KeyError:
                try:
                    odds_by_map = [matchodds2gameodds(np.mean(np.array(match_odds)[:,0]), np.mean(np.array(
                        match_odds)[:,1]))]
                except Exception:
                    odds_by_map = [(-1, -1)]

            odds[map_id] = odds_by_map
        matchid2odds[match_id] = odds
        # if counter == 3:
        #     return matchid2odds
        # counter += 1
    return matchid2odds

from train_model.input_vector import Input, InputWinPred

def fold_in_odds():
    leagues = ["LCK", "LCS", "LEC"]
    odds = dict()
    for league in leagues:
        with open("game_ids2odds" + league + ".json") as f:
            odds.update(json.loads(f.read()))

    X = np.load("training_data/win_pred/train_winpred.npz")['arr_0']
    gameids = np.load("training_data/win_pred/train_winpred_gameids.npz")['arr_0']
    # X = X[:1000]


    for i, (game_row, game_id) in enumerate(zip(X, gameids)):
        blue_team_first = game_row[InputWinPred.indices["start"]["blue_side"]:Input.indices["end"][
            "blue_side"]].tolist() == [1, 0]
        try:
            odds_for_side = (np.mean(np.array(odds[str(game_id)])[:, 0]), np.mean(np.array(odds[str(
                game_id)])[:,1]))
        except KeyError:
            print(f"no odds found for gameid: {game_id}")
            odds_for_side = [2, 2]
        if np.any(np.array(odds_for_side) < 0):
            odds_for_side = [2, 2]
        if not blue_team_first:
            odds_for_side = (odds_for_side[1], odds_for_side[0])
        game_row[InputWinPred.indices["start"]["team_odds"]:InputWinPred.indices["start"]["team_odds"] + 2] = \
            odds_for_side

    with open("training_data/win_pred/train_winpred_odds.npz", "wb") as writer:
        np.savez_compressed(writer, X)



if __name__ == "__main__":
    # fold_in_odds()
    asyncio.run(main())
