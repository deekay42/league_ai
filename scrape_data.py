import cassiopeia as cass
from cassiopeia import *
import json
import arrow
from predict import PredictRoles
import numpy as np

from cassiopeia.datastores.common import HTTPError

config = cass.get_default_config()
config['pipeline']['ChampionGG'] = {
    "package": "cassiopeia_championgg",
    "api_key": "496b4c2f287421a51a41aeb51a808f74"  # Your api.champion.gg API key (or an env var containing it)
  }

config['pipeline']['RiotAPI'] = {
    "api_key": "RGAPI-39e77a14-355a-4826-8cfb-6f5cc68bdc7c",
    "limiting_share": 1.0,
    "request_error_handling": {
        "404": {
            "strategy": "throw"
        },
        "429": {
            "service": {
                "strategy": "exponential_backoff",
                "initial_backoff": 1.0,
                "backoff_factor": 2.0,
                "max_attempts": 4
            },
            "method": {
                "strategy": "retry_from_headers",
                "max_attempts": 5
            },
            "application": {
                "strategy": "retry_from_headers",
                "max_attempts": 5
            }
        },
        "500": {
            "strategy": "exponential_backoff",
            "initial_backoff": 1.0,
            "backoff_factor": 2.0,
            "max_attempts": 4
        },
        "503": {
            "strategy": "exponential_backoff",
            "initial_backoff": 1.0,
            "backoff_factor": 2.0,
            "max_attempts": 4
        },
        "timeout": {
            "strategy": "exponential_backoff",
            "initial_backoff": 1.0,
            "backoff_factor": 2.0,
            "max_attempts": 4
        },
        "403": {
            "strategy": "throw"
        }
    }
  }


def get_team_positions(team, predictor):
    data = []
    for participant in team.participants:
        data.append([participant.champion.id, participant.summoner_spell_d.id, participant.summoner_spell_f.id,
                     participant.stats.kills, participant.stats.deaths, participant.stats.assists,
                     participant.stats.gold_earned, participant.stats.total_minions_killed,
                     participant.stats.neutral_minions_killed, participant.stats.wards_placed, participant.stats.level])
    data = np.array(data)
    champs = np.stack(data[:, 0])
    spells = np.ravel(np.stack(data[:, 1:3]))
    rest = np.ravel(np.stack(data[:, 3:]))


    return predictor.predict(champs, spells, rest)

cass.set_default_region("NA")
cass.apply_settings(config)

match = cass.get_match(3092968903, region="KR")
predictor = PredictRoles()
roles_winning_team = get_team_positions(match.blue_team, predictor)

def get_match_ids(summoners):
    # with open("matchids", "r") as f:
    #     match_ids = f.readlines()
    # match_ids = [x.strip() for x in match_ids]
    # match_ids = set(match_ids)

    match_ids = set()

    with open("matchids2", "w") as f:
        for summoner in summoners:
            try:
                summ_match_hist = summoner.match_history(queues={Queue.ranked_solo_fives},
                                                                  begin_time=arrow.Arrow(2018, 8, 1, 0, 0, 0))
                for match in summ_match_hist:
                    match_id = match.id
                    if match_id in match_ids:
                        continue
                    match_ids.add(match_id)
                    f.write(str(match_id)+'\n')
                f.flush()
            except HTTPError:
                print('ERROR: There was an error obtaining this summoners match history')
    return match_ids

def get_top_summoners(num):
    elite_summoners = cass.get_challenger_league(Queue.ranked_solo_fives, 'KR').entries + cass.get_master_league(Queue.ranked_solo_fives, 'KR').entries
    with open("res/diamond_league_ids") as f:
        leagues = f.readlines()
    leagues = [x.strip() for x in leagues]
    high_dia_summoners = []
    for league in leagues:
        league = cass.core.league.League(id=league, region="KR")
        summoners = league.entries.filter(lambda x: x.division == Division.one)
        high_dia_summoners += summoners
    high_dia_summoners.sort(key=lambda x: x.league_points, reverse=True)
    return elite_summoners + high_dia_summoners[:min(len(high_dia_summoners), num-len(elite_summoners))]


def main():
    sort_order = [Role.top, Role.jungle, Role.middle, Role.adc, Role.support]
    champion_roles = get_data()
    # summoners = get_top_summoners(10000)
    # with open("summoner_account_ids_2", "w") as f:
    #     for summoner in summoners:
    #         f.write(str(summoner.summoner.account.id)+'\n')
    # with open("cha_mas_dia1_account_ids", encoding='utf-8') as f:
    #     summoners = f.readlines()
    # summoners = [Summoner(account=int(x.strip()), region='KR') for x in summoners]
    # match_ids = get_match_ids(summoners)
    #
    # print("Number of matches found:"+str(len(match_ids)))
    with open("matchids_final_interrupted3", "r") as f:
        match_ids = f.readlines()
    match_ids = [int(x.strip()) for x in match_ids]
    first = True
    with open("matches4", "w") as f:
        f.write('[')

        for match_id in match_ids:
            try:
                if first:
                   first = True
                else:
                    f.write(',')
                match = cass.get_match(match_id, region="KR")
                champ2participant = dict()
                participants = match.red_team.participants + match.blue_team.participants
                for participant in participants:
                    champ_id = participant.champion.id
                    part_id = participant.id
                    champ2participant[champ_id] = part_id
                winning_team = match.blue_team
                losing_team = match.red_team
                if losing_team.win:
                    winning_team, losing_team = losing_team, winning_team

                winning_team_positions = [participant.timeline.lane for participant in winning_team]
                losing_team_positions = [participant.timeline.lane for participant in losing_team]

                if not (sorted(winning_team_positions) == sorted(losing_team_positions) == ["BOTTOM", "BOTTOM", "JUNGLE", "MIDDLE", "TOP"]):
                    roles_winning_team = get_team_positions(winning_team, predictor)
                    roles_losing_team = get_team_positions(losing_team, predictor)
                teams = []
                for position in sort_order:
                    teams.append({"championId": roles_winning_team[position],
                              "participantId": champ2participant[roles_winning_team[position].id]})
                for position in sort_order:
                    teams.append({"championId": roles_losing_team[position],
                              "participantId": champ2participant[roles_losing_team[position].id]})
                events = []
                for frame in match.timeline.frames:
                    for event in frame.events:
                        event_type = event.type
                        if event_type == "ITEM_DESTROYED" or event_type == "ITEM_PURCHASED" or event_type == "ITEM_SOLD":
                            if event.item_id != 2055 and event.item_id != 3340 and event.item_id != 3341 and event.item_id != 3363 and event.item_id != 3364:
                                events.append({"participantId": event.participant_id, "itemId": event.item_id, "type": event_type, "timestamp": int(event.timestamp.total_seconds()*1000)})
                        elif event_type == "ITEM_UNDO":
                            if event.after_id != 3340 and event.after_id != 2055 and event.after_id != 3341 and event.after_id != 3363 and event.after_id != 3364:
                                events.append(
                                    {"participantId": event.participant_id, "beforeId": event.before_id, "afterId": event.after_id, "type": event_type,
                                     "timestamp": int(event.timestamp.total_seconds()*1000)})


                f.write(json.dumps({"gameId": match_id, "participants": teams, "itemsTimeline": events}))
                f.flush()
            except HTTPError:
                print('ERROR: There was an error obtaining this match')
            finally:
                print('ERROR: Skipping this one')

        # f.seek(-1, f.tell())
        # f.write(']')


if __name__ == "__main__":
    match = cass.get_match(3102390845, region="KR")
    get_team_positions(match.blue_team, predictor)
