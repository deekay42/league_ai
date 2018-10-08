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

def get_match_ids(summoners):
    # with open("matchids", "r") as f:
    #     match_ids = f.readlines()
    # match_ids = [x.strip() for x in match_ids]
    # match_ids = set(match_ids)

    match_ids = set()

    with open("scrape/matchids", "w") as f:
        f.write('[')
        first = True
        for summoner in summoners:
            try:
                summ_match_hist = summoner.match_history(queues={Queue.ranked_solo_fives},
                                                                  begin_time=arrow.Arrow(2018, 9, 16, 0, 0, 0))
                for match in summ_match_hist:
                    if first:
                        first = False
                    else:
                        f.write(',')
                    match_id = match.id
                    if match_id in match_ids:
                        continue
                    match_ids.add(match_id)
                    f.write(str(match_id)+'\n')
                f.flush()
            except HTTPError:
                print('ERROR: There was an error obtaining this summoners match history')
        f.write(']')
    return match_ids

def get_top_summoners(num):
    elite_summoners = cass.get_challenger_league(Queue.ranked_solo_fives, 'KR').entries + cass.get_master_league(Queue.ranked_solo_fives, 'KR').entries
    with open("scrape/diamond_league_ids") as f:
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
    predictor = PredictRoles()
    lane2role = {Lane.bot_lane: Role.adc, Lane.jungle: Role.jungle, Lane.mid_lane: Role.middle,
     Lane.top_lane: Role.top}


    # summoners = get_top_summoners(10000)
    # with open("scrape/summoner_account_ids", "w") as f:
    #     f.write('[')
    #     first = True
    #     for summoner in summoners:
    #         if first:
    #             first = False
    #         else:
    #             f.write(',')
    #         f.write(str(summoner.summoner.account.id)+'\n')
    #     f.write(']')


    # with open("scrape/summoner_account_ids", encoding='utf-8') as f:
    #     summoners = f.readlines()
    # summoners = [Summoner(account=int(x.strip()), region='KR') for x in summoners]


    # match_ids = get_match_ids(summoners)
    #



    with open("scrape/matchids_left_final2", "r") as f:
        match_ids = json.load(f)
    first = True
    with open("scrape/matches3", "w") as f:
        f.write('[')

        for match_id in match_ids:
            try:
                if first:
                   first = False
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

                winning_team_positions = [participant.timeline.lane for participant in winning_team.participants]
                losing_team_positions = [participant.timeline.lane for participant in losing_team.participants]

                team_roles = [-1,-1]

                if not (None not in winning_team_positions and None not in losing_team_positions and (sorted(winning_team_positions, key=lambda x: x.value) == [Lane.bot_lane, Lane.bot_lane, Lane.jungle, Lane.mid_lane,
                                                           Lane.top_lane] == sorted(
                        losing_team_positions, key=lambda x: x.value))):
                    roles_winning_team = get_team_positions(winning_team, predictor)
                    roles_losing_team = get_team_positions(losing_team, predictor)
                else:
                    for i, (team_positions, team) in enumerate(zip([winning_team_positions, losing_team_positions], [winning_team, losing_team])):
                        team_roles[i] = [lane2role[pos] for pos in team_positions]
                        first_adc_i = team_roles[i].index(Role.adc)
                        second_adc_i = len(team_roles[i]) - team_roles[i][-1::-1].index(Role.adc) - 1
                        if team.participants[first_adc_i].stats.total_minions_killed > team.participants[second_adc_i].stats.total_minions_killed:
                            team_roles[i][second_adc_i] = Role.support
                        else:
                            team_roles[i][first_adc_i] = Role.support
                    roles_winning_team = dict(zip(team_roles[0], [participant.champion.id for participant in winning_team.participants]))
                    roles_losing_team = dict(zip(team_roles[1], [participant.champion.id for participant in losing_team.participants]))

                teams = []
                for position in sort_order:
                    teams.append({"championId": roles_winning_team[position],
                              "participantId": champ2participant[roles_winning_team[position]]})
                for position in sort_order:
                    teams.append({"championId": roles_losing_team[position],
                              "participantId": champ2participant[roles_losing_team[position]]})
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
                print('HTTP ERROR: There was an error obtaining this match. Skip.')
            except Exception:
                print('ERROR: There was an error obtaining this match. Skip.')
        f.write(']')


if __name__ == "__main__":
    main()
