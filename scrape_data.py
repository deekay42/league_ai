import json

import arrow
import numpy as np
import traceback
import cass_configured as cass



def get_match_ids(summoners):
    # with open("matchids", "r") as f:
    #     match_ids = f.readlines()
    # match_ids = [x.strip() for x in match_ids]
    # match_ids = set(match_ids)

    match_ids = set()

    with open("scrape/matchids", "w") as f:
        f.write('[\n')
        first = True
        for summoner in summoners:
            try:
                summ_match_hist = summoner.match_history(queues={Queue.ranked_solo_fives},
                                                         begin_time=arrow.Arrow(2019, 4, 7, 0, 0, 0))
                for match in summ_match_hist:
                    if first:
                        first = False
                    else:
                        f.write(',')
                    match_id = match.id
                    if match_id in match_ids:
                        continue
                    match_ids.add(match_id)
                    f.write(str(match_id) + '\n')
                f.flush()
            except HTTPError:
                print('ERROR: There was an error obtaining this summoners match history')
        f.write(']\n')
    return match_ids


def get_top_summoners(num):
    elite_summoners = cass.get_challenger_league(Queue.ranked_solo_fives, 'KR').entries + cass.get_master_league(
        Queue.ranked_solo_fives, 'KR').entries
    with open("res/diamond_league_ids_short") as f:
        leagues = f.readlines()
    leagues = [x.strip() for x in leagues]
    high_dia_summoners = []
    for league in leagues:
        league = cass.core.league.League(id=league, region="KR")
        summoners = league.entries.filter(lambda x: x.division == Division.one)
        high_dia_summoners += summoners
    high_dia_summoners.sort(key=lambda x: x.league_points, reverse=True)
    league_entry_result = elite_summoners + high_dia_summoners#[:min(len(high_dia_summoners), num - len(elite_summoners))]
    league_entry_result = league_entry_result[:min(len(league_entry_result), num)]
    summoner_result = [league_entry.summoner for league_entry in league_entry_result]
    return summoner_result

def sortIfComplete(team):

    team_positions = [participant["lane"] for participant in team]

    bot_first_cs = -1
    bot_first_i = -1

    #break bot into adc and sup
    pos2summ = dict()
    for i, summ in enumerate(team):
        if summ["lane"] == "bot":
            if summ["minionsKilled"] > bot_first_cs:
                summ["lane"] = "adc"
                if bot_first_cs != -1:
                    team[bot_first_i]["lane"] = "sup"
                    pos2summ["sup"] = pos2summ.pop("adc")
                else:
                    bot_first_i = i
                bot_first_cs = summ["minionsKilled"]
            else:
                summ["lane"] = "sup"
        pos2summ[summ["lane"]] = summ


    sort_order = ["top", "jg", "mid", "adc", "sup"]

    if ("None" in team_positions or (
            sorted(team_positions) != ["bot", "bot", "jg", "mid", "top"])):
        return team, False

    result = []
    for position in sort_order:
        result.append(pos2summ[position])
    return result, True


def get_matches(match_ids):
    lane2str = {Lane.bot_lane: "bot", Lane.jungle: "jg", Lane.mid_lane: "mid",
                Lane.top_lane: "top", None: "None"}

    first = True
    with open("scrape/matches", "w") as f:
        f.write('[\n')

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


                teams = []
                for team in [winning_team, losing_team]:
                    out_team = []
                    for participant in team.participants:
                        summ_dict = {"participantId": participant.id, "championId": participant.champion.id, "spell1Id": participant.summoner_spell_d.id,
                                     "spell2Id": participant.summoner_spell_f.id,
                                     "kills": participant.stats.kills, "deaths": participant.stats.deaths,
                                     "assists": participant.stats.assists,
                                     "earned": participant.stats.gold_earned,
                                     "minionsKilled": participant.stats.total_minions_killed,
                                     "neutralMinionsKilled": participant.stats.neutral_minions_killed,
                                     "wardsPlaced": participant.stats.wards_placed,
                                     "level": participant.stats.level, "lane": lane2str[participant.timeline.lane]}
                        out_team.append(summ_dict)
                    teams.append(out_team)

                winning_team, winning_team_sorted = sortIfComplete(teams[0])
                losing_team, losing_team_sorted = sortIfComplete(teams[1])
                teams = np.ravel([winning_team, losing_team]).tolist()


                events = []
                for frame in match.timeline.frames:
                    for event in frame.events:
                        event_type = event.type
                        if event_type == "ITEM_DESTROYED" or event_type == "ITEM_PURCHASED" or event_type == "ITEM_SOLD":
                            if event.item_id != 2055 and event.item_id != 3340 and event.item_id != 3341 and event.item_id != 3363 and event.item_id != 3364:
                                events.append(
                                    {"participantId": event.participant_id, "itemId": event.item_id, "type": event_type,
                                     "timestamp": int(event.timestamp.total_seconds() * 1000)})
                        elif event_type == "ITEM_UNDO":
                            if event.after_id != 3340 and event.after_id != 2055 and event.after_id != 3341 and event.after_id != 3363 and event.after_id != 3364:
                                events.append(
                                    {"participantId": event.participant_id, "beforeId": event.before_id,
                                     "afterId": event.after_id, "type": event_type,
                                     "timestamp": int(event.timestamp.total_seconds() * 1000)})

                teams_sorted = "1,2" if winning_team_sorted and losing_team_sorted else "1" if winning_team_sorted else "2" if losing_team_sorted else "0"
                f.write(json.dumps({"gameId": match_id, "sorted": teams_sorted, "participants": teams, "itemsTimeline": events}, separators=(',', ':')))
                f.flush()
            except HTTPError as e:
                print('HTTP ERROR: There was an error obtaining this match. Skip.')
                print(repr(e))
                print(traceback.format_exc())
            except Exception as e:
                print('ERROR: There was an error obtaining this match. Skip.')
                print(repr(e))
                print(traceback.format_exc())
        f.write(']')


def scrape_matches():
    summoners = get_top_summoners(10)
    with open("scrape/summoner_account_ids", "w") as f:
        f.write('[\n')
        first = True
        for summoner in summoners:
            if first:
                first = False
            else:
                f.write(',')
            f.write(str(summoner.account_id) + '\n')
        f.write(']\n')

    # with open("scrape/summoner_account_ids", encoding='utf-8') as f:
    #    summoners = f.readlines()
    # summoners = [Summoner(account=int(x.strip()), region='KR') for x in summoners]

    match_ids = get_match_ids(summoners)



    # with open("scrape/matchids", "r") as f:
    #    match_ids = json.load(f)

    get_matches(match_ids)





if __name__ == "__main__":
    scrape_matches()
