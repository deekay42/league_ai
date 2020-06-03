import json
import traceback
from urllib.error import HTTPError

import numpy as np
from range_key_dict import RangeKeyDict

from constants import game_constants, app_constants
from utils import cass_configured as cass
import arrow
import time
import itertools

class AllGamesCollected(Exception): pass

#
# def get_match_ids(summoners, num_games, cut_off_date):
#     # with open("matchids", "r") as f:
#     #     match_ids = f.readlines()
#     # match_ids = [x.strip() for x in match_ids]
#     # match_ids = set(match_ids)
#
#     match_ids = set()
#     first = True
#     with open(app_constants.train_paths["matchids"], "w") as f:
#         f.write("[\n")
#         for summoner in summoners:
#
#             try:
#                 summ_match_hist = summoner.match_history(queues={cass.Queue.ranked_solo_fives},
#                                                          begin_time=cut_off_date)
#                 for match in summ_match_hist:
#                     match_id = match.id
#                     if match_id in match_ids:
#                         continue
#                     if num_games <= 0:
#                         raise AllGamesCollected()
#                     if first:
#                         first = False
#                     else:
#                         f.write(',')
#                     f.write(str(match_id))
#                     match_ids.add(match_id)
#                     num_games -= 1
#
#             except AllGamesCollected as e:
#                 # not a real exception, more of a goto
#                 break
#             except HTTPError as h:
#                 print('ERROR: There was an error obtaining this summoners match history')
#                 print(repr(h))
#                 print(traceback.format_exc())
#             except Exception as e:
#                 print('ERROR: There was an error obtaining this match history. Skip.')
#                 print(repr(e))
#                 print(traceback.format_exc())
#         f.write(']\n')
#     return match_ids


# get all top summoners challenger, master, D1
# def get_top_summoners():
#     elite_summoners = cass.get_challenger_league(cass.Queue.ranked_solo_fives, 'KR').entries + cass.get_master_league(
#         cass.Queue.ranked_solo_fives, 'KR').entries
#
#     with open(app_constants.asset_paths["diamond_league_ids"]) as f:
#         leagues = f.readlines()
#     leagues = [x.strip() for x in leagues]
#     high_dia_summoners = []
#     for league in leagues:
#         league = cass.core.league.League(id=league, region="KR")
#         summoners = league.entries.filter(lambda x: x.division == cass.Division.one)
#         high_dia_summoners += summoners
#     high_dia_summoners.sort(key=lambda x: x.league_points, reverse=True)
#     league_entry_result = elite_summoners + high_dia_summoners  # [:min(len(high_dia_summoners), num - len(elite_summoners))]
#     # league_entry_result = league_entry_result[:min(len(league_entry_result), num)]
#     summoner_result = [league_entry.summoner for league_entry in league_entry_result]
#     return summoner_result

class Scraper:

    def __init__(self):
        self.elite_match_ids = set()
        self.lower_match_ids = set()


    def get_league_generators(self, regions):
        leagues = [cass.Tier.diamond, cass.Tier.diamond, cass.Tier.diamond, cass.Tier.diamond, cass.Tier.platinum,
                   cass.Tier.platinum, cass.Tier.platinum, cass.Tier.platinum, cass.Tier.gold, cass.Tier.gold,
                   cass.Tier.gold, cass.Tier.gold, cass.Tier.silver, cass.Tier.silver,
                   cass.Tier.silver, cass.Tier.silver, cass.Tier.bronze, cass.Tier.bronze,
                   cass.Tier.bronze, cass.Tier.bronze, cass.Tier.iron, cass.Tier.iron,
                   cass.Tier.iron, cass.Tier.iron]
        divisions = [cass.Division.one, cass.Division.two, cass.Division.three, cass.Division.four, cass.Division.one,
                     cass.Division.two, cass.Division.three, cass.Division.four, cass.Division.one, cass.Division.two,
                     cass.Division.three, cass.Division.four, cass.Division.one, cass.Division.two, cass.Division.three,
                     cass.Division.four, cass.Division.one, cass.Division.two, cass.Division.three, cass.Division.four,
                     cass.Division.one, cass.Division.two, cass.Division.three, cass.Division.four]
        lower_league_generators = [
            {region: iter(cass.LeagueEntries(region=region, queue=cass.Queue.ranked_solo_fives, tier=league,
                                    division=division)) for region in regions} for league, division in
            zip(leagues, divisions)]
        elite_league_generators = [{region:iter(cass.get_challenger_league(cass.Queue.ranked_solo_fives,
                                                                          region).entries) for region in regions},
                                   {region: iter(cass.get_grandmaster_league(cass.Queue.ranked_solo_fives,
                                                                          region).entries) for region in regions},
                                   {region: iter(cass.get_master_league(cass.Queue.ranked_solo_fives, region).entries)
                                    for \
                region in regions}]
        return elite_league_generators, lower_league_generators



    def get_match_ids_for_tier(self, leagues, num_games_per_league, regions, start_date, out_path, logs,
                                  spillover=True):
        if spillover:
            spillover_log = open(out_path + "SPILLOVERS", "w")
        match_ids_by_region = {region:set() for region in regions}
        for i, league in enumerate(leagues):
            regions_copy = regions.copy()
            regions_priorities = [4, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1]
            while num_games_per_league[i] >= 0:
                if sum(regions_priorities) <= 0:
                    print(f"This league is done: Queue {i} {logs[i]}. Countdown left:"
                          f" {num_games_per_league[i]}")
                    if spillover and i+1 < len(leagues):
                        num_games_per_league[i+1] += num_games_per_league[i]
                        print(f"Spillover {num_games_per_league[i]} from {logs[i]} to {logs[i + 1]}")
                        spillover_log.write(f"Spillover {num_games_per_league[i]} from {logs[i]} to {logs[i + 1]}")
                    break

                regions_prioritized = np.concatenate([[regions_copy[region_index]]*prio for region_index,
                                                                                         prio in enumerate(
                        regions_priorities)], axis=0)

                for region in regions_prioritized:
                    print(f"\n\nLeague {logs[i]} Counter: {num_games_per_league[i]}")
                    try:
                        try:
                            summoner = next(league[region]).summoner
                        except StopIteration as e:
                            print(f"This region is done: Queue {i} {region} {logs[i]}. Countdown left:"
                                  f" {num_games_per_league[i]}")
                            regions_priorities[regions.index(region)] = 0

                            break

                        summ_match_hist = summoner.match_history(queues={cass.Queue.ranked_solo_fives},
                                                                 begin_time=start_date)
                        next_matches = [match.id for match in summ_match_hist]

                        prev_len = len(match_ids_by_region[region])
                        match_ids_by_region[region].update(next_matches)
                        num_games_per_league[i] -= len(match_ids_by_region[region]) - prev_len

                    except Exception as e:
                        print("Error downloading this summoner. Skip.")
                        time.sleep(2)

        for region in regions:
            with open(out_path + "_" + region, "w") as f:
                f.write(json.dumps(list(match_ids_by_region[region])))
        if spillover:
            spillover_log.close()


    def get_match_ids(self, elite_counters, lower_counters, regions, start_date):
        elite_leagues, lower_leagues = self.get_league_generators(regions)

        self.get_match_ids_for_tier(elite_leagues, elite_counters, regions, start_date,
                                       app_constants.train_paths[
            "elite_matchids"], game_constants.ELITE_LEAGUES, True)
        self.get_match_ids_for_tier(lower_leagues, lower_counters, regions, start_date,
                                       app_constants.train_paths[
            "lower_matchids"], game_constants.LOWER_LEAGUES)


    def sort_if_complete(self, team):
        team_positions = [participant["lane"] for participant in team]

        bot_first_cs = -1
        bot_first_i = -1

        # break bot into adc and sup
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

        if ("None" in team_positions or (
                sorted(team_positions) != ["bot", "bot", "jg", "mid", "top"])):
            return team, False

        result = []
        for position in game_constants.ROLE_ORDER:
            result.append(pos2summ[position])
        return result, True


    def get_matches(self, match_ids, region):
        lane2str = {cass.Lane.bot_lane: "bot", cass.Lane.jungle: "jg", cass.Lane.mid_lane: "mid",
                    cass.Lane.top_lane: "top", None: "None"}
        with open(app_constants.asset_paths["xp_table"]) as f:
            xp_table = json.load(f)
            xp_table_dict = dict()

            for i in range(len(xp_table)):
                if i == len(xp_table) - 1:
                    xp_table_dict[(xp_table[i][1], 20 * xp_table[i][1])] = int(xp_table[i][0])
                else:
                    xp_table_dict[(xp_table[i][1], xp_table[i + 1][1])] = int(xp_table[i][0])
            xp2lvl = RangeKeyDict(xp_table_dict)



        for match_id in match_ids:
            try:
                print(f"Trying to download match: {match_id}")
                match = cass.get_match(match_id, region=region)
                print(f"Downloaded: {match_id}")
                champ2participant = dict()
                participants = match.red_team.participants + match.blue_team.participants
                for participant in participants:
                    champ_id = participant.champion.id
                    part_id = participant.id
                    champ2participant[champ_id] = part_id

                winning_team = match.blue_team
                losing_team = match.red_team
                first_team_blue = 1
                if losing_team.win:
                    winning_team, losing_team = losing_team, winning_team
                    first_team_blue = 0
                print("Determine win team complete")
                teams = []
                for team in [winning_team, losing_team]:
                    out_team = []
                    for participant in team.participants:
                        summ_dict = {"participantId": participant.id, "championId": participant.champion.id,
                                     "spell1Id": participant.summoner_spell_d.id,
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

                print("Parsing teams complete")
                winning_team, winning_team_sorted = self.sort_if_complete(teams[0])
                losing_team, losing_team_sorted = self.sort_if_complete(teams[1])
                teams = np.ravel([winning_team, losing_team]).tolist()

                #this shit is necessary because riot apparently can't sort their events into the right frame
                # bucket if timestamps overlap
                PreprocessedFrame = lambda **kwargs: type("Object", (), kwargs)()
                preprocessed_frames = [match.timeline.frames[0]]
                for frame in match.timeline.frames[1:]:
                    preprocessed_frames.append(PreprocessedFrame(timestamp=frame.timestamp, participant_frames=
                    frame.participant_frames, events=frame.events))

                    for event in frame.events:
                        if event.timestamp <= preprocessed_frames[-2].timestamp:
                            preprocessed_frames[-2].events.append(preprocessed_frames[-1].events.pop(0))

                print("Sort frames complete")
                events = []
                prev_frame = preprocessed_frames[0]
                prev_participant_frames = prev_frame.participant_frames

                participant_id2header_participants_index = {participant["participantId"]: index for index, participant
                                                            in
                                                            enumerate(teams)}
                kda = np.zeros((10, 3), dtype=np.int32)
                dragon2index = game_constants.dragon2index
                dragon_kills = [[0,0,0,0],[0,0,0,0]]
                dragon_soul_type = [[0,0,0,0],[0,0,0,0]]
                dragon_soul_obtained = [0,0]
                elder_active = [0,0]
                elder_start_timer = None
                baron_start_timer = None
                baron_active = [0,0]
                turrets_destroyed = [0,0]
                rift_type = [0,0,0,0]

                event_counter = 0

                for frame_index, frame in enumerate(preprocessed_frames[1:]):
                    participant_frames = frame.participant_frames
                    # last interval might be shorter
                    if frame_index >= len(preprocessed_frames) - 2:
                        interval = ((frame.events[-1].timestamp.seconds-1) % 60) + 1
                    else:
                        interval = frame.timestamp.seconds - prev_frame.timestamp.seconds

                    gold_slope = [0] * 10
                    cs_slope = [0] * 10
                    neutral_cs_slope = [0] * 10
                    xp_slope = [0] * 10

                    for i in range(10):
                        current_participant = participant_frames[i + 1]
                        current_participant_before = prev_participant_frames[i + 1]
                        current_participant_header_index = participant_id2header_participants_index[
                            current_participant.participant_id]

                        gold_slope[current_participant_header_index] = (
                                                                                   current_participant.gold_earned - current_participant_before.gold_earned) / \
                                                                       interval
                        cs_slope[current_participant_header_index] = (
                                                                             current_participant.creep_score -
                                                                             current_participant_before.creep_score) / interval
                        neutral_cs_slope[current_participant_header_index] = (
                                                                                     current_participant.neutral_minions_killed -
                                                                                     current_participant_before.neutral_minions_killed) / interval
                        xp_slope[current_participant_header_index] = (
                                                                             current_participant.experience -
                                                                             current_participant_before.experience) / interval

                    event_current_gold = [0] * 10
                    event_total_gold = [0] * 10
                    event_cs = [0] * 10
                    event_neutral_cs = [0] * 10
                    event_xp = [0] * 10
                    event_lvl = [0] * 10

                    for event in frame.events:
                        event_type = event.type

                        if (baron_active[0] or baron_active[1]) and event.timestamp.seconds > baron_start_timer + \
                                game_constants.BARON_DURATION:
                            baron_active = [0,0]
                        if (elder_active[0] or elder_active[1]) and event.timestamp.seconds > elder_start_timer + \
                                game_constants.ELDER_DURATION:
                            elder_active = [0, 0]


                        if event_type == "ELITE_MONSTER_KILL" and event.monster_type != "RIFTHERALD":
                            team = int(participant_id2header_participants_index[event.killer_id] >= 5)
                            if event.monster_type == "DRAGON":
                                if event.monster_sub_type == "ELDER_DRAGON":
                                    elder_active[team] = 1
                                    elder_start_timer = event.timestamp.seconds
                                else:
                                    dragon_index = dragon2index[event.monster_sub_type]
                                    dragon_kills[team][dragon_index] += 1
                                    #this one determines the type of rift
                                    if sum(dragon_kills[0]) + sum(dragon_kills[1]) == 3:
                                        rift_type[dragon2index[event.monster_sub_type]] = 1
                                    #this was the soul
                                    if sum(dragon_kills[team]) == 4:
                                        dragon_soul_obtained[team] = 1
                                        dragon_soul_type[team] = rift_type
                            elif event.monster_type == "BARON_NASHOR":
                                baron_active[team] = 1
                                baron_start_timer = event.timestamp.seconds
                            #don't care about harold

                        elif event_type == "BUILDING_KILL":
                            if event.killer_id == 0:
                                pass
                                # if event.team_id == 100 and blue_team_wins:
                                if event._data[cass.EventData].side == 100 and first_team_blue or \
                                    event._data[cass.EventData].side == 200 and not first_team_blue:
                                    team = 1
                                else:
                                    team = 0
                            else:
                                team = int(participant_id2header_participants_index[event.killer_id] >= 5)
                            if event.building_type == "TOWER_BUILDING":
                                turrets_destroyed[team] += 1



                        elif event_type == "CHAMPION_KILL":
                            for assister in event.assisting_participants:
                                kda[participant_id2header_participants_index[assister]][2] += 1
                            victim_kda = kda[participant_id2header_participants_index[event.victim_id]]
                            victim_kda[1] += 1
                            # champ died to turret or jg, etc.
                            if event.killer_id != 0:
                                killer_kda = kda[participant_id2header_participants_index[event.killer_id]]
                                killer_kda[0] += 1


                        elif event_type == "ITEM_DESTROYED" or event_type == "ITEM_PURCHASED" or event_type == \
                                "ITEM_SOLD":
                            if event.item_id != 3340 and event.item_id != 3363 and event.item_id != 3364:

                                event_timestamp = event.timestamp
                                interval = (event_timestamp.seconds - prev_frame.timestamp.seconds)

                                for i in range(10):
                                    current_participant_before = prev_participant_frames[i + 1]
                                    current_participant_header_index = participant_id2header_participants_index[
                                        current_participant_before.participant_id]

                                    event_current_gold[
                                        current_participant_header_index] = current_participant_before.current_gold
                                    event_current_gold[current_participant_header_index] += interval * gold_slope[
                                        current_participant_header_index]

                                    event_total_gold[
                                        current_participant_header_index] = current_participant_before.gold_earned + \
                                                                            interval * \
                                                                            gold_slope[current_participant_header_index]

                                    event_cs[current_participant_header_index] = \
                                        current_participant_before.creep_score + interval * cs_slope[
                                            current_participant_header_index]
                                    event_neutral_cs[
                                        current_participant_header_index] = current_participant_before.neutral_minions_killed + \
                                                                            interval * neutral_cs_slope[
                                                                                current_participant_header_index]
                                    event_xp[current_participant_header_index] = current_participant_before.experience + \
                                                                                 interval * xp_slope[
                                                                                     current_participant_header_index]
                                    event_lvl[current_participant_header_index] = xp2lvl[
                                        event_xp[current_participant_header_index]]

                                events.append(
                                    {"participantId": event.participant_id,
                                     "itemId": event.item_id,
                                     "type": event_type,
                                     "timestamp": int(event.timestamp.total_seconds() * 1000),
                                     "total_gold": np.around(event_total_gold.copy(), 2).tolist(),
                                     "current_gold_sloped": np.around(event_current_gold.copy(), 2).tolist(),
                                     "cs": np.around(event_cs.copy(), 2).tolist(),
                                     "neutral_cs": np.around(event_neutral_cs.copy(), 2).tolist(),
                                     "xp": np.around(event_xp.copy(), 2).tolist(),
                                     "lvl": event_lvl.copy(),
                                     "kda": np.ravel(kda).tolist().copy(),
                                     "frame_index": frame_index + 1,
                                     "baron_active": baron_active.copy(),
                                     "elder_active": elder_active.copy(),
                                     "dragons_killed": np.ravel(dragon_kills).tolist().copy(),
                                     "dragon_soul_obtained": dragon_soul_obtained.copy(),
                                     "dragon_soul_type": np.ravel(dragon_soul_type).tolist().copy(),
                                     "turrets_destroyed": turrets_destroyed.copy(),
                                     "first_team_blue": first_team_blue
                                     })


                        elif event_type == "ITEM_UNDO":
                            if event.after_id != 3340 and event.after_id != 3363 and event.after_id != 3364:
                                events.append(
                                    {"participantId": event.participant_id,
                                     "beforeId": event.before_id,
                                     "afterId": event.after_id,
                                     "type": event_type,
                                     "frame_index": frame_index + 1,
                                     "timestamp": int(event.timestamp.total_seconds() * 1000)})
                        event_counter += 1
                    prev_participant_frames = participant_frames
                    prev_frame = frame
                teams_sorted = [1,1] if winning_team_sorted and losing_team_sorted \
                    else [1,0] if winning_team_sorted else [0,1] if losing_team_sorted else [0,0]

                yield {"gameId": region + "_" + str(match_id), "sorted": teams_sorted, "participants": teams,
                       "itemsTimeline": events}

                print(f"Processing complete {match_id}")

            except HTTPError as e:
                print('HTTP ERROR: There was an error obtaining this match. Skip.')
                time.sleep(10)
                print(repr(e))
                print(traceback.format_exc())
            except Exception as e:
                print('ERROR: There was an error obtaining this match. Skip.')
                time.sleep(10)
                print(repr(e))
                print(traceback.format_exc())
