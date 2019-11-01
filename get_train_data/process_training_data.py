import glob
import os
import time
from collections import Counter
from functools import reduce
from itertools import compress
from multiprocessing import Process, Queue

import arrow
import numpy as np
from jq import jq

from constants import game_constants
from get_train_data import scrape_data
from train_model import data_loader
from train_model import train
from utils import utils, build_path, cass_configured as cass
from utils.artifact_manager import *


class DuplicateTimestampException(Exception):
    pass


class ProcessPositionsTrainingData:

    def __init__(self, num_matches, cut_off_date):
        self.num_matches = num_matches
        self.cut_off_date = cut_off_date
        self.data_x = []
        self.champ_manager = ChampManager()
        self.spell_manager = SimpleManager("spells")


    def start(self):
        self.build_np_db()
        t = train.StaticTrainingDataTrainer()
        t.build_positions_model()


    def build_np_db(self):
        print("Building numpy database now. This may take a few minutes.")
        # if len(sys.argv) != 2:
        #     print("specify gamefile")
        #     exit(-1)
        # self.x_filename = sys.argv[1]

        print("Loading input files")
        with open(app_constants.train_paths["presorted_matches_path"]) as f:
            raw = json.load(f)
        print("Complete")
        print("Generating input & output vectors...")
        sorted_counter = 0
        progress_counter = -1

        for game in raw:
            sorted_teams = []
            progress_counter += 1
            if game["sorted"] == "1,2":
                sorted_teams.append(game['participants'][:5])
                sorted_teams.append(game['participants'][5:])
                sorted_counter += 2
            elif game["sorted"] == "1":
                sorted_teams.append(game['participants'][:5])
                sorted_counter += 1
            elif game["sorted"] == "2":
                sorted_teams.append(game['participants'][5:])
                sorted_counter += 1
            else:
                continue

            for team in sorted_teams:
                x = tuple(
                    [[self.champ_manager.lookup_by("id", str(participant['championId']))["int"],
                      self.spell_manager.lookup_by("id", str(participant['spell1Id']))["int"],
                      self.spell_manager.lookup_by("id", str(participant['spell2Id']))["int"], participant['kills'],
                      participant['deaths'], participant['assists'],
                      participant['earned'], participant['level'],
                      participant['minionsKilled'], participant['neutralMinionsKilled'], participant['wardsPlaced']] for
                     participant in team])
                self.data_x.append(x)

            print("current file {:.2%} processed".format(progress_counter / len(raw)))
        print(f"{sorted_counter} teams were in the right order out of {2 * len(raw)}")

        print("Writing to disk...")
        self.write_positions_to_np_file(chunksize=1000)


    def write_positions_to_np_file(self, chunksize=100000, train_test_split=0.15):
        old_filenames = glob.glob(app_constants.train_paths["positions_processed"] + 'train_x*.npz')
        old_filenames.extend(glob.glob(app_constants.train_paths["positions_processed"] + 'test_x*.npz'))
        for filename in old_filenames:
            os.remove(filename)

        print("Now writing numpy files to disk")
        splitpoint = len(self.data_x) * (1 - train_test_split)
        # new_file_name_x = self.x_filename[self.x_filename.rfind("/") + len('structuredForRole') + 1:]
        for i, x_chunk in enumerate(utils.chunks(self.data_x, chunksize)):
            with open(app_constants.train_paths["positions_processed"] + (
                    'test_x' if i * chunksize > splitpoint else 'train_x') + str(i) + '.npz',
                      "wb") as writer:
                np.savez_compressed(writer, x_chunk)
            print("{}% complete".format(int(min(100, int(100 * (i * chunksize / len(self.data_x)))))))


class ProcessNextItemsTrainingData:

    def __init__(self):
        super().__init__()
        self.champ_manager = ChampManager()
        self.item_manager = ItemManager()
        self.spell_manager = SimpleManager("spells")
        self.role_predictor = None

        self.data_x = None
        self.jq_scripts = {}
        for name in app_constants.jq_script_names:
            with open(app_constants.jq_base_path + name, "r") as f:
                self.jq_scripts[name] = f.read()
        self.jq_scripts["merged"] = self.jq_scripts[
                                        "sortEqualTimestamps.jq"] + ' | ' + self.jq_scripts[
                                        "buildAbsoluteItemTimeline.jq"]


    # input is a list of games
    def run_all_transformations(self, training_data):

        training_data = list(self.sort_equal_timestamps(training_data))
        training_data = list(self.remove_undone_items(training_data))

        # with open("output", "w") as f:
        #     f.write(json.dumps(training_data))
        training_data = list(jq(self.jq_scripts["buildAbsoluteItemTimeline.jq"]).transform([game]) for game in
                             training_data)

        # generator must be popped here, otherwise the next_items generator will be defined on this one and and skip
        # one item when this one advances

        result = self.post_process(matches=training_data)

        return result


    def build_absolute_item_timeline(self, matches, log_info):
        progress_counter = 0
        for match in matches:
            events = match["itemsTimeline"]
            absolute_events = [[Counter() for _ in range(10)]]
            prev_event = None
            participantId2Slot = {participant["participantId"]: index for index, participant in enumerate(match[
                                                                                                              "participants"])}
            for event in events:
                event_type = event["type"]
                itemId = event["itemId"]
                # its the first in this group of timestamps
                current_participant_items = absolute_events[-1][participantId2Slot[event["participantId"]]]

                if event_type == "ITEM_PURCHASED":
                    if not ((itemId == 2003 and 2003 in current_participant_items) or itemId == 2138 or itemId == 2139 \
                            or \
                            itemId == 2140 or \
                            itemId == 3901 or itemId == 3902 or itemId == 3903):
                        if not prev_event or event["timestamp"] != prev_event["timestamp"]:
                            absolute_events.append([Counter(counter) for counter in absolute_events[-1]])
                            tmp = []
                            for counter in absolute_events[-1]:
                                l = [[int(key) for _ in range(count)] for key, count in counter.items()]
                                l = [item for sublist in l for item in sublist]
                                tmp.append(l)
                            absolute_events[-2] = tmp
                        prev_event = event
                        current_participant_items = absolute_events[-1][participantId2Slot[event["participantId"]]]
                        current_participant_items += Counter({itemId: 1})
                elif event_type == "ITEM_DESTROYED":
                    if not (itemId == 3004 or itemId == 3003):
                        if not prev_event or event["timestamp"] != prev_event["timestamp"]:
                            absolute_events.append([Counter(counter) for counter in absolute_events[-1]])
                            tmp = []
                            for counter in absolute_events[-1]:
                                l = [[int(key) for _ in range(count)] for key, count in counter.items()]
                                l = [item for sublist in l for item in sublist]
                                tmp.append(l)
                            absolute_events[-2] = tmp
                        prev_event = event
                        current_participant_items = absolute_events[-1][participantId2Slot[event["participantId"]]]
                        current_participant_items -= Counter({itemId: 1})

                elif event_type == "ITEM_SOLD":
                    if not prev_event or event["timestamp"] != prev_event["timestamp"]:
                        absolute_events.append([Counter(counter) for counter in absolute_events[-1]])
                        tmp = []
                        for counter in absolute_events[-1]:
                            l = [[int(key) for _ in range(count)] for key, count in counter.items()]
                            l = [item for sublist in l for item in sublist]
                            tmp.append(l)
                        absolute_events[-2] = tmp
                    prev_event = event
                    current_participant_items = absolute_events[-1][participantId2Slot[event["participantId"]]]
                    if itemId == 3040:
                        itemId = 3003
                    elif itemId == 3042:
                        itemId = 3004

                    current_participant_items -= Counter({itemId: 1})

                else:
                    print("ERROR")

            absolute_events.pop(0)
            absolute_events = reduce(lambda x, y: x + [y] if not y in x else x, absolute_events, [])
            yield {"gameId": match['gameId'], "participants": match['participants'],
                   "itemsTimeline": absolute_events}

            # print(log_info + " absolute item timeline: current file {:.2%} processed".format(progress_counter / len(
            #     matches)))
            progress_counter += 1


    def sort_equal_timestamps(self, matches):
        progress_counter = 0
        for match in matches:
            events = match["itemsTimeline"]
            # some matches have ITEM_PURCHASE events by the same summoner at the same timestamp.
            # since the order of these events is nondeterministic, they cannot be reliable parsed
            # skip
            events_per_timestamp = Counter()
            for event in events:
                if event["type"] != "ITEM_PURCHASED":
                    continue
                hash_ = event["timestamp"], event["participantId"]
                if hash_ in events_per_timestamp:
                    print("DUPLICATE ITEM_PURCHASED TIMESTAMPS")
                    break
                else:
                    events_per_timestamp[hash_] = 1
            else:
                events.sort(key=ProcessNextItemsTrainingData.keyFunc)
                yield {"gameId": match['gameId'], "participants": match['participants'],
                       "itemsTimeline": events}


    @staticmethod
    def keyFunc(elem):
        if elem["type"] == "ITEM_PURCHASED":
            return elem["timestamp"] + elem["participantId"] / 10 + 0.01
        elif elem["type"] == "ITEM_DESTROYED":
            if elem["itemId"] == 2420 or elem["itemId"] == 2421:
                return elem["timestamp"] + elem["participantId"] / 10 + 0.019
            else:
                return elem["timestamp"] + elem["participantId"] / 10 + 0.02
        elif elem["type"] == "ITEM_SOLD":
            return elem["timestamp"] + elem["participantId"] / 10 + 0.03
        elif elem["type"] == "ITEM_UNDO":
            return elem["timestamp"] + elem["participantId"] / 10 + 0.04
        else:
            print(elem)
            1 / 0


    def remove_undone_items(self, matches):
        progress_counter = 0
        for match in matches:
            events = match["itemsTimeline"]
            included = np.ones([len(events)])
            for i in range(len(events) - 1, -1, -1):
                if events[i]["type"] == "ITEM_UNDO":
                    included[i] = 0

                    for j in range(i - 1, -1, -1):
                        if (included[j]
                                and events[j]["participantId"] == events[i]["participantId"]
                                and (events[j]['type'] == "ITEM_PURCHASED" or events[j]['type'] == "ITEM_SOLD")
                                and (events[i]["beforeId"] == events[j]["itemId"] or events[i]["afterId"] == events[j][
                                    "itemId"])
                                and events[i]["timestamp"] >= events[j]["timestamp"]):
                            included[j] = 0
                            k = j + 1
                            while events[j]["timestamp"] == events[k]["timestamp"] and events[j]["participantId"] == \
                                    events[k]["participantId"] and events[k]["type"] == "ITEM_DESTROYED":
                                included[k] = 0
                                k += 1
                            break

            events = list(compress(events, included))

            yield {"gameId": match['gameId'], "participants": match['participants'],
                   "itemsTimeline": events}

            # print(log_info + " remove undone items: current file {:.2%} processed".format(progress_counter / len(
            #     matches)))
            progress_counter += 1


    def update_roles(self):
        full_data_loader = data_loader.FullDataLoader()
        full_data = full_data_loader.get_train_data()
        gameId2roles = self.determine_roles(full_data)
        full_data = []
        unsorted_processed_dataloader = data_loader.UnsortedNextItemsDataLoader()
        unsorted_processed_data = unsorted_processed_dataloader.get_train_data()
        sorted_processed_data = self.apply_roles_to_unsorted_processed(gameId2roles, unsorted_processed_data)
        self.write_chunk(sorted_processed_data, app_constants.train_paths["next_items_processed_sorted"])


    def write_chunk(self, data, out_dir):
        data = list(data)
        data = np.concatenate(data, axis=0)
        train_test_split = .85

        train_filename = out_dir + f"train_sorted_processed.npz"
        test_filename = out_dir + f"test_sorted_processed.npz"

        train_test_split_point = int(len(data) * train_test_split)

        train_data = data[:train_test_split_point]
        test_data = data[train_test_split_point:]

        for filename, data in zip([train_filename, test_filename], [train_data, test_data]):
            with open(filename, "wb") as writer:
                np.savez_compressed(writer, data)


    def determine_roles(self, matches_full):
        progress_counter = 0
        unsorted_matches = []
        unsorted_matches_gameId2Index = {}
        unsorted_teams = []

        for match in matches_full:
            gameId = match["gameId"]
            events = match["itemsTimeline"]
            sorted = match["sorted"]
            teams = match["participants"]
            winning_team = teams[:5]
            losing_team = teams[5:]

            # winning team is unsorted
            if sorted == "2":
                unsorted_matches.append(match)
                unsorted_matches_gameId2Index[gameId] = len(unsorted_teams)
                unsorted_teams.append(self.team2role_predictor_input(winning_team))


            # losing team is unsorted
            elif sorted == "1":
                unsorted_matches.append(match)
                unsorted_teams.append(self.team2role_predictor_input(losing_team))


            # none are sorted
            elif sorted == "0":
                unsorted_matches.append(match)
                unsorted_matches_gameId2Index[gameId] = 0, len(unsorted_teams)

                unsorted_teams.append(self.team2role_predictor_input(winning_team))
                unsorted_teams.append(self.team2role_predictor_input(losing_team))

        # apparently this is needed to use tensorflow with multiprocessing:
        # https://github.com/tensorflow/tensorflow/issues/8220
        if not self.role_predictor:
            from train_model import model
            self.role_predictor = model.PositionsModel()
        if unsorted_teams == []:
            return

        sorted_teams = self.role_predictor.multi_predict(unsorted_teams)


        def gameId2roles(gameId):
            sorted, index = unsorted_matches_gameId2Index[gameId]
            if sorted == 0:
                return sorted_teams[index] + (np.array(sorted_teams[index + 1]) + 5).tolist()
            elif sorted == 1:
                return [0, 1, 2, 3, 4] + (np.array(sorted_teams[index]) + 5).tolist()
            elif sorted == 2:
                return sorted_teams[index] + [5, 6, 7, 8, 9]


        return gameId2roles


    def apply_roles_to_unsorted_processed(self, gameId2roles, unsorted_processed):
        champs_per_game = game_constants.CHAMPS_PER_GAME
        items_per_champ = game_constants.MAX_ITEMS_PER_CHAMP

        pos_start = 0
        pos_end = pos_start + 1
        champs_start = pos_end
        champs_end = champs_start + champs_per_game
        items_start = champs_end
        items_end = items_start + items_per_champ * 2 * champs_per_game
        total_gold_start = items_end
        total_gold_end = total_gold_start + champs_per_game
        cs_start = total_gold_end
        cs_end = cs_start + champs_per_game
        neutral_cs_start = cs_end
        neutral_cs_end = neutral_cs_start + champs_per_game
        xp_start = neutral_cs_end
        xp_end = xp_start + champs_per_game
        lvl_start = xp_end
        lvl_end = lvl_start + champs_per_game
        kda_start = lvl_end
        kda_end = kda_start + champs_per_game * 3
        current_gold_start = kda_end
        current_gold_end = current_gold_start + champs_per_game

        for gameId in unsorted_processed:
            try:
                current_game = unsorted_processed[gameId]
                permutation = gameId2roles(int(gameId))

                reordered_result = np.concatenate([current_game[:, pos_start:pos_end],
                                                   current_game[:, champs_start:champs_end][:, permutation],
                                                   current_game[:, items_start:items_end].reshape((-1, champs_per_game,
                                                                                                   items_per_champ * 2))[
                                                   :,
                                                   permutation].reshape(-1, champs_per_game * items_per_champ * 2),
                                                   current_game[:, total_gold_start:total_gold_end][:, permutation],
                                                   current_game[:, cs_start:cs_end][:, permutation],
                                                   current_game[:, neutral_cs_start:neutral_cs_end][:, permutation],
                                                   current_game[:, xp_start:xp_end][:, permutation],
                                                   current_game[:, lvl_start:lvl_end][:, permutation],
                                                   current_game[:, kda_start:kda_end].reshape((-1, champs_per_game,
                                                                                               3))[:,
                                                   permutation].reshape((-1,
                                                                         champs_per_game * 3)),
                                                   current_game[:, current_gold_start:current_gold_end][:, permutation],
                                                   current_game[:, -1:]], axis=1)
                yield reordered_result
            except KeyError as e:
                yield current_game


    #
    #
    #
    # def determine_roles(self, matches, log_info, out_path=None):
    #
    #     def sort_metadata(winning_team_part_ids, losing_team_part_ids, winning_team_positioned, losing_team_positioned,
    #                       events):
    #         sorted_winning_team_part_ids = []
    #         sorted_losing_team_part_ids = []
    #         if not winning_team_positioned:
    #             winning_team_positioned = {zip(game_constants.ROLE_ORDER, winning_team_part_ids)}
    #         if not losing_team_positioned:
    #             losing_team_positioned = {zip(game_constants.ROLE_ORDER, losing_team_part_ids)}
    #
    #         for role in game_constants.ROLE_ORDER:
    #             sorted_winning_team_part_ids.append(champid2participantid[winning_team_positioned[role]])
    #             sorted_losing_team_part_ids.append(champid2participantid[losing_team_positioned[role]])
    #         sorted_team_part_ids = sorted_winning_team_part_ids + sorted_losing_team_part_ids
    #         unsorted_team_part_ids = [participant for participant in winning_team_part_ids + losing_team_part_ids]
    #         team_permutation = [unsorted_team_part_ids.index(part_id) for part_id in
    #                             sorted_team_part_ids]
    #
    #         for event in events:
    #             if event['type'] == "ITEM_UNDO":
    #                 continue
    #             event['total_gold'] = np.array(event['total_gold'])[team_permutation].tolist()
    #             event['current_gold_sloped'] = np.array(event['current_gold_sloped'])[team_permutation].tolist()
    #             event['cs'] = np.array(event['cs'])[team_permutation].tolist()
    #             event['neutral_cs'] = np.array(event['neutral_cs'])[team_permutation].tolist()
    #             event['xp'] = np.array(event['xp'])[team_permutation].tolist()
    #             event['lvl'] = np.array(event['lvl'])[team_permutation].tolist()
    #             event['kda'] = np.array(event['kda'])[team_permutation].tolist()
    #
    #
    #     first = True
    #     if out_path:
    #         fsorted = open(out_path, "w")
    #         fsorted.write('[')
    #
    #
    #     team_index = 0
    #     progress_counter = 0
    #
    #     for unsorted_match in unsorted_matches:
    #         gameId = unsorted_match["gameId"]
    #         events = unsorted_match["itemsTimeline"]
    #         sorted = unsorted_match["sorted"]
    #         teams = unsorted_match["participants"]
    #         winning_team = teams[:5]
    #         losing_team = teams[5:]
    #         champid2participantid = {champ["championId"]: champ["participantId"] for champ in
    #                                  winning_team + losing_team}
    #         if first:
    #             first = False
    #         elif out_path:
    #             fsorted.write(',')
    #
    #         winning_team_part_ids = [part['participantId'] for part in winning_team]
    #         losing_team_part_ids = [part['participantId'] for part in losing_team]
    #
    #         # winning team is unsorted
    #         if sorted == "2":
    #             winning_team_positioned = sorted_teams[team_index]
    #             sort_metadata(winning_team_part_ids, losing_team_part_ids, winning_team_positioned,
    #                           losing_team_positioned,
    #                           events)
    #
    #             team_index += 1
    #             winning_team = []
    #             for position in game_constants.ROLE_ORDER:
    #                 winning_team.append({"championId": winning_team_positioned[position],
    #                                      "participantId": champid2participantid[winning_team_positioned[position]]})
    #             losing_team = [{"championId": champ["championId"], "participantId": champ["participantId"]} for
    #                            champ in losing_team]
    #
    #         # losing team is unsorted
    #         elif sorted == "1":
    #             losing_team_positioned = sorted_teams[team_index]
    #             sort_metadata(winning_team_part_ids, losing_team_part_ids, winning_team_positioned,
    #                           losing_team_positioned,
    #                           events)
    #
    #             team_index += 1
    #             losing_team = []
    #             for position in game_constants.ROLE_ORDER:
    #                 losing_team.append({"championId": losing_team_positioned[position],
    #                                     "participantId": champid2participantid[losing_team_positioned[position]]})
    #             winning_team = [{"championId": champ["championId"], "participantId": champ["participantId"]} for
    #                             champ in winning_team]
    #
    #
    #
    #         # none are sorted
    #         elif sorted == "0":
    #             winning_team_positioned = sorted_teams[team_index]
    #             losing_team_positioned = sorted_teams[team_index + 1]
    #             sort_metadata(winning_team_part_ids, losing_team_part_ids, winning_team_positioned,
    #                           losing_team_positioned,
    #                           events)
    #
    #             team_index += 2
    #
    #             winning_team = []
    #             for position in game_constants.ROLE_ORDER:
    #                 winning_team.append({"championId": winning_team_positioned[position],
    #                                      "participantId": champid2participantid[
    #                                          winning_team_positioned[position]]})
    #
    #             losing_team = []
    #             for position in game_constants.ROLE_ORDER:
    #                 losing_team.append({"championId": losing_team_positioned[position],
    #                                     "participantId": champid2participantid[
    #                                         losing_team_positioned[position]]})
    #
    #         yield {"gameId": gameId, "participants": winning_team + losing_team,
    #                "itemsTimeline": events}
    #         if out_path:
    #             fsorted.write(json.dumps(result[-1], separators=(',', ':')))
    #             fsorted.flush()
    #         # print(log_info + " determine roles unsorted: current file {:.2%} processed".format(progress_counter / len(
    #         #     unsorted_matches)))
    #         progress_counter += 1
    #
    #     if out_path:
    #         fsorted.write(']')
    #         fsorted.close()

    def team2role_predictor_input(self, team):
        data = np.array([[participant['championId'], participant['spell1Id'], participant['spell2Id'],
                          participant['kills'], participant['deaths'], participant['assists'],
                          participant['earned'], participant['level'],
                          participant['minionsKilled'], participant['neutralMinionsKilled'], participant['wardsPlaced']]
                         for
                         participant in team], dtype=np.str)
        champ_ids = np.stack(data[:, 0])
        spell_ids = np.ravel(np.stack(data[:, 1:3]))
        rest = np.array(np.ravel(np.stack(data[:, 3:])), dtype=np.uint16)

        champ_ints = [self.champ_manager.lookup_by("id", champ_id)["int"] for champ_id in champ_ids]
        spell_ints = [self.spell_manager.lookup_by("id", spell_id)["int"] for spell_id in spell_ids]

        return np.concatenate([champ_ints, spell_ints, rest], axis=0)


    def encode_items(self, items):
        items_at_time_x = []
        for player_items in items:
            player_items_dict = Counter(player_items)
            player_items_dict_items = []
            processed_player_items = []
            for item in player_items_dict:
                # these items can fit multiple instances into one item slot
                if item == 2055 or item == 2003:
                    added_item = [self.item_manager.lookup_by('id', str(item))['int'],
                                  player_items_dict[
                                      item]]
                    processed_player_items.append(added_item)
                    player_items_dict_items.append(added_item)
                elif item == 2138 or item == 2139 or item == 2140:
                    continue
                else:
                    added_item = self.item_manager.lookup_by('id', str(item))['int']
                    processed_player_items.extend([[added_item, 1]] * player_items_dict[item])
                    player_items_dict_items.append((added_item, player_items_dict[item]))

            if processed_player_items == []:
                processed_player_items = [[0, 6], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]]
            else:
                empties_length = game_constants.MAX_ITEMS_PER_CHAMP - len(processed_player_items)
                padding_length = game_constants.MAX_ITEMS_PER_CHAMP - len(player_items_dict_items)

                try:
                    if empties_length < 0:
                        raise ValueError()

                    if padding_length == 0:
                        empties = np.array([]).reshape((0, 2)).astype(int)
                        padding = np.array([]).reshape((0, 2)).astype(int)
                    if padding_length == 1:
                        empties = [[0, empties_length]]
                        padding = np.array([]).reshape((0, 2)).astype(int)
                    elif padding_length > 1:
                        empties = [[0, empties_length]]
                        padding = [[-1, -1]] * (padding_length - 1)


                except ValueError as e:
                    print(f"ERROR: {processed_player_items}")
                    print(items)
                    raise e

                processed_player_items = np.concatenate([player_items_dict_items, empties, padding],
                                                        axis=0).tolist()

            items_at_time_x.append(processed_player_items)

        return np.array(items_at_time_x)


    def update_current_gold(self, meta, delta_current_gold, participantid2index):
        if meta["type"] == "ITEM_SOLD":
            delta_update = cass.Item(id=int(meta["itemId"]), region="KR").gold.sell
            delta_current_gold[participantid2index[meta['participantId']]] += delta_update
        elif meta["type"] == "ITEM_PURCHASED":
            new_item = cass.Item(id=int(meta["itemId"]), region="KR")
            part_index = participantid2index[meta['participantId']]
            participant_current_items = meta['absolute_items'][part_index]
            if participant_current_items:
                participant_current_items = [self.item_manager.lookup_by('id', str(item)) for item in
                                             participant_current_items]
                participant_current_items = [int(item['buyable_id']) if 'buyable_id' in item else int(item[
                                                                                                          'id']) for
                                             item in
                                             participant_current_items]
            component_items = build_path.build_path(participant_current_items, new_item)[0]
            delta_update = 0
            for component_item in component_items:
                delta_update -= component_item.gold.base

            meta['current_gold'] = (np.array(meta['current_gold_sloped']) + \
                                    np.array(delta_current_gold)).tolist()
            # lol = sum(1 if i < -100 else 0 for i in meta['current_gold'])
            # if lol > 0:
            #     print(f"less than zero: {meta}")
            del meta['current_gold_sloped']
            return delta_update


    def post_process(self, matches):

        for i, game in enumerate(matches):
            out = []
            game = game[0]
            participantid2index = {participant['participantId']: j for j, participant in enumerate(game[
                                                                                                       'participants'])}
            delta_current_gold = [0] * 10
            prev_frame_index = -1

            champs = np.array([participant['championId'] for participant in game['participants']])
            champs = [self.champ_manager.lookup_by("id", str(champ))["int"] for champ in champs]

            lol = 0
            for current_index, event in enumerate(game['itemsTimeline']):
                pos = participantid2index[event['participantId']]
                if prev_frame_index != event["frame_index"]:
                    prev_frame_index = event["frame_index"]
                    delta_current_gold = [0] * 10

                delta_update = self.update_current_gold(event, delta_current_gold, participantid2index)
                if not delta_update:
                    continue

                delta_current_gold[pos] += delta_update

                if pos > 4:
                    continue
                lol += 1
                try:
                    items_at_time_x = self.encode_items(event['absolute_items'])

                except ValueError as e:
                    print(f"Error occurred. Probably >6 items. event: {event} gameId: {game['gameId']}")
                    yield {}
                    break

                y = self.item_manager.lookup_by("id", str(event['itemId']))["int"]

                out.append(np.concatenate([[pos], champs, np.ravel(items_at_time_x),
                                           np.around(event['total_gold']).astype(int),
                                           np.around(event['cs']).astype(int),
                                           np.around(event['neutral_cs']).astype(int),
                                           np.around(event['xp']).astype(int),
                                           np.around(event['lvl']).astype(int),
                                           np.ravel(event['kda']).tolist(),
                                           np.around(event['current_gold']).astype(int),
                                           [y]
                                           ], 0))
            else:
                yield {str(game['gameId']): out}


    def deflate_next_items(self, matches, log_info, out_file_name=None):
        for i, game in enumerate(matches):
            next_full_item_state = [[], [], [], [], []]
            deflated_next_items = []
            for item_state in game["winningTeamNextItems"][::-1]:
                new_full_item_state = []
                for summ_next_item, summ_next_completed in zip(item_state, next_full_item_state):
                    if not summ_next_item:
                        new_full_item_state.append([])
                        continue
                    next_item_json = self.item_manager.lookup_by("id", str(summ_next_item))
                    starting_item = "starting" in next_item_json and 0 in next_item_json["starting"]
                    if not starting_item:
                        new_full_item_state.append(summ_next_item)
                    else:
                        new_full_item_state.append(summ_next_completed)
                deflated_next_items.append(new_full_item_state)
                next_full_item_state = new_full_item_state
            yield {"gameId": game['gameId'], "winningTeamNextItems": deflated_next_items[::-1]}


    def early_items_only(self, matches_inf, matches_next_items, log_info, out_file_name=None):

        for i, (inf_game, next_items_game) in enumerate(zip(matches_inf, matches_next_items)):
            result = [[], [], [], [], []]
            summs_with_completed_first_item = [0, 0, 0, 0, 0]
            invalid_summs = [0, 0, 0, 0, 0]
            for current_item_state, next_item_state in zip(inf_game["itemsTimeline"], next_items_game[
                "winningTeamNextItems"]):

                if summs_with_completed_first_item == [1, 1, 1, 1, 1]:
                    break

                for summ_index, next_item in enumerate(next_item_state):
                    if invalid_summs[summ_index] or summs_with_completed_first_item[summ_index]:
                        continue
                    if next_item == []:
                        summs_with_completed_first_item[summ_index] = 1
                        continue
                    next_item_json = self.item_manager.lookup_by("id", str(next_item))

                    complete = next_item_json["completion"] == "complete" if "completion" in next_item_json \
                        else False
                    if complete:
                        summs_with_completed_first_item[summ_index] = 1
                        continue
                    starting_item = "starting" in next_item_json and 0 in next_item_json["starting"]
                    valid_item = not ("active" in next_item_json and not next_item_json["active"])
                    if not starting_item or not valid_item:
                        print(f"NOT STARTING ITEM OR INVALID ITEM: {next_item_json}")
                        invalid_summs[summ_index] = 1
                        continue

                    result[summ_index].append([current_item_state, next_item])

            result = [summ_build_path if not invalid else [] for summ_build_path, invalid in zip(result, invalid_summs)]
            yield {"gameId": inf_game['gameId'], "participants": inf_game['participants'], "itemsTimeline": result}


    # chunklen is the total number of games the thread pool(typically 4) are processing together, so chunklen/4 per
    # thread
    def start(self, num_threads=os.cpu_count(), chunklen=400):
        class ProcessTrainingDataWorker(Process):

            def __init__(self, in_queue, out_queue, thread_index, transformations):
                super().__init__()
                self.in_queue = in_queue
                self.out_queue = out_queue
                self.run_transformations = transformations
                self.thread_index = thread_index


            def run(self):
                try:
                    while True:
                        games = self.in_queue.get()
                        if not games:
                            break
                        # pr = cProfile.Profile()
                        # pr.enable()

                        result = list(self.run_transformations(games))
                        out_queue.put(list(result))
                        # self.write_next_item_chunk_to_numpy_file(training_data_early_game, self.out_dir_early)

                    # pr.disable()
                    # s = io.StringIO()
                    # sortby = 'cumulative'
                    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
                    # ps.print_stats()
                    # print(s.getvalue())
                    print(f"Thread {self.thread_index} complete")
                except Exception as e:
                    print(f"ERROR: There was an error transforming these matches!!")
                    print(repr(e))
                    print(traceback.format_exc())
                    raise e


        class WriteResultWorker(Process):

            def __init__(self, out_queue, chunksize, out_dir):
                super().__init__()
                self.in_queue = in_queue
                self.out_queue = out_queue
                self.chunksize = chunksize
                self.out_dir = out_dir


            def write_chunk(self, data, chunk_counter):
                train_filename = self.out_dir + f"train_{chunk_counter}.npz"
                with open(train_filename, "wb") as writer:
                    np.savez_compressed(writer, **data)


            def run(self):

                chunk_counter = 0
                terminated = False
                while not terminated:
                    chunk = {}
                    for i in range(self.chunksize):
                        next_item = self.out_queue.get()
                        if next_item == None:
                            terminated = True
                            break
                        if next_item == []:
                            continue
                        chunk.update(next_item[0])

                    self.write_chunk(chunk, chunk_counter)
                    chunk_counter += 1


        in_queue = Queue(maxsize=4000)
        out_queue = Queue()
        callback = lambda input_: self.run_all_transformations(input_)
        workers = [ProcessTrainingDataWorker(in_queue, out_queue, i, callback) for i in range(num_threads)]

        writer = WriteResultWorker(out_queue, chunklen, app_constants.train_paths[
            "next_items_processed_unsorted"])

        for worker in workers:
            worker.start()
        writer.start()

        for i, match in enumerate(scrape_data.scrape_matches(50000, arrow.Arrow(2019, 10, 4, 0, 0, 0))):
            in_queue.put([match])
            print(f"Match {i}")

        for worker in workers:
            while worker.is_alive():
                time.sleep(1)
                in_queue.put(None)

        out_queue.put(None)
        writer.join()

        print("All complete.")


    @staticmethod
    def list_diff(first, second):
        diff = Counter()
        for item in first:
            diff[item] += 1
        for item in second:
            diff[item] -= 1
        diff = list(diff.elements())
        assert len(diff) <= 1
        if not diff:
            return []
        else:
            return diff[0]


    def build_np_db_for_next_items_early_game(self, games, log_info):
        result = dict()
        i = 0
        for game in games:

            team1_team_champs = np.array(game['participants'][:5])
            team2_team_champs = np.array(game['participants'][5:])

            team1_team_champs = [self.champ_manager.lookup_by("id", str(champ))["int"] for champ in team1_team_champs]
            team2_team_champs = [self.champ_manager.lookup_by("id", str(champ))["int"] for champ in team2_team_champs]

            # next items could be shorter than absolute items because at match end there are no next item predictions, or the losing could continue buying

            for summ_index, summ_itemsTimeline in enumerate(game["itemsTimeline"]):
                for items_x, item_y in summ_itemsTimeline:

                    try:
                        team1_team_items_at_time_x = items_x[:5]
                        team1_team_items_at_time_x = [
                            np.pad(player_items, (0, game_constants.MAX_ITEMS_PER_CHAMP - len(player_items)),
                                   'constant',
                                   constant_values=(0, 0)) for player_items in team1_team_items_at_time_x]
                        team1_team_items_at_time_x = np.ravel(team1_team_items_at_time_x).astype(int)

                        team2_team_items_at_time_x = items_x[5:]
                        team2_team_items_at_time_x = [
                            np.pad(player_items, (0, game_constants.MAX_ITEMS_PER_CHAMP - len(player_items)),
                                   'constant',
                                   constant_values=(
                                       0, 0)) for player_items in team2_team_items_at_time_x]
                        team2_team_items_at_time_x = np.ravel(team2_team_items_at_time_x).astype(int)
                    except ValueError as e:
                        print("ERROR: Probably more than 6 items for summoner: GameId: " + str(game_x['gameId']))
                        print(repr(e))
                        break

                    try:
                        team1_team_items_at_time_x = [self.item_manager.lookup_by("id", str(item))["int"] for item in
                                                      team1_team_items_at_time_x]
                        team2_team_items_at_time_x = [self.item_manager.lookup_by("id", str(item))["int"] for item in
                                                      team2_team_items_at_time_x]
                    except KeyError as e:
                        print("Error: KeyError")
                        print(e)
                        break

                    x = tuple(np.concatenate([[summ_index], team1_team_champs,
                                              team2_team_champs,
                                              team1_team_items_at_time_x,
                                              team2_team_items_at_time_x], 0))

                    # empty items should be set to 0, not empty list
                    y = 0 if item_y == [] else item_y

                    try:
                        y = self.item_manager.lookup_by("id", str(y))["int"]
                    except KeyError as e:
                        print("Error: KeyError")
                        print(e)
                        break

                    # don't include dupes. happens when someone buys a potion and consumes it
                    # also don't include empty item recommendations
                    if x not in result and y:
                        result[x] = y
                else:
                    continue
                # print(log_info + " build_db {0:.0%} complete".format(i / len(abs_inf)))
                i += 1

        return result

        return result_x, result_y


    # no need to shuffle here. only costs time. shuffling will happen during training before each epoch
    @staticmethod
    def _uniformShuffle(l1, l2):
        assert len(l1) == len(l2)
        rng_state = np.random.get_state()
        np.random.shuffle(l1)
        np.random.set_state(rng_state)
        np.random.shuffle(l2)


if __name__ == "__main__":
    # p = ProcessPositionsTrainingData(50000, arrow.Arrow(2019, 7, 14, 0, 0, 0))
    # p.start()

    l = ProcessNextItemsTrainingData()
    l.start()
    # l.update_roles()

    #
    # t = train.StaticTrainingDataTrainer()
    # t.build_next_items_model()
