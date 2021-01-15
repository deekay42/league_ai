
import os
import os
import time
from collections import Counter
from functools import reduce
from itertools import compress
from multiprocessing import Process, Queue

import glob
import numpy as np
from jq import jq

from constants import game_constants
from get_train_data import scrape_data
from train_model import data_loader, train
from utils import build_path, cass_configured as cass
from utils.artifact_manager import *
import arrow
from utils import misc
from train_model.model import NextItemModel
from train_model.train import NextItemsTrainer, BootsTrainer, FirstItemsTrainer, StarterItemsTrainer, ChampImgTrainer
from train_model.input_vector import Input
import uuid


class DuplicateTimestampException(Exception):
    pass


class UndoFrameException(Exception):
    pass



class ProcessTrainingDataWorker(Process):

    def __init__(self, in_queue, out_queues, transformations, thread_index):
        super().__init__()
        self.in_queue = in_queue
        self.out_queues = out_queues
        self.transformations = transformations
        self.thread_index = thread_index



    def run(self):
        while True:
            # print(f"Thread {self.thread_index}: in_q full: {self.in_queue.full()} empty: {self.in_queue.empty()}")
            games = self.in_queue.get()
            if games is None:
                break
            # pr = cProfile.Profile()
            # pr.enable()
            try:
                for transformation, out_queue in zip(self.transformations, self.out_queues):
                    results = list(transformation(games))
                    for result in results:
                        out_queue.put(result)
            except Exception as e:
                print(f"ERROR: There was an error transforming these matches!!")
                print(repr(e))
                print(traceback.format_exc())
            # self.write_next_item_chunk_to_numpy_file(training_data_early_game, self.out_dir_early)

            # pr.disable()
            # s = io.StringIO()
            # sortby = 'cumulative'
            # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            # ps.print_stats()
            # print(s.getvalue())
        print(f"Thread {self.thread_index} complete")



class WriteResultWorker(Process):

    def __init__(self, out_queue, chunksize, out_dir, region):
        super().__init__()
        self.out_queue = out_queue
        self.chunksize = chunksize
        self.out_dir = out_dir
        self.region = region


    def split_into_train_test(train_test_split):
        def decorator_split(func):
            def wrapper(self, data, chunk_counter):
                splitpoint = len(data) * (1 - train_test_split)
                train_filename = self.out_dir + f"train_{chunk_counter}.npz"
                test_filename = self.out_dir + f"test_{chunk_counter}.npz"
                func(data[:splitpoint], train_filename)
                func(data[splitpoint:], test_filename)
            return wrapper
        return decorator_split


    @split_into_train_test(0.15)
    def write_train_test_chunk(self, data, filename):
        with open(filename, "wb") as writer:
            np.savez_compressed(writer, **data)


    def write_chunk(self, data, chunk_counter, out_dir):
        filename = out_dir + f"train_{self.region}_{uuid.uuid4()}.npz"
        with open(filename, "wb") as writer:
            np.savez_compressed(writer, **data)


    def run(self):
        chunk_counter = 0
        terminated = False
        while not terminated:
            chunk = {}
            for i in range(self.chunksize):
                next_item = self.out_queue.get()
                if next_item is None:
                    while not self.out_queue.empty():
                        self.out_queue.get()
                    terminated = True
                    break
                chunk.update(next_item)

            self.write_chunk(chunk, chunk_counter, self.out_dir)
            chunk_counter += 1
        print(f"Writer {self.out_dir} complete")


class NextItemsWriteResultWorker(WriteResultWorker):
    def __init__(self, out_queue, chunksize, out_dir_inf, out_dir_uninf, out_dir_comp, region):
        super().__init__(out_queue, chunksize, None, region)
        self.out_dir_inf = out_dir_inf
        self.out_dir_uninf = out_dir_uninf
        self.out_dir_comp = out_dir_comp


    def run(self):
        chunk_counter = 0
        terminated = False
        while not terminated:
            chunk_inf = {}
            chunk_uninf = {}
            chunk_comp = {}
            for i in range(self.chunksize):
                next_item = self.out_queue.get()
                if next_item is None:
                    while not self.out_queue.empty():
                        self.out_queue.get()
                    terminated = True
                    break
                chunk_inf.update(next_item[0])
                chunk_uninf.update(next_item[1])
                chunk_comp.update(next_item[2])

            self.write_chunk(chunk_inf, chunk_counter, self.out_dir_inf)
            self.write_chunk(chunk_uninf, chunk_counter, self.out_dir_uninf)
            self.write_chunk(chunk_comp, chunk_counter, self.out_dir_comp)
            chunk_counter += 1
        print(f"Writer {self.out_dir_inf} {self.out_dir_uninf} {self.out_dir_comp} complete")


class ProcessNextItemsTrainingData:

    def __init__(self):
        super().__init__()
        self.champ_manager = ChampManager()
        self.item_manager = ItemManager()
        self.spell_manager = SimpleManager("spells")
        self.role_predictor = None
        self.scraper = scrape_data.Scraper()
        self.data_x = None
        self.jq_scripts = {}
        for name in app_constants.jq_script_names:
            with open(app_constants.jq_base_path + name, "r") as f:
                self.jq_scripts[name] = f.read()
        self.jq_scripts["merged"] = self.jq_scripts[
                                        "sortEqualTimestamps.jq"] + ' | ' + self.jq_scripts[
                                        "buildAbsoluteItemTimeline.jq"]


    def team2role_predictor_input_perm(self, team):
        data = np.array([(self.champ_manager.lookup_by("id", str(participant['championId']))["int"],
                         self.spell_manager.lookup_by("id", str(participant['spell1Id']))["int"],
                            self.spell_manager.lookup_by("id", str(participant['spell2Id']))["int"],
                          participant['kills'], participant['deaths'], participant['assists'],
                          participant['earned'], participant['level'],
                          participant['minionsKilled'], participant['neutralMinionsKilled'], participant[
                             'wardsPlaced']) for participant in team], dtype=np.int32)
        return data


    def run_positions_transformations(self, games, sorted_state):
        for game in games:
            for team_offset, team_sorted in enumerate(game["sorted"]):
                encoded_team = self.team2role_predictor_input_perm(game['participants'][team_offset * 5:team_offset * 5 + 5])
                result = {str(game['gameId'])+"_"+str(team_offset):  encoded_team}
                if team_sorted == sorted_state:
                    yield result


    # input is a list of games
    def run_next_item_transformations(self, training_data, region):
        transformations = [self.sort_equal_timestamps, lambda x: self.remove_undone_items(x, region),
                           self.remove_unwanted_items,
                           self.insert_null_items,
                           self.build_abs_timeline,
                           lambda x: self.post_process(x, region)]
        result = training_data
        for transformation in transformations:
            result = transformation(result)
        return result


    def remove_unwanted_items(self, matches):
        for match in matches:
            new_events = []
            events = match["itemsTimeline"]
            for event_index, event in enumerate(events):
                if not (event["type"] == "ITEM_PURCHASED" and event['itemId'] > 7000):
                    new_events.append(event)
            match['itemsTimeline'] = new_events
            yield match




    def insert_null_items(self, matches):
        for match in matches:
            participantid2index = {participant['participantId']: j for j, participant in enumerate(match[
                                                                                                       'participants'])}
            events = match["itemsTimeline"]
            next_event_by_summ = dict()
            last_event_by_summ = np.full(10, -1)

            for event_index, event in enumerate(events):
                if event["type"] == "ITEM_PURCHASED":
                    summ_index = participantid2index[event["participantId"]]
                    if last_event_by_summ[summ_index] != -1:
                        next_event_by_summ[last_event_by_summ[summ_index]] = event_index
                    last_event_by_summ[summ_index] = event_index
                event_index += 1
            for last_event in last_event_by_summ:
                next_event_by_summ[last_event] = -1

            new_events = []
            event_index = 0
            while event_index < len(events):
                event = events[event_index]
                new_events.append(event)
                append_event = event

                if event["type"] == "ITEM_PURCHASED":
                    next_event = events[next_event_by_summ[event_index]]
                    if next_event == -1 or (next_event['timestamp'] - event['timestamp']) > 15000:
                        append_event = dict(event)
                        append_event['itemId'] = 0
                        append_event['timestamp'] += 1

                        if event_index < len(events) - 1:
                            event_index += 1
                            next_event = events[event_index]
                            while next_event['timestamp'] == event['timestamp'] and event['participantId'] == \
                                    next_event['participantId']:
                                new_events.append(next_event)
                                event_index += 1
                                next_event = events[event_index]
                            event_index -= 1
                        new_events.append(append_event)

                event_index += 1
            match['itemsTimeline'] = new_events
            yield match


    def build_abs_timeline(self, training_data):
        for game in training_data:
            yield jq(self.jq_scripts["buildAbsoluteItemTimeline.jq"]).transform([game])


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
            # since the order of these events is nondeterministic, they cannot be reliably parsed
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


    def remove_undone_items(self, matches, region):
        progress_counter = 0

        for match in matches:
            events = match["itemsTimeline"]
            included = np.ones([len(events)])
            participantid2index = {participant['participantId']: j for j, participant in
                                   enumerate(match['participants'])}

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
                            spillover_event = False
                            if events[j]["frame_index"] != events[i]["frame_index"]:
                                spillover_event = True
                                if events[j]['type'] == "ITEM_PURCHASED":
                                    spillover_gold = cass.Item(id=int(events[j]["itemId"]),
                                                               region=region).gold.total
                                elif events[j]['type'] == "ITEM_SOLD":
                                    spillover_gold = cass.Item(id=int(events[j]["itemId"]), region=region).gold.sell
                            while events[j]["timestamp"] == events[k]["timestamp"] and events[j]["participantId"] == \
                                    events[k]["participantId"] and events[k]["type"] == "ITEM_DESTROYED":
                                included[k] = 0
                                if spillover_event:
                                    spillover_gold -= cass.Item(id=int(events[k]["itemId"]), region=region).gold.base
                                k += 1

                            if spillover_event:
                                for l in range(i, -1, -1):
                                    if events[l]["frame_index"] != events[i]["frame_index"]:
                                        break
                                l += 1
                                while True:
                                    if "current_gold_sloped" in events[l]:
                                        events[l]["current_gold_sloped"][participantid2index[events[i]["participantId"]]] += spillover_gold
                                    l += 1
                                    if l >= len(events) or events[l]["frame_index"] != events[l - 1]["frame_index"]:
                                        break
                            break

            events = list(compress(events, included))

            yield {"gameId": match['gameId'], "participants": match['participants'],
                   "itemsTimeline": events}

            # print(log_info + " remove undone items: current file {:.2%} processed".format(progress_counter / len(
            #     matches)))
            progress_counter += 1


    def calc_champ_role_stats(self):
        total_champ_distrib = dict()

        total_champ_distrib = {champ_int: sum(self.champs_vs_roles[champ_int].values()) for champ_int in
                               self.champs_vs_roles}
        champs_vs_roles_names = {ChampManager().lookup_by("int", champ_int)["name"]: champ_dist for champ_int,
                                                                                                    champ_dist in
                                 self.champs_vs_roles.items()}
        total_champs_vs_roles = sum([sum(champ_roles.values()) for champ_roles in champs_vs_roles_names.values()])
        champs_vs_roles_rel = {int(champ_int): {
            champ_role: champ_dist/total_champ_distrib[
            champ_int] for champ_role, champ_dist in champ_roles.items()} for champ_int,champ_roles in
                                 self.champs_vs_roles.items()}
        # champs_vs_roles_rel = {int(champ_int): {
        #         champ_role: champ_dist/total_champs_vs_roles for champ_role, champ_dist in champ_roles.items()} for champ_int,champ_roles in
        #                              self.champs_vs_roles.items()}
        total_champ_distrib = sorted(np.array(list(total_champ_distrib.items())), key=lambda a: int(a[1]))

        total_role_distrib = {'top': Counter(), 'jg': Counter(), 'mid': Counter(), 'adc': Counter(), 'sup': Counter()}
        for current_champ in champs_vs_roles_names:
            for role in champs_vs_roles_names[current_champ]:
                total_role_distrib[role] += Counter({current_champ: champs_vs_roles_names[current_champ][role]})
        for role in total_role_distrib:
            total_role_distrib[role] = sorted(np.array(list(total_role_distrib[role].items())), key=lambda a: int(a[1]))

        return champs_vs_roles_rel, total_champ_distrib, total_role_distrib


    def update_roles(self):
        unsorted_positions_dl = data_loader.PositionsToBePredDataLoader()
        unsorted_positions = unsorted_positions_dl.read()
        gameId2roles = self.determine_roles(unsorted_positions)
        self.champs_vs_roles = dict()

        unsorted_dataset_paths = ["next_items_processed_elite_unsorted_inf",
                                  "next_items_processed_elite_unsorted_uninf",
                                  "next_items_processed_elite_unsorted_complete",
                                  "next_items_processed_lower_unsorted_inf",
                                  "next_items_processed_lower_unsorted_uninf",
                                  "next_items_processed_lower_unsorted_complete"
                                  ]
        sorted_dataset_paths = ["next_items_processed_elite_sorted_inf",
                                  "next_items_processed_elite_sorted_uninf",
                                  "next_items_processed_elite_sorted_complete",
                                  "next_items_processed_lower_sorted_inf",
                                  "next_items_processed_lower_sorted_uninf",
                                  "next_items_processed_lower_sorted_complete"
                                  ]

        for i, (unsorted_data, sorted_data) in enumerate(zip(unsorted_dataset_paths, sorted_dataset_paths)):
            print(f"Writing dataset: {i}")
            self.update_roles_by_dataset(gameId2roles, unsorted_data,sorted_data)



    def update_roles_by_dataset(self, gameId2roles, unsorted_path, sorted_path):
        unsorted_processed_dataloader = data_loader.UnsortedNextItemsDataLoader(app_constants.train_paths[
                                                                                             unsorted_path])
        unsorted_processed_data = unsorted_processed_dataloader.get_train_data()
        sorted_processed_data = self.apply_roles_to_unsorted_processed(gameId2roles,
                                                                             unsorted_processed_data)
        self.write_chunk(sorted_processed_data, app_constants.train_paths[sorted_path])


    # def cvt_to_new_positions(self, in_vec):
    #     in_vec = np.array(in_vec)
    #     champ_dim = 1
    #     spell_dim = 2
    #     rest_dim = 8
    #     champs_per_game = 5
    #
    #     spells_offset = champs_per_game*champ_dim
    #     rest_offset = spells_offset + champs_per_game * spell_dim
    #     return np.transpose([np.concatenate([in_vec[:,i*champ_dim][:,np.newaxis],
    #                     in_vec[:,spells_offset + i*spell_dim : spells_offset + (i+1)*spell_dim],
    #                     in_vec[:,rest_offset + i*rest_dim : rest_offset + (i+1)*rest_dim]], axis=1)
    #                     for i in range(champs_per_game)], (1,0,2))


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


    def determine_roles(self, unsorted_positions):
        # apparently this is needed to use tensorflow with multiprocessing:
        # https://github.com/tensorflow/tensorflow/issues/8220
        if not self.role_predictor:
            from train_model import model
            self.role_predictor = model.PositionsModel()
        if not unsorted_positions:
            return
        # unsorted_positions = dict(zip(list(unsorted_positions.keys())[:100], list(unsorted_positions.values())[:100]))
        unsorted_positions_values = list(unsorted_positions.values())
        # unsorted_positions_values = self.cvt_to_new_positions(unsorted_positions_values)
        sorted_teams = self.role_predictor.multi_predict_perm(unsorted_positions_values)
        match_id2perm = {}
        for index, unsorted_team_id in enumerate(unsorted_positions):
            gameId = unsorted_team_id[:-2]
            team_offset = int(unsorted_team_id[-1])
            permutation = match_id2perm.get(gameId, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            if team_offset == 0:
                permutation = sorted_teams[index] + permutation[5:]
            elif team_offset == 1:
                permutation = permutation[:5] + (np.array(sorted_teams[index]) + 5).tolist()
            match_id2perm[gameId] = permutation
        return match_id2perm

    def map_game_id_str2int(self, gameId):
        underscoreindex = gameId.rfind("_")
        if underscoreindex == -1:
            region_prefix = "000"
        else:
            region = gameId[:underscoreindex]
            try:
                region_prefix = game_constants.region2int[region]
            except KeyError:
                region_prefix = "000"
        gameId_int = str(region_prefix) + gameId[underscoreindex + 1:]
        gameId_int = int(gameId_int)
        return gameId_int

    def apply_roles_to_unsorted_processed(self, match_id2perm, unsorted_processed):
        champs_per_game = game_constants.CHAMPS_PER_GAME
        items_per_champ = game_constants.MAX_ITEMS_PER_CHAMP

        for gameId in unsorted_processed:
            try:
                current_game = unsorted_processed[gameId]
                if current_game.size == 0:
                    continue
                gameId_int = self.map_game_id_str2int(gameId)
                gameIds = [[gameId_int]] * current_game.shape[0]
                #this will throw a keyerror if the game is already sorted
                permutation = match_id2perm[gameId]
                new_pos_map = {i:permutation.index(i) for i in [0,1,2,3,4]}
                updated_positions = [[new_pos_map[pos[0]]] for pos in current_game[:, Input.indices["start"][
                                                                                          "pos"]:Input.indices["end"][
                    "pos"]]]

                permutable_elements = {"champs", "total_gold", "cs", "lvl", "kills", "deaths", "assists",
                                       "current_gold"}
                nonpermutable_elements = {"baron", "elder", "dragons_killed", "dragon_soul_type",
                                          "turrets_destroyed", "blue_side"}
                reordered_result = np.zeros((current_game.shape[0], Input.len + 1), dtype=np.float64)
                for slice_name in permutable_elements:
                    reordered_result[:, Input.indices["start"][slice_name]:Input.indices["end"][slice_name]] = \
                        current_game[:, Input.indices["start"][slice_name]:Input.indices["end"][slice_name]][:, permutation]
                for slice_name in nonpermutable_elements:
                    reordered_result[:, Input.indices["start"][slice_name]:Input.indices["end"][slice_name]] = \
                        current_game[:, Input.indices["start"][slice_name]:Input.indices["end"][slice_name]]
                reordered_result[:, Input.indices["start"]["gameid"]:Input.indices["end"]["gameid"]] = gameIds
                reordered_result[:, Input.indices["start"]["pos"]:Input.indices["end"]["pos"]] = updated_positions
                reordered_result[:, Input.indices["start"]["items"]:Input.indices["end"]["items"]] = \
                    current_game[:, Input.indices["start"]["items"]:Input.indices["end"]["items"]].reshape((-1, champs_per_game, items_per_champ * 2))[:,
                                                   permutation].reshape(-1, champs_per_game * items_per_champ * 2)
                reordered_result[:, -1:] = current_game[:, -1:]


                #
                # reordered_result = np.concatenate([gameIds, updated_positions,
                #                                    current_game[:, Input.champs_start:Input.champs_end][:, permutation],
                #                                    current_game[:, Input.items_start:Input.items_end].reshape((-1, champs_per_game,
                #                                                                                    items_per_champ * 2))[
                #                                    :,
                #                                    permutation].reshape(-1, champs_per_game * items_per_champ * 2),
                #                                    current_game[:, Input.total_gold_start:Input.total_gold_end][:, permutation],
                #                                    current_game[:, Input.cs_start:Input.cs_end][:, permutation],
                #                                    current_game[:, Input.neutral_cs_start:Input.neutral_cs_end][:, permutation],
                #                                    current_game[:, Input.xp_start:Input.xp_end][:, permutation],
                #                                    current_game[:, Input.lvl_start:Input.lvl_end][:, permutation],
                #                                    current_game[:, Input.kda_start:Input.kda_end].reshape((-1, champs_per_game,
                #                                                                                3))[:,
                #                                    permutation].reshape((-1,
                #                                                          champs_per_game * 3)),
                #                                    current_game[:, Input.current_gold_start:Input.current_gold_end][:, permutation],
                #                                    current_game[:, Input.baron_start:Input.baron_end],
                #                                    current_game[:, Input.elder_start:Input.elder_end],
                #                                    current_game[:, Input.dragons_killed_start:Input.dragons_killed_end],
                #                                    current_game[:, Input.dragon_soul_start:Input.dragon_soul_end],
                #                                    current_game[:,
                #                                    Input.dragon_soul_type_start:Input.dragon_soul_type_end],
                #                                    current_game[:, Input.turrets_start:Input.turrets_end],
                #                                    current_game[:,
                #                                    Input.first_team_blue_start:Input.first_team_blue_end],
                #
                #
                #                                    current_game[:, -1:]], axis=1)
                # sorted_champs = reordered_result[:, Input.champs_start+1:Input.champs_end+1][0]
                sorted_champs = reordered_result[:, Input.indices["start"]["champs"]:Input.indices["end"]["champs"]][0]
                self.stat_champs_vs_roles(sorted_champs)
                yield reordered_result
            except KeyError as e:
                sorted_champs = current_game[:, Input.indices["start"]["champs"]:Input.indices["end"]["champs"]][0]
                self.stat_champs_vs_roles(sorted_champs)
                current_game[:,Input.indices["start"]["gameid"]:Input.indices["end"]["gameid"]] = gameIds
                yield current_game


    def stat_champs_vs_roles(self, sorted_champs):
        for team in range(2):
            for position in range(5):
                current_champ = sorted_champs[team * 5 + position]
                if current_champ in self.champs_vs_roles:
                    self.champs_vs_roles[current_champ] += Counter({game_constants.ROLE_ORDER[position]:1})
                else:
                    self.champs_vs_roles[current_champ] = Counter({game_constants.ROLE_ORDER[position]: 1})


    def post_process(self, matches, region):
        for i, game in enumerate(matches):
            out_uninf = []
            out_inf = []
            out_complete = []
            game = game[0]
            participantid2index = {participant['participantId']: j for j, participant in enumerate(game[
                                                                                                       'participants'])}
            delta_current_gold = [0] * 10
            prev_frame_index = -1

            champs = np.array([participant['championId'] for participant in game['participants']])
            champs = [self.champ_manager.lookup_by("id", str(champ))["int"] for champ in champs]

            next_full_item_state = [0, 0, 0, 0, 0]
            index2comp_item_state = {}

            for index, event in enumerate(game['itemsTimeline'][::-1]):
                pos = participantid2index[event['participantId']]
                if pos > 4:
                    continue
                current_item = self.item_manager.lookup_by("id", str(event["itemId"]))
                late_item = "completion" in current_item and ("semi" in current_item["completion"] or "complete" in
                                                              current_item["completion"]) and current_item[
                    "id"] != 0
                if late_item:
                    next_full_item_state[pos] = current_item
                index2comp_item_state[len(game['itemsTimeline']) - index - 1] = next_full_item_state.copy()


            for current_index, event in enumerate(game['itemsTimeline']):

                pos = participantid2index[event['participantId']]
                if pos > 4:
                    continue
                if prev_frame_index != event["frame_index"]:
                    prev_frame_index = event["frame_index"]
                    delta_current_gold = [0] * 10

                if event['itemId'] != 0:
                    new_item = cass.Item(id=int(event["itemId"]), region=region)
                part_index = participantid2index[event['participantId']]
                if event["type"] == "ITEM_SOLD":
                    delta_current_gold[part_index] += new_item.gold.sell
                elif event["type"] == "ITEM_PURCHASED":
                    participant_current_items = event['absolute_items'][part_index]
                    if participant_current_items:
                        participant_current_items = [self.item_manager.lookup_by('id', str(item)) for item in
                                                     participant_current_items]
                        participant_current_items = [int(item['buyable_id']) if 'buyable_id' in item else int(item[
                                                                                                                  'id'])
                                                     for
                                                     item in
                                                     participant_current_items]
                    try:
                        items_at_time_x = NextItemModel.encode_items(event['absolute_items'], self.item_manager)
                    except ValueError as e:
                        continue
                    y_item = self.item_manager.lookup_by("id", str(event['itemId']))
                    y = y_item["int"]

                    # uninf_example = [[pos], champs, np.ravel(items_at_time_x),
                    #                                  np.around(event['total_gold']).astype(int),
                    #                                  np.around(event['cs']).astype(int),
                    #                                  np.around(event['neutral_cs']).astype(int),
                    #                                  np.around(event['xp']).astype(int),
                    #                                  np.around(event['lvl']).astype(int),
                    #                                  np.ravel(event['kda']).tolist(),
                    #                                  np.around(event['current_gold_sloped'] + np.array(delta_current_gold)).astype(int),
                    #                                  np.ravel(event["baron_active"]).astype(int),
                    #                                  np.ravel(event["elder_active"]).astype(int),
                    #                                  np.ravel(event["dragons_killed"]).astype(int),
                    #                                  np.ravel(event["dragon_soul_obtained"]).astype(int),
                    #                                  np.ravel(event["dragon_soul_type"]).astype(int),
                    #                                  np.ravel(event["turrets_destroyed"]).astype(int),
                    #                                  np.ravel(event["first_team_blue"]).astype(int),
                    #                                  [y]
                    #                                  ]
                    uninf_example = np.zeros((Input.len+1,), dtype=np.float64)
                    for slice_name in Input.all_slices - {"gameid", "pos", "champs", "items", "current_gold"}:

                        uninf_example[Input.indices["start"][slice_name]:Input.indices["end"][slice_name]] = \
                            np.ravel(event[slice_name]).astype(int)
                    uninf_example[Input.indices["start"]["champs"]:Input.indices["end"]["champs"]] = champs
                    uninf_example[Input.indices["start"]["items"]:Input.indices["end"]["items"]] = np.ravel(items_at_time_x)
                    uninf_example[Input.indices["start"]["pos"]:Input.indices["end"]["pos"]] = [pos]
                    uninf_example[Input.indices["start"]["current_gold"]:Input.indices["end"]["current_gold"]] = \
                        np.around(event['current_gold_sloped'] + np.array(delta_current_gold))
                    uninf_example[Input.indices["start"]["gameid"]:Input.indices["end"]["gameid"]] = self.map_game_id_str2int(game['gameId'])
                    uninf_example[-1] = y
                    out_uninf.append(uninf_example)
                    if event['itemId'] == 0:
                        out_inf.append(uninf_example)
                        continue

                    if current_index in index2comp_item_state:
                        summ_next_items = index2comp_item_state[current_index]
                        for summ_pos, next_complete_item in enumerate(summ_next_items):
                            if next_complete_item == 0 or next_complete_item["id"] == 0:
                                continue
                            # complete_example = [[summ_pos], champs, np.ravel(items_at_time_x),
                            #                  np.around(event['total_gold']).astype(int),
                            #                  np.around(event['cs']).astype(int),
                            #                  np.around(event['neutral_cs']).astype(int),
                            #                  np.around(event['xp']).astype(int),
                            #                  np.around(event['lvl']).astype(int),
                            #                  np.ravel(event['kda']).tolist(),
                            #                  np.around(event['current_gold_sloped'] + np.array(delta_current_gold)).astype(
                            #                      int),
                            #                     np.ravel(event["baron_active"]).astype(int),
                            #                     np.ravel(event["elder_active"]).astype(int),
                            #                     np.ravel(event["dragons_killed"]).astype(int),
                            #                     np.ravel(event["dragon_soul_obtained"]).astype(int),
                            #                     np.ravel(event["dragon_soul_type"]).astype(int),
                            #                     np.ravel(event["turrets_destroyed"]).astype(int),
                            #                     np.ravel(event["first_team_blue"]).astype(int),
                            #                  [next_complete_item["int"]]
                            #                  ]
                            complete_example = np.copy(uninf_example)
                            complete_example[Input.indices["start"]["pos"]:Input.indices["end"]["pos"]] = [summ_pos]
                            complete_example[-1] = next_complete_item["int"]
                            out_complete.append(complete_example)



                    #now do the inflated item path
                    component_items,_, insert_item_states,_ = build_path.build_path_nogold(participant_current_items, new_item)
                    prev_event = event
                    for i, (component_item, insert_item_state) in enumerate(zip(component_items, insert_item_states)):
                        if misc.num_itemslots(Counter(insert_item_state)) > \
                                game_constants.MAX_ITEMS_PER_CHAMP:
                            continue
                        event_copy = prev_event.copy()
                        event_copy['absolute_items'] = prev_event['absolute_items'].copy()
                        event_copy['itemId'] = component_item.id

                        try:
                            items_at_time_x = NextItemModel.encode_items(event_copy['absolute_items'],
                                                                                  self.item_manager)
                            y = self.item_manager.lookup_by("id", str(component_item.id))["int"]
                            if y == 0:
                                print(f"y==0 gameId {str(game['gameId'])}")

                            inf_example = np.copy(uninf_example)
                            inf_example[Input.indices["start"]["items"]:Input.indices["end"]["items"]] = \
                                np.ravel(items_at_time_x)
                            inf_example[-1] = y
                            inf_example[Input.indices["start"]["current_gold"]:Input.indices["end"]["current_gold"]] = \
                                np.around(event_copy['current_gold_sloped'] + np.array(delta_current_gold)).astype(int)
                            out_inf.append(inf_example)

                            #
                            #
                            #
                            # out_inf.append(np.concatenate([[pos], champs, np.ravel(items_at_time_x),
                            #                                np.around(event_copy['total_gold']).astype(int),
                            #                                np.around(event_copy['cs']).astype(int),
                            #                                np.around(event_copy['neutral_cs']).astype(int),
                            #                                np.around(event_copy['xp']).astype(int),
                            #                                np.around(event_copy['lvl']).astype(int),
                            #                                np.ravel(event_copy['kda']).tolist(),
                            #                                np.around(event_copy['current_gold_sloped'] + np.array(
                            #                                    delta_current_gold)).astype(int),
                            #                                np.ravel(event["baron_active"]).astype(int),
                            #                                np.ravel(event["elder_active"]).astype(int),
                            #                                np.ravel(event["dragons_killed"]).astype(int),
                            #                                np.ravel(event["dragon_soul_obtained"]).astype(int),
                            #                                np.ravel(event["dragon_soul_type"]).astype(int),
                            #                                np.ravel(event["turrets_destroyed"]).astype(int),
                            #                                np.ravel(event["first_team_blue"]).astype(int),
                            #
                            #                                [y]
                            #                                ], 0))
                        except ValueError as e:
                            pass

                        delta_current_gold[pos] -= component_item.gold.base
                        event_copy['absolute_items'][part_index] = insert_item_state
                        prev_event = event_copy
            else:
                yield [{str(game['gameId']): out_inf}, {str(game['gameId']): out_uninf}, {str(game['gameId']): out_complete}]




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


    def start(self, number_of_top_games, number_of_lower_games, regions, start_date):
        out_paths_positions = [app_constants.train_paths["positions_processed"],
                               app_constants.train_paths["positions_to_be_pred"]]
        out_paths_elite = [app_constants.train_paths["next_items_processed_elite_unsorted_inf"],
                           app_constants.train_paths["next_items_processed_elite_unsorted_uninf"],
                           app_constants.train_paths["next_items_processed_elite_unsorted_complete"]]

        out_paths_lower = [app_constants.train_paths["next_items_processed_lower_unsorted_inf"],
                           app_constants.train_paths["next_items_processed_lower_unsorted_uninf"],
                           app_constants.train_paths["next_items_processed_lower_unsorted_complete"]]
        for path in out_paths_elite + out_paths_lower + out_paths_positions:
            misc.remove_old_files(path)
        #
        # number_of_top_games = [number_of_top_games//game_constants.NUM_ELITE_LEAGUES]*game_constants.NUM_ELITE_LEAGUES
        # # number_of_top_games = [10000,0,0]
        # number_of_lower_games = [number_of_lower_games//(game_constants.NUM_LEAGUES-game_constants.NUM_ELITE_LEAGUES)]*(
        #         game_constants.NUM_LEAGUES-game_constants.NUM_ELITE_LEAGUES)
        #
        # self.scraper.get_match_ids(number_of_top_games, number_of_lower_games,
        #                                                               regions, start_date)
        #

        # elite_match_ids = [4585558064]
        match_ids = {"elite":{}, "lower":{}}
        elite_files = glob.glob(app_constants.train_paths["elite_matchids"]+"_*")
        lower_files = glob.glob(app_constants.train_paths["lower_matchids"] + "_*")
        for tier, files in zip(match_ids.keys(), [elite_files, lower_files]):
            for file in files:
                with open(file, "r") as f:
                    tmp_match_ids = json.load(f)
                    underscoreindex = file.rfind("_")
                    if underscoreindex == -1:
                        continue
                    region = file[file.rfind("_") + 1:]
                    match_ids[tier][region] = tmp_match_ids
        total_num_games = sum([len(matches) for matches in match_ids["elite"].values()]) + \
                          sum([len(matches) for matches in match_ids["lower"].values()])

        offset = 0
        for tier, paths in zip(match_ids, [out_paths_elite, out_paths_lower]):
            for region in match_ids[tier]:
                self.start_processing(match_ids[tier][region], region, paths[0], paths[1], paths[2], total_num_games,
                                      offset, tier)
                offset += len(match_ids[tier][region])



    # chunklen is the total number of games the thread pool(typically 4) are processing together, so chunklen/4 per
    # thread
    def start_processing(self, match_ids, region, inf_path, uninf_path, complete_path, total_num_games, offset, tier,
                         num_threads=os.cpu_count(), chunklen=400):

        in_queue = Queue()
        out_queues = [Queue(), Queue(), Queue()]
        transformations = [lambda a: self.run_next_item_transformations(a, region),
                           lambda a: self.run_positions_transformations(a, 1),
                           lambda a: self.run_positions_transformations(a, 0)]
        out_keys = ["positions_processed", "positions_to_be_pred"]
        workers = [ProcessTrainingDataWorker(in_queue, out_queues, transformations, i) for i in range(num_threads)]

        writers = [NextItemsWriteResultWorker(out_queues[0], chunklen, inf_path, uninf_path,
                   complete_path, region)]
        writers.extend([WriteResultWorker(out_queue, chunklen, app_constants.train_paths[out_key], region) for out_key,
                                                                                                       out_queue
                   in zip(out_keys, out_queues[1:])])


        for process in workers + writers:
            process.start()

        for i, match in enumerate(self.scraper.get_matches(match_ids, region)):
            in_queue.put([match])
            print(f"\n\nTOTAL: Match {i + offset} out of {total_num_games} ({100*(i+offset)/total_num_games:.2f}%)")
            print(f"{tier} {region}: Match {i} out of {len(match_ids)} ({100 * (i) / len(match_ids):.2f}%)")


        print("Trying to stop workers.")
        for worker in workers:
            while worker.is_alive():
                time.sleep(5)
                in_queue.put(None)
            worker.join()

        print("Workers stopped.\nTrying to stop writers.")
        for writer, out_queue in zip(writers, out_queues):
            while writer.is_alive():
                time.sleep(5)
                out_queue.put(None)
            writer.join()

        print("All complete.")


    def champ_vs_roles_elite_lower(self):
        dataloader_elite = data_loader.SortedNextItemsDataLoader(app_constants.train_paths[
                                                                     "next_items_processed_elite_sorted_complete"])
        dataloader_lower = data_loader.SortedNextItemsDataLoader(app_constants.train_paths[
                                                                     "next_items_processed_lower_sorted_complete"])
        X_elite, _ = dataloader_elite.get_train_data()
        X_elite = np.unique(X_elite[:, Input.indices['start']['champs']:Input.indices['end']['champs']], axis=0)
        print("Elite done")

        X_lower, _ = dataloader_lower.get_train_data()
        X_lower = np.unique(X_lower[:, Input.indices['start']['champs']:Input.indices['end']['champs']], axis=0)
        print("Lower done")
        X = np.concatenate([X_elite, X_lower], axis=0)
        self.champs_vs_roles = dict()
        champ_configs = np.unique(X, axis=0)
        print(champ_configs.shape)
        for champ_config in champ_configs:
            self.stat_champs_vs_roles(champ_config)
        champs_vs_roles_rel = self.calc_champ_role_stats()[0]
        with open(app_constants.asset_paths["champ_vs_roles"], "w") as f:
            f.write(json.dumps(champs_vs_roles_rel, separators=(',', ':')))



if __name__ == "__main__":
    # p = ProcessPositionsTrainingData(50000, arrow.Arrow(2019, 7, 14, 0, 0, 0))
    # p.start()

    regions = ["KR", "EUW", "NA", "EUNE", "BR", "TR", "LAS", "LAN", "RU", "JP", "OCE"]
    l = ProcessNextItemsTrainingData()
    # l.champ_vs_roles_elite_lower()

    # games_by_top_leagues = [4000, 3000,2000,1000,950,900,850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350,
    #                             300, 250, 200, 150, 100, 50]
    # games_by_top_leagues = [4000, 3000,300,300,300,300,300, 300, 300, 300, 300, 300, 200, 200, 200, 200, 200,
    #                             200, 200, 200, 150, 100, 50]


    #takes roughly 1 hour to download 1000 games
    #about 1800 pure challenger games are played across all regions in a day
    number_of_top_games = 12000
    number_of_lower_games = 24000

    start_date = cass.Patch.latest(region="NA").start
    # start_date = start_date.shift(days=1)
    #### start_date = arrow.Arrow(2019, 11, 28, 0, 0, 0)
    l.start(number_of_top_games, number_of_lower_games,regions=regions, start_date=start_date)

    # l.start_processing([3422740467], "NA", ".", ".", ".", 1,0, "lol")
    s = train.PositionsTrainer()
    s.train()
    l.update_roles()
    t = train.ChampsEmbeddingTrainer()
    t.load_champ_item_dist()
    t.build_champ_embeddings_model()
    t = NextItemsTrainer()

    print("NOW TRAINING EARLY GAME")
    try:
        t.build_next_items_standard_game_model()
    except Exception as e:
        print(e)
    print("NOW TRAINING LATE GAME")
    try:
        t.build_next_items_late_game_model()
    except Exception as e:
        print(e)
    print("NOW TRAINING BOOTS GAME")
    try:
        b = BootsTrainer()
        b.train()
    except Exception as e:
        print(e)

    print("NOW TRAINING starter GAME")
    try:
        st = StarterItemsTrainer()
        st.train()
    except Exception as e:
        print(e)

    print("NOW TRAINING first item GAME")
    try:
        fi = FirstItemsTrainer()
        fi.train()
    except Exception as e:
        print(e)

    # print("NOW TRAINING champ img GAME")
    # try:
    #     ci = ChampImgTrainer()
    #     ci.train()
    # except Exception as e:
    #     print(e)





    #1. uncomment positions model
    #
