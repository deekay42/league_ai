import os
import os
import time
from collections import Counter
from functools import reduce
from itertools import compress
from multiprocessing import Process, Queue

import numpy as np
from jq import jq

from constants import game_constants
from get_train_data import scrape_data
from train_model import data_loader, train
from utils import build_path, cass_configured as cass
from utils.artifact_manager import *
import arrow
from utils import utils
from train_model.model import NextItemEarlyGameModel


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

    def __init__(self, out_queue, chunksize, out_dir):
        super().__init__()
        self.out_queue = out_queue
        self.chunksize = chunksize
        self.out_dir = out_dir


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
        filename = out_dir + f"train_{chunk_counter}.npz"
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
    def __init__(self, out_queue, chunksize, out_dir_inf, out_dir_uninf):
        super().__init__(out_queue, chunksize, None)
        self.out_dir_inf = out_dir_inf
        self.out_dir_uninf = out_dir_uninf

    def run(self):
        chunk_counter = 0
        terminated = False
        while not terminated:
            chunk_inf = {}
            chunk_uninf = {}
            for i in range(self.chunksize):
                next_item = self.out_queue.get()
                if next_item is None:
                    while not self.out_queue.empty():
                        self.out_queue.get()
                    terminated = True
                    break
                chunk_inf.update(next_item[0])
                chunk_uninf.update(next_item[1])

            self.write_chunk(chunk_inf, chunk_counter, self.out_dir_inf)
            self.write_chunk(chunk_uninf, chunk_counter, self.out_dir_uninf)
            chunk_counter += 1
        print(f"Writer {self.out_dir_inf} {self.out_dir_uninf} complete")


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
    def run_next_item_transformations(self, training_data):
        transformations = [self.sort_equal_timestamps, self.remove_undone_items, self.insert_null_items,
                           self.build_abs_timeline,
                           self.post_process]
        result = training_data
        for transformation in transformations:
            result = transformation(result)
        return result


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


    def remove_undone_items(self, matches):
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


    def update_roles(self):
        unsorted_positions_dl = data_loader.PositionsToBePredDataLoader()
        unsorted_positions = unsorted_positions_dl.read()
        gameId2roles = self.determine_roles(unsorted_positions)
        unsorted_processed_dataloader_inf = data_loader.UnsortedNextItemsDataLoader(app_constants.train_paths[
                                                                                    "next_items_processed_unsorted_inf"])

        unsorted_processed_data_inf = unsorted_processed_dataloader_inf.get_train_data()
        sorted_processed_data_inf = self.apply_roles_to_unsorted_processed(gameId2roles, unsorted_processed_data_inf)
        self.write_chunk(sorted_processed_data_inf, app_constants.train_paths["next_items_processed_sorted_inf"])

        unsorted_processed_dataloader_uninf = data_loader.UnsortedNextItemsDataLoader(app_constants.train_paths[
                                                                                          "next_items_processed_unsorted_uninf"])
        unsorted_processed_data_uninf = unsorted_processed_dataloader_uninf.get_train_data()
        sorted_processed_data_uninf = self.apply_roles_to_unsorted_processed(gameId2roles,
                                                                             unsorted_processed_data_uninf)
        self.write_chunk(sorted_processed_data_uninf, app_constants.train_paths["next_items_processed_sorted_uninf"])


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


    def apply_roles_to_unsorted_processed(self, match_id2perm, unsorted_processed):
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
                gameIds = [[int(gameId)]] * current_game.shape[0]
                permutation = match_id2perm[gameId]
                new_pos_map = {i:permutation.index(i) for i in [0,1,2,3,4]}
                updated_positions = [[new_pos_map[pos[0]]] for pos in current_game[:, pos_start:pos_end]]


                reordered_result = np.concatenate([gameIds, updated_positions,
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
                yield np.concatenate([gameIds, current_game], axis=1)


    def post_process(self, matches):

        for i, game in enumerate(matches):
            out_uninf = []
            out_inf = []
            game = game[0]
            participantid2index = {participant['participantId']: j for j, participant in enumerate(game[
                                                                                                       'participants'])}
            delta_current_gold = [0] * 10
            prev_frame_index = -1

            champs = np.array([participant['championId'] for participant in game['participants']])
            champs = [self.champ_manager.lookup_by("id", str(champ))["int"] for champ in champs]


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
                        items_at_time_x = NextItemEarlyGameModel.encode_items(event['absolute_items'], self.item_manager)
                    except ValueError as e:
                        continue
                    y = self.item_manager.lookup_by("id", str(event['itemId']))["int"]
                    uninf_example = [[pos], champs, np.ravel(items_at_time_x),
                                                     np.around(event['total_gold']).astype(int),
                                                     np.around(event['cs']).astype(int),
                                                     np.around(event['neutral_cs']).astype(int),
                                                     np.around(event['xp']).astype(int),
                                                     np.around(event['lvl']).astype(int),
                                                     np.ravel(event['kda']).tolist(),
                                                     np.around(event['current_gold_sloped'] + np.array(delta_current_gold)).astype(int),
                                                     [y]
                                                     ]
                    out_uninf.append(np.concatenate(uninf_example, 0))
                    if event['itemId'] == 0:
                        out_inf.append(np.concatenate(uninf_example, 0))
                        continue
                    component_items,_, insert_item_states,_ = build_path.build_path(participant_current_items, new_item)
                    prev_event = event
                    for i, (component_item, insert_item_state) in enumerate(zip(component_items, insert_item_states)):
                        if NextItemEarlyGameModel.num_itemslots(Counter(insert_item_state)) > \
                                game_constants.MAX_ITEMS_PER_CHAMP:
                            continue
                        event_copy = prev_event.copy()
                        event_copy['absolute_items'] = prev_event['absolute_items'].copy()
                        event_copy['itemId'] = component_item.id

                        try:
                            items_at_time_x = NextItemEarlyGameModel.encode_items(event_copy['absolute_items'],
                                                                                  self.item_manager)
                            y = self.item_manager.lookup_by("id", str(component_item.id))["int"]
                            if y == 0:
                                print(f"y==0 gameId {str(game['gameId'])}")
                            out_inf.append(np.concatenate([[pos], champs, np.ravel(items_at_time_x),
                                                           np.around(event_copy['total_gold']).astype(int),
                                                           np.around(event_copy['cs']).astype(int),
                                                           np.around(event_copy['neutral_cs']).astype(int),
                                                           np.around(event_copy['xp']).astype(int),
                                                           np.around(event_copy['lvl']).astype(int),
                                                           np.ravel(event_copy['kda']).tolist(),
                                                           np.around(event_copy['current_gold_sloped'] + np.array(
                                                               delta_current_gold)).astype(int),
                                                           [y]
                                                           ], 0))
                        except ValueError as e:
                            pass

                        delta_current_gold[pos] -= component_item.gold.base
                        event_copy['absolute_items'][part_index] = insert_item_state
                        prev_event = event_copy
            else:
                yield [{str(game['gameId']): out_inf}, {str(game['gameId']): out_uninf}]


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
    def start(self, games_by_top_leagues, region, start_date, num_threads=os.cpu_count(), chunklen=400):
        utils.remove_old_files(app_constants.train_paths["positions_processed"])
        utils.remove_old_files(app_constants.train_paths["positions_to_be_pred"])
        in_queue = Queue()
        out_queues = [Queue(), Queue(), Queue()]
        transformations = [self.run_next_item_transformations,
                           lambda a: self.run_positions_transformations(a, 1),
                           lambda a: self.run_positions_transformations(a, 0)]
        out_keys = ["positions_processed", "positions_to_be_pred"]
        workers = [ProcessTrainingDataWorker(in_queue, out_queues, transformations, i) for i in range(num_threads)]

        writers = [NextItemsWriteResultWorker(out_queues[0], chunklen, app_constants.train_paths[
            "next_items_processed_unsorted_inf"], app_constants.train_paths["next_items_processed_unsorted_uninf"])]
        writers.extend([WriteResultWorker(out_queue, chunklen, app_constants.train_paths[out_key]) for out_key,
                                                                                                       out_queue
                   in zip(out_keys, out_queues[1:])])


        for process in workers + writers:
            process.start()

        # with open(app_constants.train_paths["presorted_matches_path"], "w") as f:
        #     f.write('[\n')
        for i, match in enumerate(scrape_data.scrape_matches(games_by_top_leagues, region, start_date)):
            # if i > 0:
            #     f.write(',')
            # f.write(json.dumps(match, separators=(',', ':')))
            # f.flush()
            in_queue.put([match])
            print(f"Match {i}")
        # f.write('\n]')

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



if __name__ == "__main__":
    # p = ProcessPositionsTrainingData(50000, arrow.Arrow(2019, 7, 14, 0, 0, 0))
    # p.start()
    region = "EUW"
    l = ProcessNextItemsTrainingData()
    # games_by_top_leagues = [1500, 1400, 1300, 1200, 1100, 1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]
    games_by_top_leagues = [4000, 3000,2000,1000,950,900,850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350,
                                300, 250, 200, 150, 100, 50]
    start_date = cass.Patch.latest(region="EUW").start
    # start_date = arrow.Arrow(2019, 11, 28, 0, 0, 0)
    l.start(games_by_top_leagues=games_by_top_leagues,region=region, start_date=start_date)
    # s = train.PositionsTrainer()
    # s.train()
    # l.update_roles()

    # unsorted_processed_dataloader = data_loader.UnsortedNextItemsDataLoader()
    # unsorted_processed_data = unsorted_processed_dataloader.get_train_data()
    # print("lol")
