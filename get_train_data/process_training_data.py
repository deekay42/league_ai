import glob
import os
from collections import Counter
from functools import reduce
from itertools import compress
from multiprocessing import Process, Queue

import gc
import numpy as np
from jq import jq

from constants import game_constants
from get_train_data import scrape_data
from train_model import train
from utils import utils, build_path, cass_configured as cass
from utils.artifact_manager import *


class ProcessPositionsTrainingData:

    def __init__(self, num_matches, cut_off_date):
        self.num_matches = num_matches
        self.cut_off_date = cut_off_date
        self.data_x = []
        self.champ_manager = ChampManager()
        self.spell_manager = SpellManager()


    def start(self):
        scrape_data.scrape_matches(self.num_matches, self.cut_off_date)
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
        self.spell_manager = SpellManager()
        self.role_predictor = None

        self.data_x = None
        self.jq_scripts = {}
        for name in app_constants.jq_script_names:
            with open(app_constants.jq_base_path + name, "r") as f:
                self.jq_scripts[name] = f.read()
        self.jq_scripts["merged"] = self.jq_scripts[
                                        "sortEqualTimestamps"] + ' | ' + self.jq_scripts["buildAbsoluteItemTimeline"]


    # input is a list of games
    def run_all_transformations(self, training_data, log_info):

        if log_info is not None:
            print(log_info + " Starting determine roles")
        training_data = self.determine_roles(matches=training_data, log_info=log_info)

        if log_info is not None:
            print(log_info + " Determine roles complete.\n" + log_info + " Starting item undos roles")
        training_data = self.remove_undone_items(training_data, log_info)
        if log_info is not None:
            print(log_info + " Item undos complete.\n" + log_info + " Starting sort equal timestamp")
        training_data = self.sort_equal_timestamps(training_data, log_info)
        if log_info is not None:
            print(log_info + " Sort equal timestamp complete.\n" + log_info + " Starting absolute items")
        # with open("output", "w") as f:
        #     f.write(json.dumps(training_data))
        # absolute_items = self.build_absolute_item_timeline(matches_set, log_info)
        training_data = (jq(self.jq_scripts["buildAbsoluteItemTimeline"]).transform([game]) for game in training_data)
        if log_info is not None:
            print(log_info + " Absolute items complete.\n" + log_info + " Starting Inflate items")
        training_data = self.inflate_items(matches=training_data, log_info=log_info)
        if log_info is not None:
            print(log_info + " inflate items complete.\n" + log_info + " Starting jq_next.")
        next_items = (jq(self.jq_scripts["extractNextItemsForWinningTeam"]).transform([game]) for game in training_data)
        if log_info is not None:
            print(log_info + " jq_next complete.\n" + log_info + " Starting build_db")
        training_data = self.build_np_db_for_next_items(training_data, next_items, log_info=log_info)

        return training_data


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


    def sort_equal_timestamps(self, matches, log_info):

        progress_counter = 0
        for match in matches:
            events = match["itemsTimeline"]
            events.sort(key=ProcessNextItemsTrainingData.keyFunc)
            if match["gameId"] == 3776579595:
                with open("output", "w") as f:
                    f.write(json.dumps({"gameId": match['gameId'], "participants": match['participants'],
                                        "itemsTimeline": events}))
            yield {"gameId": match['gameId'], "participants": match['participants'],
                   "itemsTimeline": events}

            # print(log_info + " sort equal timestamps: current file {:.2%} processed".format(progress_counter /
            #                                                                                       len(
            #     matches)))
            progress_counter += 1


    @staticmethod
    def keyFunc(elem):
        if elem["type"] == "ITEM_SOLD":
            return elem["timestamp"] + 0.1
        elif elem["type"] == "ITEM_DESTROYED":
            return elem["timestamp"] + 0.2
        elif elem["type"] == "ITEM_PURCHASED":
            return elem["timestamp"] + 0.3
        else:
            1 / 0


    def remove_undone_items(self, matches, log_info):
        progress_counter = 0
        for match in matches:
            events = match["itemsTimeline"]
            included = np.ones([len(events)])
            for i in range(len(events) - 1, -1, -1):
                if events[i]["type"] == "ITEM_UNDO":
                    included[i] = 0
                    for j in range(i - 1, -1, -1):
                        if included[j] and (events[j]["type"] == "ITEM_PURCHASED" or
                                            events[j]["type"] == "ITEM_SOLD"):
                            if events[j]["participantId"] == events[i]["participantId"] and (events[i]["beforeId"] == events[j]["itemId"] or events[i]["afterId"] == events[j]["itemId"]) and events[i]["timestamp"] >= events[j]["timestamp"]:
                                included[j] = 0
                                break

            events = list(compress(events, included))

            yield {"gameId": match['gameId'], "participants": match['participants'],
                   "itemsTimeline": events}

            # print(log_info + " remove undone items: current file {:.2%} processed".format(progress_counter / len(
            #     matches)))
            progress_counter += 1


    def determine_roles(self, matches, log_info, out_path=None):
        first = True
        if out_path:
            fsorted = open(out_path, "w")
            fsorted.write('[')

        progress_counter = 0
        unsorted_matches = []
        unsorted_teams = []
        for match in matches:
            gameId = match["gameId"]
            events = match["itemsTimeline"]
            sorted = match["sorted"]
            teams = match["participants"]
            winning_team = teams[:5]
            losing_team = teams[5:]

            if first:
                first = False
            elif out_path:
                fsorted.write(',')

            if not self.role_predictor:
                from train_model import model
                self.role_predictor = model.PositionsModel()

            # winning team is unsorted
            if sorted == "2":
                unsorted_matches.append(match)
                unsorted_teams.append(self.team2role_predictor_input(winning_team))


            # losing team is unsorted
            elif sorted == "1":
                unsorted_matches.append(match)
                unsorted_teams.append(self.team2role_predictor_input(losing_team))

            # both are already sorted
            elif sorted == "1,2":
                winning_team = [{"championId": champ["championId"], "participantId": champ["participantId"]} for
                                champ in winning_team]
                losing_team = [{"championId": champ["championId"], "participantId": champ["participantId"]} for champ in
                               losing_team]
                yield {"gameId": gameId, "participants": winning_team + losing_team,
                       "itemsTimeline": events}
                if out_path:
                    fsorted.write(json.dumps(result[-1], separators=(',', ':')))
                    fsorted.flush()

            # none are sorted
            elif sorted == "0":
                unsorted_matches.append(match)
                unsorted_teams.append(self.team2role_predictor_input(winning_team))
                unsorted_teams.append(self.team2role_predictor_input(losing_team))

            # print(log_info + " determine roles sorted: current file {:.2%} processed".format(progress_counter / len(
            #     matches)))
            progress_counter += 1

        # apparently this is needed to use tensorflow with multiprocessing:
        # https://github.com/tensorflow/tensorflow/issues/8220
        if not self.role_predictor:
            from train_model import model
            self.role_predictor = model.PositionsModel()

        sorted_teams = self.role_predictor.multi_predict(unsorted_teams)
        team_index = 0
        progress_counter = 0

        for unsorted_match in unsorted_matches:
            gameId = unsorted_match["gameId"]
            events = unsorted_match["itemsTimeline"]
            sorted = unsorted_match["sorted"]
            teams = unsorted_match["participants"]
            winning_team = teams[:5]
            losing_team = teams[5:]
            champid2participantid = {champ["championId"]: champ["participantId"] for champ in
                                     winning_team + losing_team}
            if first:
                first = False
            elif out_path:
                fsorted.write(',')

            # winning team is unsorted
            if sorted == "2":
                winning_team_positioned = sorted_teams[team_index]
                team_index += 1
                winning_team = []
                for position in game_constants.ROLE_ORDER:
                    winning_team.append({"championId": winning_team_positioned[position],
                                         "participantId": champid2participantid[winning_team_positioned[position]]})
                losing_team = [{"championId": champ["championId"], "participantId": champ["participantId"]} for
                               champ in losing_team]

            # losing team is unsorted
            elif sorted == "1":
                losing_team_positioned = sorted_teams[team_index]
                team_index += 1
                losing_team = []
                for position in game_constants.ROLE_ORDER:
                    losing_team.append({"championId": losing_team_positioned[position],
                                        "participantId": champid2participantid[losing_team_positioned[position]]})
                winning_team = [{"championId": champ["championId"], "participantId": champ["participantId"]} for
                                champ in winning_team]

            # none are sorted
            elif sorted == "0":
                winning_team_positioned = sorted_teams[team_index]
                losing_team_positioned = sorted_teams[team_index + 1]

                team_index += 2

                winning_team = []
                for position in game_constants.ROLE_ORDER:
                    winning_team.append({"championId": winning_team_positioned[position],
                                         "participantId": champid2participantid[
                                             winning_team_positioned[position]]})

                losing_team = []
                for position in game_constants.ROLE_ORDER:
                    losing_team.append({"championId": losing_team_positioned[position],
                                        "participantId": champid2participantid[
                                            losing_team_positioned[position]]})

            yield {"gameId": gameId, "participants": winning_team + losing_team,
                   "itemsTimeline": events}
            if out_path:
                fsorted.write(json.dumps(result[-1], separators=(',', ':')))
                fsorted.flush()
            # print(log_info + " determine roles unsorted: current file {:.2%} processed".format(progress_counter / len(
            #     unsorted_matches)))
            progress_counter += 1

        if out_path:
            fsorted.write(']')
            fsorted.close()


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


    @staticmethod
    def inflate_items(matches, log_info, out_file_name=None):
        if out_file_name:
            f = open(out_file_name, "w")
            f.write('[')
        for i, game in enumerate(matches):
            game = game[0]
            # print(log_info + " inflate_items {0:.0%} complete".format(i / len(matches)))
            out_itemsTimeline = []
            if i > 0 and out_file_name:
                f.write(',')
            prev_state = [sublist.copy() for sublist in game['itemsTimeline'][0]]
            out_itemsTimeline += [[sublist.copy() for sublist in prev_state]]
            for item_state in game['itemsTimeline']:
                summ_index = -1
                for summ_items, prev_summ_items in zip(item_state, prev_state):
                    assert (0 not in summ_items)
                    summ_index += 1
                    # identical item states get skipped this way
                    if summ_items == prev_summ_items:
                        continue
                    else:
                        new_item = ProcessNextItemsTrainingData.list_diff(summ_items, prev_summ_items)
                        if new_item:
                            new_item = cass.Item(id=int(new_item), region="KR")
                            try:
                                l = new_item.name
                            except Exception as e:
                                print(repr(e))
                                print(f"ERROR: item might have been removed {new_item}")

                            insert_item_states = build_path.build_path(prev_state[summ_index], new_item)[2]
                            if len(insert_item_states) > 1:
                                for summ_item_state in insert_item_states:
                                    if len(summ_item_state) <= game_constants.MAX_ITEMS_PER_CHAMP:
                                        out_itemsTimeline += [[sublist.copy() for sublist in prev_state]]
                                        out_itemsTimeline[-1][summ_index] = summ_item_state
                            else:
                                out_itemsTimeline += [item_state]
                            break
                        else:
                            out_itemsTimeline += [item_state]
                prev_state = [sublist.copy() for sublist in item_state]
            yield {"gameId": game['gameId'], "participants": game['participants'],
                   "itemsTimeline": out_itemsTimeline}
            if out_file_name:
                f.write(json.dumps(result[-1], separators=(',', ':')))
                f.write("\n")
                f.flush()
        if out_file_name:
            f.write(']')
            f.close()


    # chunklen should be set so that app doesn't start thrashing
    def start(self, num_threads=os.cpu_count(), chunklen=1000):
        class ProcessTrainingDataWorker(Process):

            def __init__(self, queue, transformations, chunksize, out_dir, thread_index, train_test_split=0.85):
                super().__init__()
                self.queue = queue
                # ?? what is this for??
                self.chunksize = chunksize
                self.out_dir = out_dir
                self.thread_index = thread_index
                self.out_path = out_dir + str(thread_index)
                self.run_transformations = transformations
                self.train_test_split = train_test_split


            def write_next_item_chunk_to_numpy_file(self, data_dict):
                x_train_filename = self.out_dir + f"train_x_thread_{self.thread_index}.npz"
                y_train_filename = self.out_dir + f"train_y_thread_{self.thread_index}.npz"
                x_test_filename = self.out_dir + f"test_x_thread_{self.thread_index}.npz"
                y_test_filename = self.out_dir + f"test_y_thread_{self.thread_index}.npz"

                train_test_split_point = int(len(data_dict) * self.train_test_split)
                train_dict = dict(list(data_dict.items())[:train_test_split_point])
                test_dict = dict(list(data_dict.items())[train_test_split_point:])

                train_x = list(train_dict.keys())
                train_y = list(train_dict.values())
                test_x = list(test_dict.keys())
                test_y = list(test_dict.values())
                for filename, data in zip([x_train_filename, y_train_filename, x_test_filename, y_test_filename],
                                          [train_x, train_y, test_x, test_y]):
                    with open(filename,
                              "wb") as writer:
                        np.savez_compressed(writer, data)


            def run(self):
                try:
                    games = self.queue.get()
                    train_dict = self.run_transformations(games)
                    self.write_next_item_chunk_to_numpy_file(train_dict)
                    print(f"Thread {self.thread_index} complete")
                except Exception as e:
                    print(f"ERROR: There was an error transforming these matches!!")
                    print(repr(e))
                    print(traceback.format_exc())
                    raise e


        utils.remove_old_files(app_constants.train_paths["next_items_processed"])
        queue = Queue()
        thread_pool = [ProcessTrainingDataWorker(queue, lambda input_,
                                                               thread_index=i: self.run_all_transformations(input_,
                                                                                                            "   Thread "
                                                                                                            "" + str(
                                                                                                                thread_index)),
                                                 10, app_constants.train_paths["next_items_processed"], i) for i in
                       range(1000)]

        with open(app_constants.train_paths["presorted_matches_path"], "r") as psorted:
            matches = json.load(psorted)

        chunk_counter = 0

        for chunk in utils.chunks(matches, chunklen):
            threads = []
            for sub_list in utils.split(chunk, num_threads):
                queue.put(sub_list)
                threads.append(thread_pool.pop(0))

            for thread in threads:
                thread.start()
            print("Processing started")
            # free memory for main thread
            # del matches[:]
            # del matches
            # gc.collect()

            for thread in threads:
                thread.join()
            chunk_counter += 1
            print("Chunks {0:.0%} complete".format(chunk_counter / (len(matches) / chunklen)))
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


    def build_np_db_for_next_items(self, abs_inf, next_items, log_info):
        result = dict()
        i = 0
        for game_x, game_y in zip(abs_inf, next_items):
            game_y = game_y[0]
            team1_team_champs = np.array(game_x['participants'][:5])
            team2_team_champs = np.array(game_x['participants'][5:])

            team1_team_champs = [self.champ_manager.lookup_by("id", str(champ))["int"] for champ in team1_team_champs]
            team2_team_champs = [self.champ_manager.lookup_by("id", str(champ))["int"] for champ in team2_team_champs]

            # next items could be shorter than absolute items because at match end there are no next item predictions, or the losing could continue buying
            next_items = game_y['winningTeamNextItems']
            absolute_items = game_x['itemsTimeline'][:len(next_items)]

            for items_x, items_y in zip(absolute_items, next_items):

                try:
                    team1_team_items_at_time_x = items_x[:5]
                    team1_team_items_at_time_x = [
                        np.pad(player_items, (0, game_constants.MAX_ITEMS_PER_CHAMP - len(player_items)), 'constant',
                               constant_values=(0, 0)) for player_items in team1_team_items_at_time_x]
                    team1_team_items_at_time_x = np.ravel(team1_team_items_at_time_x).astype(int)

                    team2_team_items_at_time_x = items_x[5:]
                    team2_team_items_at_time_x = [
                        np.pad(player_items, (0, game_constants.MAX_ITEMS_PER_CHAMP - len(player_items)), 'constant',
                               constant_values=(
                                   0, 0)) for player_items in team2_team_items_at_time_x]
                    team2_team_items_at_time_x = np.ravel(team2_team_items_at_time_x).astype(int)
                except ValueError as e:
                    print("ERROR: Probably more than 6 items for summoner: GameId: " + str(game_x['gameId']))
                    print(repr(e))
                    raise e

                try:
                    team1_team_items_at_time_x = [self.item_manager.lookup_by("id", str(item))["int"] for item in
                                                  team1_team_items_at_time_x]
                    team2_team_items_at_time_x = [self.item_manager.lookup_by("id", str(item))["int"] for item in
                                                  team2_team_items_at_time_x]
                except KeyError as e:
                    print(e)

                x = tuple(np.concatenate([team1_team_champs,
                                          team2_team_champs,
                                          team1_team_items_at_time_x,
                                          team2_team_items_at_time_x], 0))
                # empty items should be set to 0, not empty list
                y = [0 if i == [] else i for i in items_y]

                try:
                    y = [self.item_manager.lookup_by("id", str(item))["int"] for item in y]
                except KeyError as e:
                    print(e)

                # don't include dupes. happens when someone buys a potion and consumes it
                if x not in result:
                    result[x] = y
            # print(log_info + " build_db {0:.0%} complete".format(i / len(abs_inf)))
            i += 1

        return result


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
    # l = ProcessNextItemsTrainingData()
    # l.start()

    t = train.StaticTrainingDataTrainer()
    t.build_next_items_model()
