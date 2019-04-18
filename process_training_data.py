import copy
import glob
import json
import os
import time
import traceback
from collections import Counter
from multiprocessing import Process

import numpy as np

import AssetManager
import build_path
import cass_configured as cass
import constants
import predict
import train
import utils
from jq import jq


class ProcessTrainingData:

    def __init__(self):
        self.converter = AssetManager.AssetManager()
        self.role_predictor = None
        self.sort_order = ["top", "jq", "mid", "adc", "sup"]
        self.jq_base_path = "jq/"
        self.out_base_path = "training_data/next_items/"
        self.out_processed_path = self.out_base_path + "processed/"
        self.sorted_matches_path = self.out_base_path + "matches_sorted"
        self.jq_script_names = ["itemUndos_robust", "sortEqualTimestamps", "buildAbsoluteItemTimeline",
                                "extractNextItemsForWinningTeam"]
        self.jq_scripts = {}
        for name in self.jq_script_names:
            with open(self.jq_base_path + name, "r") as f:
                self.jq_scripts[name] = f.read()
        self.jq_scripts["merged"] = self.jq_scripts["itemUndos_robust"] + ' | ' + self.jq_scripts[
            "sortEqualTimestamps"] + ' | ' + self.jq_scripts["buildAbsoluteItemTimeline"]

    def build_positions_model():
        # scrape_data.scrape_matches()

        self.buildNumpyDBForPositions()
        train.train_positions_network()


    # input is a list of games
    def run_all_transformations(self, input_, log_info):

        if log_info != None:
            print(log_info + " Starting determine roles")
        matches_sorted = self.determine_roles(matches=input_)
        if log_info != None:
            print(log_info + " Determine roles complete.\nStarting jq_merged")
        absolute_items = jq(self.jq_scripts["merged"]).transform(matches_sorted)
        if log_info != None:
            print(log_info + " jq_merged complete.\nStarting Inflate items")
        abs_inf = self.inflate_items(matches=absolute_items)
        if log_info != None:
            print(log_info + " inflate items complete.\nStarting jq_next.")
        next_items = jq(self.jq_scripts["extractNextItemsForWinningTeam"]).transform(abs_inf)
        if log_info != None:
            print(log_info + " jq_next complete.\nStarting build_db")
        final_train_data = self.build_np_db_for_next_items(abs_inf, next_items)

        return final_train_data

    def build_next_items_model_parallel(self, num_threads):
        with open("training_data/next_items/matches_presorted", "r") as psorted:
            matches = json.load(psorted)

        class ProcessTrainingDataWorker(Process):
            def __init__(self, games, transformations, chunksize, out_dir, thread_index, train_test_split=0.85):
                super().__init__()

                self.games = games
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

                self.train_test_split_point = int(len(data_dict) * self.train_test_split)
                train_dict = dict(list(data_dict.items())[:self.train_test_split_point])
                test_dict = dict(list(data_dict.items())[self.train_test_split_point:])

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
                    train_dict = self.run_transformations(self.games)
                except Exception as e:
                    print(f"ERROR: There was an error transforming these matches!!")
                    print(repr(e))
                    print(traceback.format_exc())


                self.write_next_item_chunk_to_numpy_file(train_dict)
                print(f"Thread {self.thread_index} complete")

        threads = []
        self.remove_old_files()
        for i, sub_list in enumerate(utils.split(matches, num_threads)):
            threads.append(
                ProcessTrainingDataWorker(sub_list, lambda input_, thread_index=i: self.run_all_transformations(input_, "Thread "+str(thread_index)),
                                          10, self.out_processed_path, i))
        for thread in threads:
            thread.start()
        print("Processing started")
        for thread in threads:
            thread.join()
        print("All complete.")

    def remove_old_files(self):
        old_filenames = glob.glob(self.out_processed_path + '*')
        for filename in old_filenames:
            os.remove(filename)

    def get_team_positions(self, team):
        if not self.role_predictor:
            self.role_predictor = predict.PredictRoles()

        data = np.array([[participant['championId'], participant['spell1Id'], participant['spell2Id'],
                          participant['kills'], participant['deaths'], participant['assists'],
                          participant['earned'], participant['level'],
                          participant['minionsKilled'], participant['neutralMinionsKilled'], participant['wardsPlaced']]
                         for
                         participant in team], dtype=np.str)

        champs = np.stack(data[:, 0])
        spells = np.ravel(np.stack(data[:, 1:3]))
        rest = np.array(np.ravel(np.stack(data[:, 3:])), dtype=np.uint16)

        return self.role_predictor.predict(champs, spells, rest)

    def determine_roles(self, matches, out_path=None):

        first = True

        if out_path:
            fsorted = open(out_path, "w")
            fsorted.write('[')

        result = []
        # progress_counter = 0
        for match in matches:
            gameId = match["gameId"]
            events = match["itemsTimeline"]
            sorted = match["sorted"]
            teams = match["participants"]
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
                winning_team_positioned = self.get_team_positions(winning_team)
                winning_team = []
                for position in self.sort_order:
                    winning_team.append({"championId": winning_team_positioned[position],
                                         "participantId": champid2participantid[winning_team_positioned[position]]})
                losing_team = [{"championId": champ["championId"], "participantId": champ["participantId"]} for
                               champ in losing_team]

            # losing team is unsorted
            elif sorted == "1":
                losing_team_positioned = self.get_team_positions(losing_team)
                losing_team = []
                for position in self.sort_order:
                    losing_team.append({"championId": losing_team_positioned[position],
                                        "participantId": champid2participantid[losing_team_positioned[position]]})
                winning_team = [{"championId": champ["championId"], "participantId": champ["participantId"]} for
                                champ in winning_team]
            # both are already sorted
            elif sorted == "1,2":
                winning_team = [{"championId": champ["championId"], "participantId": champ["participantId"]} for
                                champ in winning_team]
                losing_team = [{"championId": champ["championId"], "participantId": champ["participantId"]} for champ in
                               losing_team]
            # none are sorted
            elif sorted == "0":
                winning_team_positioned = self.get_team_positions(winning_team)
                winning_team = []
                for position in self.sort_order:
                    winning_team.append({"championId": winning_team_positioned[position],
                                         "participantId": champid2participantid[
                                             winning_team_positioned[position]]})
                losing_team_positioned = self.get_team_positions(losing_team)
                losing_team = []
                for position in self.sort_order:
                    losing_team.append({"championId": losing_team_positioned[position],
                                        "participantId": champid2participantid[
                                            losing_team_positioned[position]]})
            result.append({"gameId": gameId, "participants": winning_team + losing_team,
                           "itemsTimeline": events})
            if out_path:
                fsorted.write(json.dumps(result[-1], separators=(',', ':')))
                fsorted.flush()
            # print("determine roles: current file {:.2%} processed".format(progress_counter / len(matches)))
            # progress_counter += 1

        if out_path:
            fsorted.write(']')
            fsorted.close()
        return result

    @staticmethod
    def list_diff(first, second):
        diff = Counter()
        for item in first:
            diff[item] += 1
        for item in second:
            diff[item] -= 1
        diff = list(diff.elements())
        assert len(diff) <= 1
        if diff == []:
            return []
        else:
            return diff[0]

    def inflate_items(self, matches, out_file_name=None):

        result = []
        if out_file_name:
            f = open(out_file_name, "w")
            f.write('[')
        for i, game in enumerate(matches):
            # print("{0:.0%} complete".format(i / len(matches)))
            out_itemsTimeline = []
            if i > 0 and out_file_name:
                f.write(',')
            prev_state = copy.deepcopy(game['itemsTimeline'][0])
            out_itemsTimeline += [copy.deepcopy(prev_state)]
            for item_state in game['itemsTimeline']:
                summ_index = -1
                for summ_items, prev_summ_items in zip(item_state, prev_state):
                    summ_index += 1
                    # identical item states get skipped this way
                    if summ_items == prev_summ_items:
                        continue
                    else:
                        new_item = ProcessTrainingData.list_diff(summ_items, prev_summ_items)
                        if new_item:
                            new_item = cass.Item(id=int(new_item), region="KR")
                            l = new_item.name
                            insert_item_states = build_path.build_path(prev_state[summ_index], new_item)[2]
                            if len(insert_item_states) > 1:
                                for summ_item_state in insert_item_states:
                                    if len(summ_item_state) <= constants.MAX_ITEMS_PER_CHAMP:
                                        out_itemsTimeline += [copy.deepcopy(prev_state)]
                                        out_itemsTimeline[-1][summ_index] = summ_item_state
                            else:
                                out_itemsTimeline += [item_state]
                            break
                        else:
                            out_itemsTimeline += [item_state]
                prev_state = copy.deepcopy(item_state)
            result.append({"gameId": game['gameId'], "participants": game['participants'],
                           "itemsTimeline": out_itemsTimeline})
            if out_file_name:
                f.write(json.dumps(result[-1], separators=(',', ':')))
                f.write("\n")
                f.flush()
        if out_file_name:
            f.write(']')
            f.close()
        return result

    def build_np_db_for_next_items(self, abs_inf, next_items):


        result = dict()
        for game_x, game_y in zip(abs_inf, next_items):
            team1_team_champs = np.array(game_x['participants'][:5])
            team2_team_champs = np.array(game_x['participants'][5:])
            converter = AssetManager.AssetManager()
            team1_team_champs = [converter.champ_id2champ_int(champ) for champ in team1_team_champs]
            team2_team_champs = [converter.champ_id2champ_int(champ) for champ in team2_team_champs]

            # next items could be shorter than absolute items because at match end there are no next item predictions, or the losing could continue buying
            next_items = game_y['winningTeamNextItems']
            absolute_items = game_x['itemsTimeline'][:len(next_items)]

            for items_x, items_y in zip(absolute_items, next_items):
                team1_team_items_at_time_x = items_x[:5]
                team1_team_items_at_time_x = [
                    np.pad(player_items, (0, constants.MAX_ITEMS_PER_CHAMP - len(player_items)), 'constant',
                           constant_values=(0, 0)) for player_items in team1_team_items_at_time_x]
                team1_team_items_at_time_x = np.ravel(team1_team_items_at_time_x).astype(int)

                team2_team_items_at_time_x = items_x[5:]
                team2_team_items_at_time_x = [
                    np.pad(player_items, (0, constants.MAX_ITEMS_PER_CHAMP - len(player_items)), 'constant',
                           constant_values=(
                               0, 0)) for player_items in team2_team_items_at_time_x]
                team2_team_items_at_time_x = np.ravel(team2_team_items_at_time_x).astype(int)

                team1_team_items_at_time_x = [converter.item_id2item_int(int(item)) for item in
                                              team1_team_items_at_time_x]
                team2_team_items_at_time_x = [converter.item_id2item_int(int(item)) for item in
                                              team2_team_items_at_time_x]

                x = tuple(np.concatenate([team1_team_champs,
                                          team2_team_champs,
                                          team1_team_items_at_time_x,
                                          team2_team_items_at_time_x], 0))
                # empty items should be set to 0, not empty list
                y = [0 if i == [] else i for i in items_y]
                y = [converter.item_id2item_int(item) for item in y]

                # don't include dupes. happens when someone buys a potion and consumes it
                if x not in result:
                    result[x] = y

        return result

    def buildNumpyDBForPositions(self):
        print("Building numpy database now. This may take a few minutes.")
        # if len(sys.argv) != 2:
        #     print("specify gamefile")
        #     exit(-1)
        # self.x_filename = sys.argv[1]
        self.x_filename = "scrape/matches_old"
        print("Loading input files")
        with open(self.x_filename) as f:
            self.raw = json.load(f)
        print("Complete")

        self.data_x = []
        print("Generating input & output vectors...")
        sorted_counter = 0
        progress_counter = -1

        for game in self.raw:
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
                    [[self.converter.champ_id2champ_int(str(participant['championId'])), self.converter.spell_id2int(
                        participant['spell1Id']), self.converter.spell_id2int(
                        participant['spell2Id']), participant['kills'], participant['deaths'], participant['assists'],
                      participant['earned'], participant['level'],
                      participant['minionsKilled'], participant['neutralMinionsKilled'], participant['wardsPlaced']] for
                     participant in team])
                self.data_x.append(x)

            print("current file {:.2%} processed".format(progress_counter / len(self.raw)))
        print(f"{sorted_counter} teams were in the right order out of {2 * len(self.raw)}")

        print("Writing to disk...")
        self.writePositionsToNumpyFile(chunksize=10)

    def writePositionsToNumpyFile(self, chunksize=100000, train_test_split=0.15):
        old_filenames = glob.glob('training_data/positions/processed/train_x*.npz')
        old_filenames.extend(glob.glob('training_data/positions/processed/test_x*.npz'))
        for filename in old_filenames:
            os.remove(filename)

        print("Now writing numpy files to disk")
        splitpoint = len(self.data_x) * (1 - train_test_split)
        # new_file_name_x = self.x_filename[self.x_filename.rfind("/") + len('structuredForRole') + 1:]
        for i, x_chunk in enumerate(utils.chunks(self.data_x, chunksize)):
            with open('training_data/positions/processed/' + (
                    'test_x_' if i * chunksize > splitpoint else 'train_x') + str(i) + '.npz',
                      "wb") as writer:
                np.savez_compressed(writer, x_chunk)
            print("{}% complete".format(int(min(100, 100 * (i * chunksize / len(self.data_x))))))

    # no need to shuffle here. only costs time. shuffling will happen during training before each epoch
    @staticmethod
    def _uniformShuffle(l1, l2):
        assert len(l1) == len(l2)
        rng_state = np.random.get_state()
        np.random.shuffle(l1)
        np.random.set_state(rng_state)
        np.random.shuffle(l2)


if __name__ == "__main__":
    p = ProcessTrainingData()
    start = time.time()
    p.build_next_items_model_parallel(2)
    end = time.time()
    print(f"It took {end - start} seconds")
