import copy
import glob
import os
import traceback
from collections import Counter
from multiprocessing import Process
import arrow

import numpy as np
from jq import jq

from constants import game_constants, app_constants
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
        self.write_positions_to_np_file(chunksize=10)


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
        self.jq_scripts["merged"] = self.jq_scripts["itemUndos_robust"] + ' | ' + self.jq_scripts[
            "sortEqualTimestamps"] + ' | ' + self.jq_scripts["buildAbsoluteItemTimeline"]


    # input is a list of games
    def run_all_transformations(self, input_, log_info):

        if log_info is not None:
            print(log_info + " Starting determine roles")
        matches_sorted = self.determine_roles(matches=input_)
        if log_info is not None:
            print(log_info + " Determine roles complete.\nStarting jq_merged")
        absolute_items = jq(self.jq_scripts["merged"]).transform(matches_sorted)
        if log_info is not None:
            print(log_info + " jq_merged complete.\nStarting Inflate items")
        abs_inf = self.inflate_items(matches=absolute_items)
        if log_info is not None:
            print(log_info + " inflate items complete.\nStarting jq_next.")
        next_items = jq(self.jq_scripts["extractNextItemsForWinningTeam"]).transform(abs_inf)
        if log_info is not None:
            print(log_info + " jq_next complete.\nStarting build_db")
        final_train_data = self.build_np_db_for_next_items(abs_inf, next_items)

        return final_train_data


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
                for position in game_constants.ROLE_ORDER:
                    winning_team.append({"championId": winning_team_positioned[position],
                                         "participantId": champid2participantid[winning_team_positioned[position]]})
                losing_team = [{"championId": champ["championId"], "participantId": champ["participantId"]} for
                               champ in losing_team]

            # losing team is unsorted
            elif sorted == "1":
                losing_team_positioned = self.get_team_positions(losing_team)
                losing_team = []
                for position in game_constants.ROLE_ORDER:
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
                for position in game_constants.ROLE_ORDER:
                    winning_team.append({"championId": winning_team_positioned[position],
                                         "participantId": champid2participantid[
                                             winning_team_positioned[position]]})
                losing_team_positioned = self.get_team_positions(losing_team)
                losing_team = []
                for position in game_constants.ROLE_ORDER:
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


    def get_team_positions(self, team):
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

        input_ = np.concatenate([champ_ints, spell_ints, rest], axis=0)

        # apparently this is needed to use tensorflow with multiprocessing:
        # https://github.com/tensorflow/tensorflow/issues/8220
        if not self.role_predictor:
            from train_model import model
            self.role_predictor = model.PositionsModel()

        return self.role_predictor.predict(input_)


    @staticmethod
    def inflate_items(matches, out_file_name=None):
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
                    assert(0 not in summ_items)
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


    def start(self, num_threads=2):
        with open(app_constants.train_paths["presorted_matches_path"], "r") as psorted:
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
                    train_dict = self.run_transformations(self.games)
                    self.write_next_item_chunk_to_numpy_file(train_dict)
                    print(f"Thread {self.thread_index} complete")
                except Exception as e:
                    print(f"ERROR: There was an error transforming these matches!!")
                    print(repr(e))
                    print(traceback.format_exc())


        threads = []
        utils.remove_old_files(app_constants.train_paths["next_items_processed"])
        for i, sub_list in enumerate(utils.split(matches, num_threads)):
            threads.append(
                ProcessTrainingDataWorker(sub_list, lambda input_,
                                                           thread_index=i: self.run_all_transformations(input_,
                                                                                                        "Thread " + str(
                                                                                                            thread_index)),
                                          10, app_constants.train_paths["next_items_processed"], i))
        for thread in threads:
            thread.start()
        print("Processing started")
        for thread in threads:
            thread.join()
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


    def build_np_db_for_next_items(self, abs_inf, next_items):
        result = dict()
        for game_x, game_y in zip(abs_inf, next_items):
            team1_team_champs = np.array(game_x['participants'][:5])
            team2_team_champs = np.array(game_x['participants'][5:])

            team1_team_champs = [self.champ_manager.lookup_by("id", str(champ))["int"] for champ in team1_team_champs]
            team2_team_champs = [self.champ_manager.lookup_by("id", str(champ))["int"] for champ in team2_team_champs]

            # next items could be shorter than absolute items because at match end there are no next item predictions, or the losing could continue buying
            next_items = game_y['winningTeamNextItems']
            absolute_items = game_x['itemsTimeline'][:len(next_items)]

            for items_x, items_y in zip(absolute_items, next_items):
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
    # p = ProcessPositionsTrainingData(100, arrow.Arrow(2019, 4, 25, 0, 0, 0))
    p = ProcessNextItemsTrainingData()
    p.start()
