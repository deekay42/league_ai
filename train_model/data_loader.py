import glob
import itertools
from abc import ABC, abstractmethod
from collections import Counter
import os
import numpy as np
import json

from constants import app_constants, game_constants
from tflearn.data_utils import to_categorical
from utils.artifact_manager import ChampManager, ItemManager
import sklearn
from train_model.input_vector import Input, InputWinPred

legacy_indices = dict()
legacy_indices["start"] = dict()
legacy_indices["half"] = dict()
legacy_indices["end"] = dict()
legacy_indices["mid"] = dict()

legacy_indices["start"]["gameid"] = 0
legacy_indices["end"]["gameid"] = legacy_indices["start"]["gameid"] + 1

legacy_indices["start"]["pos"] = legacy_indices["end"]["gameid"]
legacy_indices["end"]["pos"] = legacy_indices["start"]["pos"] + 1

legacy_indices["start"]["champs"] = legacy_indices["end"]["pos"]
legacy_indices["half"]["champs"] = legacy_indices["start"]["champs"] + game_constants.CHAMPS_PER_TEAM
legacy_indices["end"]["champs"] = legacy_indices["start"]["champs"] + game_constants.CHAMPS_PER_GAME

legacy_indices["start"]["items"] = legacy_indices["end"]["champs"]
legacy_indices["half"]["items"] = legacy_indices["start"][
                                   "items"] + game_constants.MAX_ITEMS_PER_CHAMP * 2 * game_constants.CHAMPS_PER_TEAM
legacy_indices["end"]["items"] = legacy_indices["start"][
                                  "items"] + game_constants.MAX_ITEMS_PER_CHAMP * 2 * game_constants.CHAMPS_PER_GAME

legacy_indices["start"]["total_gold"] = legacy_indices["end"]["items"]
legacy_indices["half"]["total_gold"] = legacy_indices["start"]["total_gold"] + game_constants.CHAMPS_PER_TEAM
legacy_indices["end"]["total_gold"] = legacy_indices["start"]["total_gold"] + game_constants.CHAMPS_PER_GAME

legacy_indices["start"]["cs"] = legacy_indices["end"]["total_gold"]
legacy_indices["half"]["cs"] = legacy_indices["start"]["cs"] + game_constants.CHAMPS_PER_TEAM
legacy_indices["end"]["cs"] = legacy_indices["start"]["cs"] + game_constants.CHAMPS_PER_GAME

legacy_indices["start"]["neutral_cs"] = legacy_indices["end"]["cs"]
legacy_indices["half"]["neutral_cs"] = legacy_indices["start"]["neutral_cs"] + game_constants.CHAMPS_PER_TEAM
legacy_indices["end"]["neutral_cs"] = legacy_indices["start"]["neutral_cs"] + game_constants.CHAMPS_PER_GAME

legacy_indices["start"]["xp"] = legacy_indices["end"]["neutral_cs"]
legacy_indices["half"]["xp"] = legacy_indices["start"]["xp"] + game_constants.CHAMPS_PER_TEAM
legacy_indices["end"]["xp"] = legacy_indices["start"]["xp"] + game_constants.CHAMPS_PER_GAME

legacy_indices["start"]["lvl"] = legacy_indices["end"]["xp"]
legacy_indices["half"]["lvl"] = legacy_indices["start"]["lvl"] + game_constants.CHAMPS_PER_TEAM
legacy_indices["end"]["lvl"] = legacy_indices["start"]["lvl"] + game_constants.CHAMPS_PER_GAME

legacy_indices["start"]["kda"] = legacy_indices["end"]["lvl"]
legacy_indices["half"]["kda"] = legacy_indices["start"]["kda"] + game_constants.CHAMPS_PER_TEAM * 3
legacy_indices["end"]["kda"] = legacy_indices["start"]["kda"] + game_constants.CHAMPS_PER_GAME * 3

legacy_indices["start"]["current_gold"] = legacy_indices["end"]["kda"]
legacy_indices["half"]["current_gold"] = legacy_indices["start"]["current_gold"] + game_constants.CHAMPS_PER_TEAM
legacy_indices["end"]["current_gold"] = legacy_indices["start"]["current_gold"] + game_constants.CHAMPS_PER_GAME

legacy_indices["start"]["baron"] = legacy_indices["end"]["current_gold"]
legacy_indices["half"]["baron"] = legacy_indices["start"]["baron"] + 1
legacy_indices["end"]["baron"] = legacy_indices["start"]["baron"] + 2

legacy_indices["start"]["elder"] = legacy_indices["end"]["baron"]
legacy_indices["half"]["elder"] = legacy_indices["start"]["elder"] + 1
legacy_indices["end"]["elder"] = legacy_indices["start"]["elder"] + 2

legacy_indices["start"]["dragons_killed"] = legacy_indices["end"]["elder"]
legacy_indices["half"]["dragons_killed"] = legacy_indices["start"]["dragons_killed"] + 4
legacy_indices["end"]["dragons_killed"] = legacy_indices["start"]["dragons_killed"] + 8

legacy_indices["start"]["dragon_soul"] = legacy_indices["end"]["dragons_killed"]
legacy_indices["half"]["dragon_soul"] = legacy_indices["start"]["dragon_soul"] + 1
legacy_indices["end"]["dragon_soul"] = legacy_indices["start"]["dragon_soul"] + 2

legacy_indices["start"]["dragon_soul_type"] = legacy_indices["end"]["dragon_soul"]
legacy_indices["half"]["dragon_soul_type"] = legacy_indices["start"]["dragon_soul_type"] + 4
legacy_indices["end"]["dragon_soul_type"] = legacy_indices["start"]["dragon_soul_type"] + 8

legacy_indices["start"]["turrets_destroyed"] = legacy_indices["end"]["dragon_soul_type"]
legacy_indices["half"]["turrets_destroyed"] = legacy_indices["start"]["turrets_destroyed"] + 1
legacy_indices["end"]["turrets_destroyed"] = legacy_indices["start"]["turrets_destroyed"] + 2

legacy_indices["start"]["first_team_blue"] = legacy_indices["end"]["turrets_destroyed"]
legacy_indices["end"]["first_team_blue"] = legacy_indices["start"]["first_team_blue"] + 1

total_len = legacy_indices["end"]["first_team_blue"]

class DataLoaderBase(ABC):

    def __init__(self):
        self.train_x = []
        self.train_y = []
        self.test_x = []
        self.test_y = []

    @abstractmethod
    def read_test_from_np_files(self):
        pass

    @abstractmethod
    def read_train_from_np_files(self):
       pass

    @abstractmethod
    def get_train_data(self):
        pass


    @abstractmethod
    def get_test_data(self):
        pass


class FullDataLoader:

    def __init__(self):
        self.train_filename = app_constants.train_paths["presorted_matches_path"]
        self.train = []

    def get_train_data(self):
        with open(self.train_filename) as f:
            return json.load(f)


class UnsortedNextItemsDataLoader:

    def __init__(self, path):
        self.train_filenames = sorted(glob.glob(path + 'train_*.npz'))
        self.train = {}

    def get_train_data(self):
        if not self.train:
            self.read_train_from_np_files()
        return self.train

    def read_train_from_np_files(self):
        if not self.train_filenames:
            raise FileNotFoundError("No numpy files in that location")
        for i in self.train_filenames:
            data = np.load(i)
            self.train.update(data)


class SortedNextItemsDataLoader(DataLoaderBase):

    def __init__(self, path):
        self.train_filenames = sorted(glob.glob(path + 'train_*.npz'))
        self.test_filenames = sorted(glob.glob(path + 'test_*.npz'))
        self.train = []
        self.test = []



    @staticmethod
    def _generate_train_test(train_test_x, train_test_y):
        x = np.concatenate(
            (np.tile(np.arange(5), len(train_test_x))[:, np.newaxis], np.repeat(train_test_x, 5, axis=0)), axis=1)
        y = np.ravel(train_test_y)

        valid_ind = y != 0
        result_y = y[valid_ind]
        result_x = x[valid_ind]
        return result_x, result_y


    def get_y_distribution(self):
        _, y_results = self._generate_train_test(self.train_x, self.train_y)
        return Counter(y_results)


    def get_test_data(self, cond=None):
        if not self.test:
            self.read_test_from_np_files()
        if cond:
            self.test = self.test[cond(self.test)]
        # X = self.legacy_transform(self.test)
        X = self.test[:, :-1]
        Y = self.test[:, -1].astype(np.int32)
        return X,Y

    def legacy_transform(self, train_test):
        X = np.zeros((train_test.shape[0], Input.len))

        slice_names = {"gameid", "pos", "items", "champs", "total_gold", "lvl", "current_gold", "baron", "elder",
                       "dragons_killed",
                       "dragon_soul_type", "turrets_destroyed"}
        for slice_name in slice_names:
            X[:, Input.indices["start"][slice_name]:Input.indices["end"][slice_name]] = \
                train_test[:, legacy_indices["start"][slice_name]:legacy_indices["end"][slice_name]]

        kills = train_test[:, legacy_indices["start"]["kda"]:legacy_indices["end"]["kda"]:3]
        deaths = train_test[:, legacy_indices["start"]["kda"] + 1:legacy_indices["end"]["kda"]:3]
        assists = train_test[:, legacy_indices["start"]["kda"] + 2:legacy_indices["end"]["kda"]:3]
        X[:, Input.indices["start"]["kills"]:Input.indices["end"]["kills"]] = kills
        X[:, Input.indices["start"]["deaths"]:Input.indices["end"]["deaths"]] = deaths
        X[:, Input.indices["start"]["assists"]:Input.indices["end"]["assists"]] = assists

        blues = train_test[:, legacy_indices["start"]["first_team_blue"]].astype(np.int32)
        blue_side = np.ones((train_test.shape[0], 2))
        blue_side[np.arange(blues.shape[0]), blues] = 0
        X[:, Input.indices["start"]["blue_side"]:Input.indices["end"]["blue_side"]] = blue_side

        total_cs = train_test[:, legacy_indices["start"]["cs"]:legacy_indices["end"]["cs"]] + train_test[:,
                                                                                              legacy_indices["start"][
                                                                                                  "neutral_cs"]:
                                                                                              legacy_indices[
                                                                                                  "end"][
                                                                                                  "neutral_cs"]]
        X[:, Input.indices["start"]["cs"]:Input.indices["end"]["cs"]] = total_cs
        return X


    def get_train_data(self, cond=None):
        if not self.train:
            self.read_train_from_np_files()
        if cond:
            self.train = self.train[cond(self.train)]

        # X = self.legacy_transform(self.train)
        X = self.train[:, :-1]
        Y = self.train[:,-1].astype(np.int32)
        # self.stat_items()
        return X, Y


    def stat_items(self):
        stats = dict()
        for example in self.train:
            pos = example[1]
            champ_int = example[pos+2]
            item_name = ItemManager().lookup_by("int", example[-1])["name"]
            if not champ_int in stats:
                stats[champ_int] = dict()
            if not pos in stats[champ_int]:
                stats[champ_int][pos] = Counter()
            if not item_name in stats[champ_int][pos]:
                stats[champ_int][pos][item_name] = 1
            else:
                stats[champ_int][pos][item_name] += 1
        return stats


    def get_item_distrib_by_champ(self):
        if not self.train:
            self.read_train_from_np_files()

        champ_distrib = {champ_int:[0]*ItemManager().get_num("int") for champ_int in ChampManager().get_ints()}

        full_item_ints = [item_int for item_int in ItemManager().get_completes()]
        prev_game_id = self.train[0][0]
        for i, example in enumerate(self.train):
            if example[0] != prev_game_id:
                prev_game_id = example[0]
                final_prev_example = self.train[i-1]
                for j, champ_int in enumerate(final_prev_example[2:12]):
                    champ_items = final_prev_example[(j+1)*12:(j+2)*12:2]
                    valid_i = np.isin(champ_items, full_item_ints)
                    for item_i in champ_items[valid_i]:
                        champ_distrib[champ_int][item_i] += 1


        normalized_d = dict()
        for champ, distrib in champ_distrib.items():
            normalized_d[champ] = (np.array(distrib) / sum(distrib))

        x = np.array(list(normalized_d.keys()))[1:]
        y = np.array(list(normalized_d.values()))[1:]

        min_max_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
        y = min_max_scaler.fit_transform(y)


        return x, y


    def get_item_distrib_vs_champ(self):
        if not self.train:
            self.read_train_from_np_files()

        champ_distrib = {champ_int:[0]*ItemManager().get_num("int") for champ_int in ChampManager().get_ints()}

        nonsituational_items_blackout = [item_int for item_int in ItemManager().get_situationals()]
        nonsituational_items_blackout = np.array(list(set(range(ItemManager().get_num("int"))) - set(
            nonsituational_items_blackout))).astype(np.int32)
        complete_items = [item_int for item_int in ItemManager().get_completes()]

        prev_game_id = self.train[0][0]
        for i, example in enumerate(self.train):
            if example[0] != prev_game_id:
                prev_game_id = example[0]
                final_prev_example = self.train[i-1]
                team_items = [[],[]]
                for team in range(2):
                    for j, champ_int in enumerate(final_prev_example[2+5*team:7+5*team]):
                        champ_items = final_prev_example[(j+1)*12+team*5*12:(j+2)*12+team*5*12:2]
                        valid_i = np.isin(champ_items, complete_items)
                        team_items[team].extend(champ_items[valid_i])


                for team in range(2):
                    for j, champ_int in enumerate(final_prev_example[2+5*team:7+5*team]):
                        for item_i in team_items[int(not team)]:
                            champ_distrib[champ_int][item_i] += 1

        normalized_d = dict()
        for champ, distrib in champ_distrib.items():
            normalized_d[champ] = (np.array(distrib)/sum(distrib))

        x = np.array(list(normalized_d.keys()))[1:]
        y = np.array(list(normalized_d.values()))[1:]


        min_max_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
        y = min_max_scaler.fit_transform(y)
        y[:, nonsituational_items_blackout] = 0

        return x,y


    def get_item_distrib_by_champ_v2(self):
        if not self.train:
            self.read_train_from_np_files()
        # self.train = self.train[:1000]

        result_x, result_y = [], []

        full_item_ints = [item_int for item_int in ItemManager().get_completes()]
        prev_game_id = self.train[0][0]
        for i, example in enumerate(self.train):
            if example[0] != prev_game_id:
                prev_game_id = example[0]
                final_prev_example = self.train[i-1]
                for j, champ_int in enumerate(final_prev_example[2:12]):
                    champ_items = final_prev_example[(j+1)*12:(j+2)*12:2]
                    valid_i = np.isin(champ_items, full_item_ints)
                    items_k_hot = np.sum(to_categorical(champ_items[valid_i], nb_classes=ItemManager().get_num("int")), axis=0)
                    result_x.append(champ_int)
                    result_y.append(items_k_hot)
        return result_x, result_y


    def get_test_data_raw(self):
        if not self.test:
            self.read_test_from_np_files()

        X, Y = self.test[:, :-1], self.test[:, -1]
        return X, Y


    def get_train_data_raw(self):
        if not self.train:
            self.read_train_from_np_files()

        X, Y = self.train[:, :-1], self.train[:, -1]
        return X, Y


    def get_all_unfolded(self):
        if not self.train_x:
            self.read_train_from_np_files()
        if not self.test_x:
            self.read_test_from_np_files()
        return self.train_x, self.train_y, self.test_x, self.test_y


    def read_train_from_np_files(self):
        self.train = self.read_train_test_from_np_files(self.train_filenames)


    def read_test_from_np_files(self):
        self.test = self.read_train_test_from_np_files(self.test_filenames)


    def read_train_test_from_np_files(self, filenames):
        if not filenames:
            raise FileNotFoundError("No numpy files in that location")
        result = []
        for i in filenames:
            data = np.load(i)['arr_0']
            result += list(data)
        return np.array(result)


class PositionsToBePredDataLoader:

    def __init__(self):
        self.train_x_filenames = sorted(glob.glob(app_constants.train_paths["positions_to_be_pred"] + 'train*.npz'))
        super().__init__()


    def read(self):
        if not self.train_x_filenames:
            raise FileNotFoundError("No numpy files in that location")
        result = dict()
        for i in self.train_x_filenames:
            data = dict(np.load(i))
            result.update(data)
        return result


class PositionsDataLoader(DataLoaderBase):

    def __init__(self):
        self.train_x_filenames = sorted(glob.glob(app_constants.train_paths["positions_processed"] + 'train*.npz'))
        self.test_x_filenames = sorted(glob.glob(app_constants.train_paths["positions_processed"] + 'test*.npz'))
        super().__init__()


    def permutate_data(self, data):
        order = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
        perms = list(itertools.permutations([0, 1, 2, 3, 4], 5))
        x_perms = np.reshape([data[:, perm] for perm in perms], (len(data)*len(perms), -1))
        y_perms = np.reshape(np.repeat(np.array([order[perm,] for perm in perms]),len(data), axis=0),
                             (len(data)*len(perms),-1))
        return x_perms, y_perms


    def get_train_data(self):
        if not self.train_x:
            self.read_train_from_np_files()
        return self.permutate_data(self.train_x)


    def get_test_data(self):
        if not self.test_x:
            self.read_test_from_np_files()
        return self.permutate_data(self.test_x)


    def read_test_from_np_files(self):
        if not self.test_x_filenames:
            raise FileNotFoundError("No test numpy files in that location")

        for i in self.test_x_filenames:
            data = list(dict(np.load(i)).values())
            self.test_x += data
        self.test_x = np.array(self.test_x)



    def read_train_from_np_files(self):
        if not self.train_x_filenames:
            raise FileNotFoundError("No train numpy files in that location")

        for i in self.train_x_filenames:
            data = list(dict(np.load(i)).values())
            self.train_x += data
        self.train_x = np.array(self.train_x)


    def train2test(self):
        for index, train_fname in enumerate(self.train_x_filenames[:round(len(self.train_x_filenames)/10 + 1)]):
            train_index = train_fname.rfind("train")
            os.rename(train_fname, train_fname[:train_index] + f"test_{index}.npz")
