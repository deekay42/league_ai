import glob
import itertools
from abc import ABC, abstractmethod
from collections import Counter
import os
import numpy as np
import json

from constants import app_constants
from tflearn.data_utils import to_categorical
from utils.artifact_manager import ChampManager, ItemManager
import sklearn


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
            print("Condition given")
            self.test = self.test[cond(self.test)]
        X, Y = self.test[:, 1:-1], self.test[:, -1]
        return X,Y

    def get_train_data(self, cond=None):
        if not self.train:
            self.read_train_from_np_files()
        if cond:
            print("Condition given")
            self.train = self.train[cond(self.train)]
        X,Y = self.train[:, 1:-1], self.train[:, -1]
        return X,Y


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
        if not self.test_x:
            self.read_test_from_np_files()
        return self.test_x, self.test_y


    def get_train_data_raw(self):
        if not self.train_x:
            self.read_train_from_np_files()
        return self.train_x, self.train_y


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
