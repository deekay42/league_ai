import glob
import itertools
from abc import ABC, abstractmethod
from collections import Counter
import os
import numpy as np
import json

from constants import app_constants


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

    def __init__(self):
        self.train_filenames = sorted(glob.glob(app_constants.train_paths["next_items_processed_unsorted"] + 'train_*.npz'))
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


    def get_test_data(self):
        if not self.test:
            self.read_test_from_np_files()
        return self.test[:, :-1], self.test[:,-1]

    def get_train_data(self):
        if not self.train:
            self.read_train_from_np_files()
        return self.train[:, :-1], self.train[:, -1]


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



class PositionsDataLoader(DataLoaderBase):

    def __init__(self):
        self.train_x_filenames = sorted(glob.glob(app_constants.train_paths["positions_processed"] + 'train_x*.npz'))
        self.test_x_filenames = sorted(glob.glob(app_constants.train_paths["positions_processed"] + 'test_x*.npz'))
        super().__init__()


    def get_train_data(self):
        if not self.train_x:
            self.read_train_from_np_files()
        result_x, result_y = [], []
        order = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
        progress_counter = 0
        for x in self.train_x:
            progress_counter += 1
            comp_order = ((comp, pos) for comp, pos in zip(x, order))
            for champ_position in itertools.permutations(comp_order, 5):
                champ_position = np.array(champ_position)
                team_comp = np.stack(champ_position[:, 0])
                positions = np.stack(champ_position[:, 1])
                result_x.append(
                    np.concatenate((np.ravel(team_comp[:, 0]), np.ravel(team_comp[:, 1:3]), np.ravel(team_comp[:, 3:])),
                                   axis=0))
                result_y.append(np.ravel(positions))
            print("training data {:.2%} generated".format(progress_counter / len(self.train_x)))
        return np.array(result_x), np.array(result_y)


    def get_test_data(self):
        if not self.test_x:
            self.read_test_from_np_files()
        result_x, result_y = [], []
        order = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
        for x in self.test_x:

            comp_order = ((comp, pos) for comp, pos in zip(x, order))
            for champ_position in itertools.permutations(comp_order, 5):
                champ_position = np.array(champ_position)
                team_comp = np.stack(champ_position[:, 0])
                positions = np.stack(champ_position[:, 1])
                result_x.append(
                    np.concatenate((np.ravel(team_comp[:, 0]), np.ravel(team_comp[:, 1:3]), np.ravel(team_comp[:, 3:])),
                                   axis=0))
                result_y.append(np.ravel(positions))
        return np.array(result_x), np.array(result_y)

    def read_test_from_np_files(self):
        if not self.test_x_filenames or not self.test_y_filenames:
            raise FileNotFoundError("No test numpy files in that location")

        for i in self.test_x_filenames:
            data = np.load(i)['arr_0']
            self.test_x += list(data)


    def read_train_from_np_files(self):
        if not self.train_x_filenames or not self.train_y_filenames:
            raise FileNotFoundError("No train numpy files in that location")

        for i in self.train_x_filenames:
            data = np.load(i)['arr_0']
            self.train_x += list(data)