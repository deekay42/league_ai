import glob
import itertools
from abc import ABC, abstractmethod

import numpy as np

from constants import app_constants


class DataLoaderBase(ABC):

    def __init__(self):
        self.train_x = []
        self.train_y = []
        self.test_x = []
        self.test_y = []
        self.read_from_np_files()


    @abstractmethod
    def read_from_np_files(self):
        pass

    @abstractmethod
    def get_train_data(self):
        pass


    @abstractmethod
    def get_test_data(self):
        pass


class NextItemsDataLoader(DataLoaderBase):

    def __init__(self):
        self.train_x_filenames = sorted(glob.glob(app_constants.train_paths["next_items_processed"] + 'train_x*.npz'))
        self.test_x_filenames = sorted(glob.glob(app_constants.train_paths["next_items_processed"] + 'test_x*.npz'))
        self.train_y_filenames = sorted(glob.glob(app_constants.train_paths["next_items_processed"] + 'train_y*.npz'))
        self.test_y_filenames = sorted(glob.glob(app_constants.train_paths["next_items_processed"] + 'test_y*.npz'))
        super().__init__()


    @staticmethod
    def _generate_train_test(train_test_x, train_test_y):
        x = np.concatenate(
            (np.tile(np.arange(5), len(train_test_x))[:, np.newaxis], np.repeat(train_test_x, 5, axis=0)), axis=1)
        y = np.ravel(train_test_y)

        valid_ind = y != 0
        result_y = y[valid_ind]
        result_x = x[valid_ind]
        return result_x, result_y


    def get_test_data(self):
        return self._generate_train_test(self.test_x, self.test_y)


    def get_train_data(self):
        return self._generate_train_test(self.train_x, self.train_y)


    def read_from_np_files(self):
        if not self.train_x_filenames or not self.train_y_filenames \
                or not self.test_x_filenames or not self.test_y_filenames:
            raise FileNotFoundError("No train or test numpy files in that location")

        for i in self.train_x_filenames:
            data = np.load(i)['arr_0']
            self.train_x += list(data)
        for i in self.train_y_filenames:
            data = np.load(i)['arr_0']
            self.train_y += list(data)
        for i in self.test_x_filenames:
            data = np.load(i)['arr_0']
            self.test_x += list(data)
        for i in self.test_y_filenames:
            data = np.load(i)['arr_0']
            self.test_y += list(data)


class PositionsDataLoader(DataLoaderBase):

    def __init__(self):
        self.train_x_filenames = sorted(glob.glob(app_constants.train_paths["positions_processed"] + 'train_x*.npz'))
        self.test_x_filenames = sorted(glob.glob(app_constants.train_paths["positions_processed"] + 'test_x*.npz'))
        super().__init__()


    def get_train_data(self):
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


    def read_from_np_files(self):
        if not self.train_x_filenames or not self.test_x_filenames:
            raise FileNotFoundError("No train or test numpy files in that location")

        for i in self.train_x_filenames:
            data = np.load(i)['arr_0']
            self.train_x += list(data)
        for i in self.test_x_filenames:
            data = np.load(i)['arr_0']
            self.test_x += list(data)
