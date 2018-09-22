import json
from random import *
import math
import numpy as np
import glob
from itertools import zip_longest
import utils
from tflearn.data_utils import shuffle, to_categorical
import sys

TRAIN_TEST_SPLIT=0.95


class DataLoader:

    def __init__(self):
        self.train_x = []
        self.train_y = []
        self.test_x = []
        self.test_y = []
        try:
            self.readFromNumpyFiles()
            return
        except FileNotFoundError as error:
            repr(error)
            print("Unable to read numpy files. Did you build the database first?")

    def get_train_data(self, pos):
        train_x, train_y = [], []
        for x, y in zip(self.train_x, self.train_y):
            y_summ = y[pos]
            if y_summ == 0:
                continue
            else:
                train_x += [x]
                train_y += [y_summ]
        self.train_x = np.array(train_x)
        self.train_y = np.array(train_y)
        return self.train_x, self.train_y

    def get_test_data(self, pos):
        test_x, test_y = [], []
        for x, y in zip(self.test_x, self.test_y):
            y_summ = y[pos]
            if y_summ == 0:
                continue
            else:
                test_x += [x]
                test_y += [y_summ]
        self.test_x = np.array(test_x)
        self.test_y = np.array(test_y)
        return self.test_x, self.test_y

    def transform_X_to_one_hot(self, X, game_config):
        champs_per_game = game_config["champs_per_game"]
        total_num_champs = game_config["total_num_champs"]
        total_num_items = game_config["total_num_items"]
        champs_one_hot = to_categorical(X[0:champs_per_game], nb_classes=total_num_champs)
        items_one_hot = to_categorical(X[champs_per_game:], nb_classes=total_num_items)
        result = np.concatenate([np.ravel(champs_one_hot), np.ravel(items_one_hot)])
        return result

    def transform_Y_to_one_hot(self, Y, game_config):
        items_one_hot = to_categorical(Y, nb_classes=game_config["total_num_items"])
        return np.ravel(items_one_hot)

    def readFromNumpyFiles(self):
        self.train_x_filenames = sorted(glob.glob('training_data/processed/*_train_x*.npz'))
        self.test_x_filenames = sorted(glob.glob('training_data/processed/*_test_x*.npz'))
        self.train_y_filenames = sorted(glob.glob('training_data/processed/*_train_y*.npz'))
        self.test_y_filenames = sorted(glob.glob('training_data/processed/*_test_y*.npz'))

        if not self.train_x_filenames or not self.train_y_filenames or not self.test_x_filenames or not self.test_y_filenames:
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