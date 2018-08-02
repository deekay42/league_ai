import json
from random import *
import tensorflow as tf
import math
import numpy as np
import glob
from itertools import zip_longest
import utils
from tflearn.data_utils import shuffle, to_categorical

TRAIN_TEST_SPLIT=0.95

MAX_ITEMS_PER_CHAMP = 6
EXAMPLES_PER_CHUNK = 100000
CHAMPS_PER_GAME = 10
SPELLS_PER_CHAMP = 2
SPELLS_PER_GAME = SPELLS_PER_CHAMP * CHAMPS_PER_GAME
NUM_FEATURES = CHAMPS_PER_GAME + CHAMPS_PER_GAME * SPELLS_PER_CHAMP + CHAMPS_PER_GAME * MAX_ITEMS_PER_CHAMP

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
            print("Building numpy database now. This may take a few minutes.")
            self.buildNumpyDB()

        try:
            self.readFromNumpyFiles()
            return
        except FileNotFoundError as error:
            repr(error)
            print("Unable to read numpy files. Is your disc full or do you not have write access to directory?")

    def get_train_data(self):
        return np.array(self.train_x), np.array(self.train_y)

    def get_test_data(self):
        return np.array(self.test_x), np.array(self.test_y)

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

    @staticmethod
    def _uniformShuffle(l1, l2):
        assert len(l1) == len(l2)
        rng_state = np.random.get_state()
        np.random.shuffle(l1)
        np.random.set_state(rng_state)
        np.random.shuffle(l2)

    # def prepareNextEpoch(self):
    #     self._uniformShuffle(self.train_x_filenames, self.train_y_filenames)
    #     self.current_training_x_y = self._getNextTrainingFile()
    #     self.training_counter = 0

    #This function converts the input data into a set of tfrecords files
    def buildNumpyDB(self):
        with open("training_data/unprocessed/final_compressed_split_x_test") as f:
            self.raw_x = json.load(f)
        with open("training_data/unprocessed/final_compressed_split_y_test") as f:
            self.raw_y = json.load(f)

        self.data_x_y = dict()
        print("Generating input & output vectors...")
        for game_x, game_y in zip(self.raw_x, self.raw_y):
            team1_team_champs = np.array(game_x['participants'][:5])
            team2_team_champs = np.array(game_x['participants'][5:])
            converter = utils.Converter()
            team1_team_champs = [converter.champ_id2int(champ) for champ in team1_team_champs]
            team2_team_champs = [converter.champ_id2int(champ) for champ in team2_team_champs]

            #next items could be shorter than absolute items because at match end there are no next item predictions, or the losing could continue buying
            next_items = game_y['winningTeamNextItems']
            absolute_items = game_x['itemsTimeline'][:len(next_items)]

            for items_x, items_y in zip(absolute_items, next_items):
                team1_team_items_at_time_x = items_x[:5]
                team1_team_items_at_time_x = [np.pad(player_items, (0, MAX_ITEMS_PER_CHAMP-len(player_items)), 'constant', constant_values=(0, 0)) for player_items in team1_team_items_at_time_x]
                team1_team_items_at_time_x = np.ravel(team1_team_items_at_time_x).astype(int)

                team2_team_items_at_time_x = items_x[5:]
                team2_team_items_at_time_x = [np.pad(player_items, (0, MAX_ITEMS_PER_CHAMP - len(player_items)), 'constant',
                                                      constant_values = (
                                                          0, 0)) for player_items in team2_team_items_at_time_x]
                team2_team_items_at_time_x = np.ravel(team2_team_items_at_time_x).astype(int)

                team1_team_items_at_time_x = [converter.item_id2int(int(item)) for item in team1_team_items_at_time_x]
                team2_team_items_at_time_x = [converter.item_id2int(int(item)) for item in team2_team_items_at_time_x]

                x = tuple(np.concatenate([team1_team_champs,
                                    team2_team_champs,
                                    team1_team_items_at_time_x,
                                    team2_team_items_at_time_x], 0))
                y = [converter.item_id2int(item) for item in items_y]

                # don't include dupes. happens when someone buys a potion and consumes it
                if x not in self.data_x_y:
                    self.data_x_y[x] = y

        print("Writing to disk...")
        self.writeToNumpyFile(EXAMPLES_PER_CHUNK)

    def champ2id(self, champname):
        try:
            return str(self.lookup_table[champname])
        except AttributeError:
            with open("res/champ2id") as f:
                self.lookup_table = json.load(f)
            try:
                return str(self.lookup_table[champname])
            except KeyError as k:
                print("Key {} not found".format(k))

    def readFromNumpyFiles(self):
        # Creates a dataset that reads all of the examples from filenames.
        self.train_x_filenames = sorted(glob.glob('training_data/processed/*_train_x.npz'))
        self.test_x_filenames = sorted(glob.glob('training_data/processed/*_test_x.npz'))
        self.train_y_filenames = sorted(glob.glob('training_data/processed/*_train_y.npz'))
        self.test_y_filenames = sorted(glob.glob('training_data/processed/*_test_y.npz'))

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


    def writeToNumpyFile(self, chunksize):
        def _chunks(l, n):
            n = max(1, n)
            return [l[i:i + n] for i in range(0, len(l), n)]

        x = list(self.data_x_y.keys())
        y = list(self.data_x_y.values())
        counter = 0
        print("Now writing numpy files to disk")
        for x_chunk, y_chunk in zip(_chunks(x, chunksize), _chunks(y, chunksize)):
            with open('training_data/processed/'+str(counter)+'_train_x.npz', "wb") as writer:
                np.savez_compressed(writer, x_chunk)
            with open('training_data/processed/'+str(counter)+'_train_y.npz', "wb") as writer:
                np.savez_compressed(writer, y_chunk)

            counter += 1
            print("{}% complete".format(int(min(100, 100*(counter*chunksize/len(x))))))

# l = DataLoader()