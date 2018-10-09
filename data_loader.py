import numpy as np
import glob
from tflearn.data_utils import to_categorical
import itertools

TRAIN_TEST_SPLIT=0.95

class DataLoaderBase:
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


class NextItemsDataLoader(DataLoaderBase):

    def __init__(self):
        self.train_x_filenames = sorted(glob.glob('training_data/next_items/processed/*_train_x*.npz'))
        self.test_x_filenames = sorted(glob.glob('training_data/next_items/processed/*_test_x*.npz'))
        self.train_y_filenames = sorted(glob.glob('training_data/next_items/processed/*_train_y*.npz'))
        self.test_y_filenames = sorted(glob.glob('training_data/next_items/processed/*_test_y*.npz'))
        super().__init__()

    def get_train_data(self):
        train_x, train_y = [], []
        pos_indicator = [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]
        for x, y in zip(self.train_x, self.train_y):
            for pos in range(5):
                y_summ = y[pos]
                if y_summ == 0:
                    continue
                else:
                    train_x += [np.concatenate([pos_indicator[pos], x])]
                    train_y += [y_summ]
        self.train_x = np.array(train_x)
        self.train_y = np.array(train_y)
        return self.train_x, self.train_y

    def get_test_data(self):
        test_x, test_y = [], []
        pos_indicator = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
        for x, y in zip(self.test_x, self.test_y):
            for pos in range(5):
                y_summ = y[pos]
                if y_summ == 0:
                    continue
                else:
                    test_x += [np.concatenate([pos_indicator[pos], x])]
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


class PositionsDataLoader(DataLoaderBase):

    def __init__(self):
        self.train_x_filenames = sorted(glob.glob('training_data/positions/processed/*_train_x*.npz'))
        self.test_x_filenames = sorted(glob.glob('training_data/positions/processed/*_test_x*.npz'))
        super().__init__()

    def get_train_data(self):
        result_x, result_y = [], []
        order = [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]
        for x in self.train_x:


            comp_order = ((comp,pos) for comp, pos in zip(x, order))
            for champ_position in itertools.permutations(comp_order, 5):
                champ_position = np.array(champ_position)
                team_comp = np.stack(champ_position[:, 0])
                positions = np.stack(champ_position[:, 1])
                result_x.append(
                    np.concatenate((np.ravel(team_comp[:, 0]), np.ravel(team_comp[:, 1:3]), np.ravel(team_comp[:, 3:])),
                                   axis=0))
                result_y.append(np.ravel(positions))
        return np.array(result_x), np.array(result_y)


    def get_test_data(self):
        result_x, result_y = [], []
        order = [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]
        for x in self.test_x:


            comp_order = ((comp,pos) for comp, pos in zip(x, order))
            for champ_position in itertools.permutations(comp_order, 5):
                champ_position = np.array(champ_position)
                team_comp = np.stack(champ_position[:, 0])
                positions = np.stack(champ_position[:, 1])
                result_x.append(
                    np.concatenate((np.ravel(team_comp[:, 0]), np.ravel(team_comp[:, 1:3]), np.ravel(team_comp[:, 3:])),
                                   axis=0))
                result_y.append(np.ravel(positions))
        return np.array(result_x), np.array(result_y)


    def readFromNumpyFiles(self):

        if not self.train_x_filenames or not self.test_x_filenames:
            raise FileNotFoundError("No train or test numpy files in that location")

        for i in self.train_x_filenames:
            data = np.load(i)['arr_0']
            self.train_x += list(data)
        for i in self.test_x_filenames:
            data = np.load(i)['arr_0']
            self.test_x += list(data)