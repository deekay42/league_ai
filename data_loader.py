import numpy as np
import glob
from tflearn.data_utils import to_categorical
import itertools
import threading
from threading import Thread


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
        self.threadLock = threading.Lock()
        super().__init__()

    class GenerateTrainingData(Thread):

        def __init__(self, train_x, train_y, return_x, return_y, counter, threadLock):
            self.train_x = train_x
            self.train_y = train_y
            self.return_x = return_x
            self.return_y = return_y
            self.counter = counter
            self.threadLock = threadLock


        def generate_train_data(self):
            result_train_x, result_train_y = [], []
            pos_indicator = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
            for x, y in zip(self.train_x, self.train_y):
                for pos in range(5):
                    y_summ = y[pos]
                    if y_summ == 0:
                        continue
                    else:
                        result_train_x += [np.concatenate([pos_indicator[pos], x])]
                        result_train_y += [y_summ]
            return np.array(result_train_x), np.array(result_train_y)

        def run(self):
            print("Thread " + str(counter) + " started")
            x,y = self.generate_train_data()
            with self.threadLock:
                self.return_x.extend(x)
                self.return_y.extend(y)
            print("Thread "+str(counter)+" complete")

    def _generate_train_test_data(self, train_test_x, train_test_y):
        num_threads = 4
        collective_x = []
        collective_y = []
        threads = []
        for counter, (chunk_x, chunk_y) in enumerate(zip(np.array_split(train_test_x, num_threads), np.array_split(train_test_y, num_threads))):
            thread = GenerateTrainingData(chunk_x, chunk_y, collective_x, collective_y, counter, self.threadLock)
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()
        return collective_x, collective_y


    def get_train_data(self):
        return self._generate_train_test_data(self.train_x, self.train_y)

    def get_test_data(self):
        return self._generate_train_test_data(self.test_x, self.test_y)

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