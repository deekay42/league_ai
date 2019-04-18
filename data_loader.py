import numpy as np
import glob
import itertools
# import h5py
# import h5py_cache as h5c
from threading import Lock, Thread
from queue import Queue
import time

TRAIN_TEST_SPLIT=0.95





class DataLoaderBase:
    def __init__(self):
        self.train_x = []
        self.train_y = []
        self.test_x = []
        self.test_y = []
        self.readFromNumpyFiles()

class NextItemsDataLoader(DataLoaderBase):

    def __init__(self):
        self.train_x_filenames = sorted(glob.glob('training_data/next_items/processed/train_x*.npz'))
        self.test_x_filenames = sorted(glob.glob('training_data/next_items/processed/test_x*.npz'))
        self.train_y_filenames = sorted(glob.glob('training_data/next_items/processed/train_y*.npz'))
        self.test_y_filenames = sorted(glob.glob('training_data/next_items/processed/test_y*.npz'))
        # h5f = h5py.File('training_data/next_items/hdf5/data.h5', 'r')
        # self.train_x_hdf5 = h5f['train_x']
        # self.train_y_hdf5 = h5f['train_y']
        # self.test_x_hdf5 = h5f['test_x']
        # self.test_y_hdf5 = h5f['test_y']

        super().__init__()



    def _generate_train_test(self, train_test_x, train_test_y):
        x = np.concatenate((np.tile(np.arange(5),len(train_test_x))[:,np.newaxis], np.repeat(train_test_x, 5, axis=0)), axis=1)
        y = np.ravel(train_test_y)
        valid_ind = y != 0
        result_y = y[valid_ind]
        result_x = x[valid_ind]
        return result_x, result_y

    def build_HDF5(self):
        self.readFromNumpyFiles()
        print("Files loaded")
        print("Generating training data")
        self.train_x, self.train_y = self._generate_train_test(self.train_x, self.train_y)
        self.test_x, self.test_y = self._generate_train_test(self.test_x, self.test_y)
        print("Training data complete")

        # train_x_y = np.concatenate([self.train_x, self.train_y[:,np.newaxis]], axis = 1)
        # self.train_x, self.train_y = [], []
        # print("Now shuffling")
        # np.random.shuffle(train_x_y)
        # print("Shuffling complete")
        # shuffled = []

        # for i in range(0, len(train_x_y), 10 ** 6):
        #     print("Shuffling {:.2%} complete".format(i / len(train_x_y)))
        #     seg = train_x_y[i:i + 10**6]
        #     np.random.shuffle(seg)
        #     shuffled.extend(seg)

        # print("get arrays back in shape")
        # self.train_x = train_x_y[:, :-1]
        # self.train_y = train_x_y[:, -1]

        print("writing files")
        # chunk_shape_x = (self.train_x.shape[-1],1)
        # chunk_shape_y = (1,)
        h5f = h5c.File('training_data/next_items/hdf5/data.h5', 'w')
        print("0 complete")
        h5f.create_dataset('train_x', data=self.train_x, shape=self.train_x.shape)
        print("1 complete")
        h5f.create_dataset('train_y', data=self.train_y, shape=self.train_y.shape)
        print("2 complete")
        h5f.create_dataset('test_x', data=self.test_x, shape=self.test_x.shape)
        print("3 complete")
        h5f.create_dataset('test_y', data=self.test_y, shape=self.test_y.shape)
        print("4 complete")
        h5f.close()


    class GenerateTrainingData(Thread):
        def __init__(self, queue, gen_x_y):
            Thread.__init__(self)
            self.queue = queue
            self.generate_x_y = gen_x_y

        def run(self):
            while True:
                x, y = self.generate_x_y()
                self.queue.put((x, y))

    def _get_next_examples(self):
        start_index = self.randints.pop()
        end_index = min(start_index+self.batchsize, self.train_len)
        return self.train_x_hdf5[start_index: end_index], self.train_y_hdf5[start_index: end_index]

    def get_next_train_subepoch(self, num_examples):
        if not self.threads:
            # self.
            for _ in range(1):
                t = self.GenerateTrainingData(self.queue, self._get_next_examples)
                self.threads.append(t)
                t.start()
        X,Y = [], []
        for _ in range(num_examples):
            (x,y) = self.queue.get()
            X.append(x)
            Y.append(y)
        return (X,Y)

    def get_test_data(self):
        # X,Y = [],[]
        # for i in range(self.test_x_hdf5.shape[0]):
        #     X.append(self.test_x_hdf5[i])
        #     Y.append(self.test_y_hdf5[i])

        # return self.test_x_hdf5, self.test_y_hdf5
        return self._generate_train_test(self.test_x, self.test_y)
    def get_train_data(self):
        # return self.train_x_hdf5, self.train_y_hdf5
        return self._generate_train_test(self.train_x, self.train_y)


    # def get_train_data(self, pos):
    #     self.readFromNumpyFiles()
    #     train_x, train_y = [], []
    #     for x, y in zip(self.train_x, self.train_y):
    #         y_summ = y[pos]
    #         if y_summ == 0:
    #             continue
    #         else:
    #             train_x += [x]
    #             train_y += [y_summ]
    #     self.train_x = np.array(train_x)
    #     self.train_y = np.array(train_y)
    #     return self.train_x, self.train_y
    #
    # def get_test_data(self, pos):
    #     test_x, test_y = [], []
    #     for x, y in zip(self.test_x, self.test_y):
    #         y_summ = y[pos]
    #         if y_summ == 0:
    #             continue
    #         else:
    #             test_x += [x]
    #             test_y += [y_summ]
    #     self.test_x = np.array(test_x)
    #     self.test_y = np.array(test_y)
    #     return self.test_x, self.test_y

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
        self.train_x_filenames = sorted(glob.glob('training_data/positions/processed/train_x*.npz'))
        self.test_x_filenames = sorted(glob.glob('training_data/positions/processed/test_x*.npz'))
        super().__init__()

    def get_train_data(self):
        result_x, result_y = [], []
        order = [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]
        progress_counter = 0
        for x in self.train_x:
            progress_counter += 1
            comp_order = ((comp,pos) for comp, pos in zip(x, order))
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



if __name__ == "__main__":
    d = NextItemsDataLoader()
    d.build_HDF5()
    # h5f = h5py.File('training_data/next_items/hdf5/data.h5', 'r')
    # train_x = h5f['train_x']
    # print(train_x.shape)
    # rando = np.random.randint(24000000, size=128*100).reshape((100,128))
    # for i in rando:
    #     t1 = time.time()
    #     tmp = train_x[sorted(list(i))]
    #     t2 = time.time()-t1
    #     print(t2)

