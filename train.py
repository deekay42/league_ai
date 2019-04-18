import glob
import json
import os
from multiprocessing import Process, JoinableQueue
import multiprocessing

import cv2 as cv
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical

import data_loader
import generate
import utils
from network import *
import shutil


class MonitorCallback(tflearn.callbacks.Callback):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def on_epoch_end(self, training_state):
        self.f.write(
            "Epoch {0} accuracy {1:.2f} | loss {2:.2f}\n".format(training_state.epoch, training_state.acc_value,
                                                                 training_state.global_loss))
        self.f.flush()


class Trainer:
    def __init__(self):

        self.num_epochs = 30
        self.batch_size = 128
        self.queue = JoinableQueue(10)
        self.workers = []
        self.out_paths = dict()
        self.base_out_path = "models/"

        for label in ["train", "best"]:

            paths = { "base": self.base_out_path + label + "/"}

            paths.update({"imgs": paths["base"] + "imgs/",
            "next_items_out_path": paths["base"] + "next_items/",
            "positions_out_path": paths["base"] + "positions/"})

            paths.update({"champ_imgs": paths["imgs"] + "champs/",
            "item_imgs": paths["imgs"] + "items/",
            "self_imgs": paths["imgs"] + "self/"})
            self.out_paths[label] = paths

        self.cvt = AssetManager()

    def remove_old_files(self, path):
        old_filenames = glob.glob(path + '*')
        for filename in old_filenames:
            os.remove(filename)

    def showCoords(self, img, champ_coords, champ_size, item_coords, item_size, self_coords, self_size):
        # print(champ_coords)
        # print(item_coords)
        # print(self_coords)
        for coord in champ_coords:
            cv.rectangle(img, tuple(coord), (coord[0] + champ_size, coord[1] + champ_size), (255, 0, 0), 1)
        for coord in item_coords:
            cv.rectangle(img, tuple(coord), (coord[0] + item_size, coord[1] + item_size), (255, 0, 0), 1)
        for coord in self_coords:
            cv.rectangle(img, tuple(coord), (coord[0] + self_size, coord[1] + self_size), (255, 0, 0), 1)
        cv.imshow("lol", img)
        cv.waitKey(0)

    def load_img_test_data(self):
        image_paths = ['test_data/easy/1.png', 'test_data/easy/2.png', 'test_data/easy/3.png']
        image_paths.sort()
        imgs = [cv.imread(path) for path in image_paths]
        with open('test_data/easy/test_labels.json', "r") as f:
            elems = json.load(f)

        champs_y = [list(v['champs']) for k, v in elems.items()]
        items_y = [list(v['items']) for k, v in elems.items()]
        self_y = [v['self'] for k, v in elems.items()]

        champs_y = [[self.cvt.champ_str2img_int(champ) for champ in champ_list] for champ_list in champs_y]
        items_y = [[self.cvt.item_str2img_int(item) for item in items_list] for items_list in items_y]
        champs_y = np.ravel(champs_y)
        items_y = np.ravel(items_y)

        res_cvt = ui_constants.ResConverter(1440, 900)
        champ_coords = utils.generateChampCoordinates(res_cvt.CHAMP_LEFT_X_OFFSET, res_cvt.CHAMP_RIGHT_X_OFFSET,
                                                      res_cvt.CHAMP_Y_DIFF,
                                                      res_cvt.CHAMP_Y_OFFSET)
        item_coords = utils.generateItemCoordinates(res_cvt.ITEM_X_DIFF, res_cvt.ITEM_LEFT_X_OFFSET,
                                                    res_cvt.ITEM_RIGHT_X_OFFSET,
                                                    res_cvt.ITEM_Y_DIFF,
                                                    res_cvt.ITEM_Y_OFFSET, 2, 2)
        self_coords = utils.generateChampCoordinates(res_cvt.SELF_INDICATOR_LEFT_X_OFFSET,
                                                     res_cvt.SELF_INDICATOR_RIGHT_X_OFFSET,
                                                     res_cvt.SELF_INDICATOR_Y_DIFF,
                                                     res_cvt.SELF_INDICATOR_Y_OFFSET)

        champ_coords = np.reshape(champ_coords, (-1, 2))
        item_coords = np.reshape(item_coords, (-1, 2))
        self_coords = np.reshape(self_coords, (-1, 2))
        # item_coords = [(coord[0] + 2, coord[1] + 2) for coord in item_coords]

        champs_x = []
        items_x = []
        self_x = []
        for img in imgs:
            items_x_raw = [img[coord[1]:coord[1] + res_cvt.ITEM_SIZE, coord[0]:coord[0] + res_cvt.ITEM_SIZE] for coord
                           in item_coords]
            items_x += [cv.resize(img, ui_constants.ITEM_IMG_SIZE, cv.INTER_AREA) for img in items_x_raw]
            champs_x_raw = [img[coord[1]:coord[1] + res_cvt.CHAMP_SIZE, coord[0]:coord[0] + res_cvt.CHAMP_SIZE] for
                            coord in champ_coords]
            champs_x += [cv.resize(img, ui_constants.CHAMP_IMG_SIZE, cv.INTER_AREA) for img in champs_x_raw]
            self_x_raw = [
                img[coord[1]:coord[1] + res_cvt.SELF_INDICATOR_SIZE, coord[0]:coord[0] + res_cvt.SELF_INDICATOR_SIZE]
                for coord in self_coords]
            self_x += [cv.resize(img, ui_constants.SELF_IMG_SIZE, cv.INTER_AREA) for img in self_x_raw]


        # for i, item in enumerate(items_x):
        #     cv.imshow(str(i), item)
        # cv.waitKey(0)


        # self.showCoords(cv.imread('test_data/easy/1.png'), champ_coords, ui_constants.CHAMP_IMG_SIZE[0], item_coords, ui_constants.ITEM_IMG_SIZE[0], self_coords, ui_constants.SELF_IMG_SIZE[0])

        self_y = to_categorical(self_y, nb_classes=10)
        self_y = np.reshape(self_y, (-1, 1))
        return champs_x, items_x, self_x, champs_y, items_y, self_y

    def preprocess(self, img, med_it, kernel):
        for _ in range(med_it):
            img = cv.medianBlur(img, kernel)
        return img

    def start_generating_train_data(self, num, generator):

        class TrainingDataWorker(Process):
            def __init__(self, q, gen):
                super().__init__()
                self.q = q
                self.gen = gen
                self.stopped = multiprocessing.Event()

            def stop(self):

                self.stopped.set()


            def run(self):
                while not self.stopped.is_set():
                    self.q.put(self.gen())


        for _ in range(num):
            p = TrainingDataWorker(self.queue, generator)
            self.workers.append(p)
            p.start()

    def stop_generating_train_data(self):
        for p in self.workers:
            p.stop()

        # flush the queue to prevent deadlock
        while True:
            running = any(p.is_alive() for p in self.workers)
            while not self.queue.empty():
                self.queue.get()
                self.queue.task_done()
            if not running:
                break

        for p in self.workers:
            p.join()


    def determine_best_eval(self, main_evals, pre_evals):
        max_main = -1
        max_pre = -1
        best_model_index = -1
        for i, (main, pre) in enumerate(zip(main_evals, pre_evals)):
            if main > max_main:
                max_main = main
                max_pre = sum(pre)
                best_model_index = i
            elif main == max_main:
                sum_pre = sum(pre)
                if sum_pre > max_pre:
                    max_pre = sum_pre
                    best_model_index = i

        # epoch counter is 1 based
        return best_model_index + 1

    def save_best_model(self, best_model_index, train_path, best_path):
        #self.remove_old_files(best_path)
        best_model_files = glob.glob(train_path + "my_model" + str(best_model_index) + "*")
        best_model_files.append(train_path + "/accuracies")
        for file in best_model_files:
            shutil.copy2(file, best_path)


    def build_new_champ_model(self):
        train_path = self.out_paths["train"]["champ_imgs"]
        best_path = self.out_paths["best"]["champ_imgs"]
        champ_imgs = AssetManager().get_champ_imgs()
        training_data_generator = lambda: generate.generate_training_data(champ_imgs, 100, ui_constants.CHAMP_IMG_SIZE)
        network = ChampImgNetwork().build()
        X_test, _, _, Y_test, _, _ = self.load_img_test_data()
        self.build_new_img_model(network, training_data_generator, X_test, Y_test, train_path, best_path)

    def build_new_item_model(self):
        train_path = self.out_paths["train"]["item_imgs"]
        best_path = self.out_paths["best"]["item_imgs"]
        item_imgs = AssetManager().get_item_imgs()
        training_data_generator = lambda: generate.generate_training_data(item_imgs, 100, ui_constants.ITEM_IMG_SIZE)
        network = ItemImgNetwork().build()
        _, X_test, _, _, Y_test, _ = self.load_img_test_data()
        self.build_new_img_model(network, training_data_generator, X_test, Y_test, train_path, best_path)


    def build_new_img_model(self, network, training_data_generator, X_test, Y_test,train_path, best_path):
        self.remove_old_files(train_path)

        with open(train_path + "accuracies", "w") as logfile:
            monitorCallback = MonitorCallback(logfile)
            main_evals, pre_evals = self._train_img_network(network, training_data_generator, X_test, Y_test, train_path, logfile, monitorCallback)

        best_model_index = self.determine_best_eval(main_evals, pre_evals)
        self.save_best_model(best_model_index, train_path, best_path)



    def train_self_network(self):
        self.remove_old_files(self.self_img_out_path)
        self_imgs = AssetManager().get_self_imgs()
        training_data_generator = lambda: generate.generate_training_data(self_imgs, 1024, ui_constants.SELF_IMG_SIZE)
        network = SelfImgNetwork().build()
        _, _, X_test, _, _, Y_test = self.load_img_test_data()
        self._train_img_network(network, training_data_generator, X_test, Y_test, self.self_img_out_path)


    def _train_img_network(self, net, training_data_generator, X_test, Y_test, out_dir, logfile, monitorCallback):
        model = tflearn.DNN(net, tensorboard_verbose=3)
        X_preprocessed_test = [[self.preprocess(x, 1, 3) for x in X_test], \
                               [self.preprocess(x, 1, 5) for x in X_test], \
                               [self.preprocess(x, 2, 3) for x in X_test], \
                               [self.preprocess(x, 2, 5) for x in X_test]]
        main_test_evals = []
        preprocessed_test_evals = []
        self.start_generating_train_data(3, training_data_generator)
        for epoch in range(self.num_epochs):
            X, Y = self.queue.get()
            self.queue.task_done()
            model.fit(X, Y, n_epoch=1, shuffle=True, validation_set=None,
                      show_metric=True, batch_size=self.batch_size, run_id='whaddup_glib_globs' + str(epoch),
                      callbacks=monitorCallback)
            model.save(out_dir + 'my_model' + str(epoch + 1))


            y = model.predict(X_test)
            y = [np.argmax(y_) for y_ in y]
            y_test = [np.argmax(y_) for y_ in Y_test]
            print("Pred Actual")
            for i in range(len(y)):
                print(f"{self.cvt.img_int2item_str(y[i])} {self.cvt.img_int2item_str(Y_test[i])}")
            print("Raw test data predictions: {0}".format(y))
            print("Actual test data  values : {0}".format(Y_test.tolist()))

            main_eval = model.evaluate(np.array(X_test), np.array(Y_test), batch_size=self.batch_size)[0]

            pre_eval = [model.evaluate(np.array(X), np.array(Y_test), batch_size=self.batch_size) for X
                                       in X_preprocessed_test]
            pre_eval = np.reshape(pre_eval, -1)
            main_test_evals.append(main_eval)
            preprocessed_test_evals.append(pre_eval)
            self.log_output(logfile, main_eval, pre_eval, epoch)
        self.stop_generating_train_data()
        return main_test_evals, preprocessed_test_evals

    def log_output(self, logfile, main_test_eval, preprocessed_test_evals, epoch_counter):


        print("Epoch {0}:\nRaw test accuracy {1:.2f} | ".format(epoch_counter + 1, main_test_eval), end = '')
        logfile.write(
            "Raw test accuracy {1:.2f} | ".format(epoch_counter + 1, main_test_eval))
        print("preprocessed images accuracy", end = '')
        logfile.write(
            "preprocessed images accuracy")
        for eval in preprocessed_test_evals:
            logfile.write(" {0:.2f} ".format(eval))
            print(" {0:.2f} ".format(eval), end='')
        logfile.write("\n\n")
        logfile.flush()

    def train_positions_network():
        # model.load('./models/my_model1')
        print("Loading training data")
        dataloader = data_loader.PositionsDataLoader()
        print("Encoding training data")
        X, Y = dataloader.get_train_data()

        print("Encoding test data")
        X_test, Y_test = dataloader.get_test_data()

        # for i in range(1,44):
        #     with tf.Graph().as_default():
        #         net = network.classify_positions(network.positions_game_config, network.positions_network_config)
        #         model = tflearn.DNN(net, tensorboard_verbose=0)
        #         model.load('./position_models/models_improved/positions/my_model'+str(i))
        #         pred1 = model.evaluate(X_test, Y_test, batch_size=batch_size)
        #         print(i)
        #         print("eval is {0:.4f}".format(pred1[0]))

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            tflearn.is_training(True, session=sess)
            # with tf.device('/device:GPU:0'):
            print("Building model")
            net = network.classify_positions(network.positions_game_config, network.positions_network_config)
            model = tflearn.DNN(net, tensorboard_verbose=0, session=sess)
            sess.run(tf.global_variables_initializer())
            print("Commencing training")
            with open("models/positions/accuracies", "w") as f:
                class MonitorCallback(tflearn.callbacks.Callback):

                    def on_epoch_end(self, training_state):
                        f.write("Epoch {0} train accuracy {1:.4f} | loss {2:.4f}\n".format(training_state.epoch,
                                                                                           training_state.acc_value,
                                                                                           training_state.global_loss))
                        f.flush()

                monitorCallback = MonitorCallback()
                for epoch in range(num_epochs):
                    model.fit(X, Y, n_epoch=1, shuffle=True, validation_set=None,
                              show_metric=True, batch_size=batch_size, run_id='whaddup_glib_globs' + str(epoch),
                              callbacks=monitorCallback)
                    pred1 = model.evaluate(X_test, Y_test, batch_size=batch_size)
                    print("eval is {0:.4f}".format(pred1[0]))

                    model.save('models/positions/my_model' + str(epoch + 1))
                    f.write("Epoch {0} eval accuracy {1:.4f}\n".format(epoch + 1, pred1[0]))
                    f.flush()


    def train_next_data_network():
        # model.load('./models/my_model1')
        print("Loading training data")
        dataloader = data_loader.NextItemsDataLoader()
        print("Encoding test data")
        X_test, Y_test = dataloader.get_test_data()
        X, Y = dataloader.get_train_data()
        # X, Y = np.random.randint(100,size=1000*71).reshape(-1,71), np.arange(1000)
        # X = np.array(X)
        # X[:,0] = 0
        # X_test, Y_test = X,Y
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            tflearn.is_training(True, session=sess)
            # with tf.device('/device:GPU:0'):
            print("Building model")
            net = network.classify_next_item(network.game_config, network.next_network_config)
            model = tflearn.DNN(net, tensorboard_verbose=0, session=sess)
            sess.run(tf.global_variables_initializer())
            print("Commencing training")
            with open("models/accuracies", "w") as f:
                class MonitorCallback(tflearn.callbacks.Callback):

                    def on_epoch_end(self, training_state):
                        f.write("Epoch {0} train accuracy {1:.4f} | loss {2:.4f}\n".format(training_state.epoch,
                                                                                           training_state.acc_value,
                                                                                           training_state.global_loss))
                        f.flush()

                monitorCallback = MonitorCallback()
                for epoch in range(num_epochs):
                    # while True:
                    #     X, Y = dataloader.get_next_train_subepoch(1)
                    #     X = np.concatenate(X, axis=0)
                    #     Y = np.concatenate(Y, axis=0)
                    #     if X == []:
                    #         break
                    model.fit(X, Y, shuffle=True, n_epoch=1, validation_set=None,
                              show_metric=True, batch_size=batch_size, run_id='whaddup_glib_globs' + str(epoch),
                              callbacks=monitorCallback)
                    pred1 = model.evaluate(X_test, Y_test, batch_size=batch_size)
                    print("eval is {0:.4f}".format(pred1[0]))
                    model.save('models/my_model' + str(epoch + 1))
                    f.write("Epoch {0} eval accuracy {1:.4f}\n".format(epoch + 1, pred1[0]))
                    f.flush()


if __name__ == '__main__':
    t = Trainer()
    t.build_new_item_model()
