import glob
import multiprocessing
import shutil
from multiprocessing import Process, JoinableQueue

from tflearn.data_utils import to_categorical

from train_model import generate, data_loader
from utils import utils
from train_model.model import *
from train_model.network import *
import json


class MonitorCallback(tflearn.callbacks.Callback):

    def __init__(self, f):
        super().__init__()
        self.f = f


    def on_epoch_end(self, training_state):
        self.f.write(
            "Epoch {0} accuracy {1:.2f} | loss {2:.2f}\n".format(training_state.epoch, training_state.acc_value,
                                                                 training_state.global_loss))
        self.f.flush()


class Trainer(ABC):

    def __init__(self):
        self.num_epochs = 30
        self.batch_size = 128
        self.model_name = "my_model"
        self.acc_file_name = "accuracies"
        self.train_path = None
        self.best_path = None
        self.X = None
        self.Y = None
        self.X_test = None
        self.Y_test = None
        self.logfile = None
        self.monitor_callback = None
        self.network = None


    @abstractmethod
    def determine_best_eval(self, scores):
        pass


    @abstractmethod
    def eval_model(self, model, epoch):
        pass


    def save_best_model(self, best_model_index):
        utils.remove_old_files(self.best_path)
        best_model_files = glob.glob(self.train_path + self.model_name + str(best_model_index) + ".*")
        best_model_files.append(self.train_path + self.acc_file_name)
        for file in best_model_files:
            shutil.copy2(file, self.best_path)


    def build_new_model(self):
        utils.remove_old_files(self.train_path)
        with open(self.train_path + self.acc_file_name, "w") as self.logfile:
            self.monitor_callback = MonitorCallback(self.logfile)
            scores = self.train_neural_network()
        best_model_index = self.determine_best_eval(scores)
        self.save_best_model(best_model_index)


    def train_neural_network(self):
        model = tflearn.DNN(self.network)
        scores = []
        for epoch in range(self.num_epochs):
            x, y = self.get_train_data()
            model.fit(x, y, n_epoch=1, shuffle=True, validation_set=None,
                      show_metric=True, batch_size=self.batch_size, run_id='whaddup_glib_globs' + str(epoch),
                      callbacks=self.monitor_callback)
            model.save(self.train_path + self.model_name + str(epoch + 1))


            # y = model.predict(self.X_test)
            # y = [np.argmax(y_) for y_ in y]
            # y_test = [np.argmax(y_) for y_ in self.Y_test]
            # print("Pred Actual")
            # for i in range(len(y)):
            #     a = self.item_manager.lookup_by('img_int', y[i])['name']
            #     b = self.item_manager.lookup_by('img_int', self.Y_test[i])['name']
            #     if a != b:
            #         print(f"----->{i}: {a} {b}")
            #     print(f"{i}: {a} {b}")
            # print("Raw test data predictions: {0}".format(y))
            # print("Actual test data  values : {0}".format(self.Y_test.tolist()))

            score = self.eval_model(model, epoch)
            scores.append(score)
        return scores


    def get_train_data(self):
        return self.X, self.Y


    def log_output(self, main_test_eval, epoch_counter):
        print("Epoch {0}:\nRaw test accuracy {1:.2f} | ".format(epoch_counter + 1, main_test_eval), end='')
        self.logfile.write(
            "Raw test accuracy {1:.2f} | ".format(epoch_counter + 1, main_test_eval))


class DynamicTrainingDataTrainer(Trainer):

    def __init__(self):
        super().__init__()
        self.num_epochs = 50
        self.queue = JoinableQueue(10)
        self.workers = []
        self.X_preprocessed_test = None
        self.training_data_generator = None
        self.champ_manager = ChampManager()
        self.item_manager = ItemManager()
        self.self_manager = SelfManager()


    def get_train_data(self):
        x, y = self.queue.get()
        self.queue.task_done()
        return x, y


    def load_img_test_data(self):
        image_paths = ['test_data/easy/1.png', 'test_data/easy/2.png', 'test_data/easy/3.png']
        image_paths.sort()
        imgs = [cv.imread(path) for path in image_paths]
        with open('test_data/easy/test_labels.json', "r") as f:
            elems = json.load(f)

        champs_y = [list(v['champs']) for k, v in elems.items()]
        items_y = [list(v['items']) for k, v in elems.items()]
        self_y = [v['self'] for k, v in elems.items()]

        champs_y = [[self.champ_manager.lookup_by("name", champ_name)["img_int"] for champ_name in champ_list] for \
                champ_list
                    in champs_y]
        items_y = [[self.item_manager.lookup_by("name", item_name)["img_int"] for item_name in items_list] for \
                items_list in
                   items_y]
        champs_y = np.ravel(champs_y)
        items_y = np.ravel(items_y)

        res_cvt = ui_constants.ResConverter(1440, 900)
        champ_coords = ChampImgModel.generate_champ_coords(res_cvt.CHAMP_LEFT_X_OFFSET, res_cvt.CHAMP_RIGHT_X_OFFSET,
                                                           res_cvt.CHAMP_Y_DIFF,
                                                           res_cvt.CHAMP_Y_OFFSET)
        item_coords = ItemImgModel.generate_item_coords(res_cvt.ITEM_X_DIFF, res_cvt.ITEM_LEFT_X_OFFSET,
                                                        res_cvt.ITEM_RIGHT_X_OFFSET,
                                                        res_cvt.ITEM_Y_DIFF,
                                                        res_cvt.ITEM_Y_OFFSET, res_cvt.ITEM_INNER_OFFSET,
                                                        res_cvt.ITEM_INNER_OFFSET)
        self_coords = ChampImgModel.generate_champ_coords(res_cvt.SELF_INDICATOR_LEFT_X_OFFSET,
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
            items_x += [cv.resize(img, ui_constants.NETWORK_ITEM_IMG_CROP, cv.INTER_AREA) for img in items_x_raw]
            champs_x_raw = [img[coord[1]:coord[1] + res_cvt.CHAMP_SIZE, coord[0]:coord[0] + res_cvt.CHAMP_SIZE] for
                            coord in champ_coords]
            champs_x += [cv.resize(img, ui_constants.NETWORK_CHAMP_IMG_CROP, cv.INTER_AREA) for img in champs_x_raw]
            self_x_raw = [
                img[coord[1]:coord[1] + res_cvt.SELF_INDICATOR_SIZE, coord[0]:coord[0] + res_cvt.SELF_INDICATOR_SIZE]
                for coord in self_coords]
            self_x += [cv.resize(img, ui_constants.NETWORK_SELF_IMG_CROP, cv.INTER_AREA) for img in self_x_raw]

        # for i, item in enumerate(items_x):
        #     cv.imshow(str(i), item)
        # cv.waitKey(0)

        # self.show_coords(cv.imread('test_data/easy/1.png'), champ_coords, ui_constants.CHAMP_IMG_SIZE[0], item_coords, ui_constants.ITEM_IMG_SIZE[0], self_coords, ui_constants.SELF_IMG_SIZE[0])

        self_y = to_categorical(self_y, nb_classes=10)
        self_y = np.reshape(self_y, (-1, 1))
        return champs_x, items_x, self_x, champs_y, items_y, self_y


    def eval_model(self, model, epoch):
        if not self.X_preprocessed_test:
            self.X_preprocessed_test = [[utils.preprocess(x, 1, 3) for x in self.X_test],
                                        [utils.preprocess(x, 1, 5) for x in self.X_test],
                                        [utils.preprocess(x, 2, 3) for x in self.X_test],
                                        [utils.preprocess(x, 2, 5) for x in self.X_test]]
        main_eval = model.evaluate(np.array(self.X_test), np.array(self.Y_test), batch_size=self.batch_size)[0]
        extra_eval = [model.evaluate(np.array(X), np.array(self.Y_test), batch_size=self.batch_size) for X
                      in self.X_preprocessed_test]
        extra_eval = np.reshape(extra_eval, -1)
        self.log_output(main_eval, extra_eval, epoch)

        return main_eval, extra_eval


    def determine_best_eval(self, scores):

        max_main = -1
        max_pre = -1
        best_model_index = -1
        for i, (main, pre) in enumerate(scores):
            if main > max_main:
                max_main = main
                max_pre = sum(pre)
                best_model_index = i
            elif main == max_main:
                sum_pre = sum(pre)
                if sum_pre >= max_pre:
                    max_pre = sum_pre
                    best_model_index = i

        # epoch counter is 1 based
        return best_model_index + 1


    def log_output(self, main_test_eval, preprocessed_test_evals, epoch_counter):
        super().log_output(main_test_eval, epoch_counter)

        print("preprocessed images accuracy", end='')
        self.logfile.write(
            "preprocessed images accuracy")

        for eval_ in preprocessed_test_evals:
            self.logfile.write(" {0:.2f} ".format(eval_))
            print(" {0:.2f} ".format(eval_), end='')

        self.logfile.write("\n\n")
        self.logfile.flush()


    def start_generating_train_data(self, num):
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
            p = TrainingDataWorker(self.queue, self.training_data_generator)
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


    def build_new_champ_model(self):
        self.train_path = app_constants.model_paths["train"]["champ_imgs"]
        self.best_path = app_constants.model_paths["best"]["champ_imgs"]
        self.training_data_generator = lambda: generate.generate_training_data(self.champ_manager.get_imgs(), 100,
                                                                               ui_constants.NETWORK_CHAMP_IMG_CROP)
        self.network = ChampImgNetwork().build()
        self.X_test, _, _, self.Y_test, _, _ = self.load_img_test_data()
        self._build_new_img_model()


    def build_new_item_model(self):
        self.train_path = app_constants.model_paths["train"]["item_imgs"]
        self.best_path = app_constants.model_paths["best"]["item_imgs"]
        self.training_data_generator = lambda: generate.generate_training_data(self.item_manager.get_imgs(), 100,
                                                                               ui_constants.NETWORK_ITEM_IMG_CROP)
        self.network = ItemImgNetwork().build()
        _, self.X_test, _, _, self.Y_test, _ = self.load_img_test_data()
        self._build_new_img_model()


    def build_new_self_model(self):
        self.train_path = app_constants.model_paths["train"]["self_imgs"]
        self.best_path = app_constants.model_paths["best"]["self_imgs"]
        self.training_data_generator = lambda: generate.generate_training_data(self.self_manager.get_imgs(), 1024,
                                                                               ui_constants.NETWORK_SELF_IMG_CROP, True)
        self.network = SelfImgNetwork().build()
        _, _, self.X_test, _, _, self.Y_test = self.load_img_test_data()
        self._build_new_img_model()


    def _build_new_img_model(self):
        self.start_generating_train_data(3)
        self.build_new_model()
        self.stop_generating_train_data()


class StaticTrainingDataTrainer(Trainer):

    def eval_model(self, model, epoch):
        main_eval = model.evaluate(np.array(self.X_test), np.array(self.Y_test), batch_size=self.batch_size)[0]
        self.log_output(main_eval, epoch)
        return main_eval


    def log_output(self, main_test_eval, epoch_counter):
        super().log_output(main_test_eval, epoch_counter)
        self.logfile.write("\n\n")
        self.logfile.flush()


    def determine_best_eval(self, scores):
        max_main = -1
        best_model_index = -1
        for i, main in enumerate(scores):
            if main >= max_main:
                max_main = main
                best_model_index = i
        # epoch counter is 1 based
        return best_model_index + 1


    def build_next_items_model(self):
        self.train_path = app_constants.model_paths["train"]["next_items"]
        self.best_path = app_constants.model_paths["best"]["next_items"]
        self.network = NextItemNetwork().build()
        print("Loading training data")
        dataloader = data_loader.NextItemsDataLoader()
        self.X, self.Y = dataloader.get_train_data()
        print("Loading test data")
        self.X_test, self.Y_test = dataloader.get_test_data()
        self.build_new_model()


    def build_positions_model(self):
        self.train_path = app_constants.model_paths["train"]["positions"]
        self.best_path = app_constants.model_paths["best"]["positions"]
        self.network = PositionsNetwork().build()
        print("Loading training data")
        dataloader = data_loader.PositionsDataLoader()
        self.X, self.Y = dataloader.get_train_data()
        print("Loading test data")
        self.X_test, self.Y_test = dataloader.get_test_data()
        self.build_new_model()


if __name__ == "__main__":
    # t = StaticTrainingDataTrainer()
    # t.build_next_items_model()

    s = DynamicTrainingDataTrainer()
    s.build_new_champ_model()