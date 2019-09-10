import multiprocessing
import json
import math
import multiprocessing
import shutil
import sys
from multiprocessing import Process, JoinableQueue

from sklearn.metrics import auc, \
    classification_report, precision_recall_curve, precision_recall_fscore_support
from tflearn.data_utils import to_categorical

from train_model import generate, data_loader
from train_model.model import *
from train_model.network import *
from collections import Counter


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
        self.num_epochs = 10
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
        self.class_weights = 1


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
        with tf.device("/gpu:0"):
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                tflearn.is_training(True)
                self.network.network_config["class_weights"] = self.class_weights
                self.network = self.network.build()
                model = tflearn.DNN(self.network, session=sess)
                sess.run(tf.global_variables_initializer())
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
                    #     else:
                    #         print(f"{i}: {a} {b}")
                    # print("Raw test data predictions: {0}".format(y))
                    # print("Actual test data  values : {0}".format(self.Y_test))

                    # y = model.predict(self.X_test)
                    # y = [np.argmax(y_) for y_ in np.reshape(y, (4, 10))]
                    # y = to_categorical(y, 10).flatten()
                    # y_test = [np.argmax(y_) for y_ in np.reshape(self.Y_test, (4, 10))]
                    # y_test = to_categorical(y_test, 10).flatten()
                    # print("Pred Actual")
                    # for i in range(len(y)):
                    #     a = self.self_manager.lookup_by('img_int', y[i])['name']
                    #     b = self.self_manager.lookup_by('img_int', self.Y_test[i][0])['name']
                    #     if a != b:
                    #         print(f"----->{i}: {a} {b}")
                    #     else:
                    #         print(f"{i}: {a} {b}")
                    # print("Raw test data predictions: {0}".format(y))
                    # print("Actual test data  values : {0}".format(self.Y_test))

                    score = self.eval_model(model, epoch)
                    scores.append(score)
        return scores


    def get_train_data(self):
        return self.X, self.Y


    def log_output(self, main_test_eval, epoch_counter):
        print("Epoch {0}:\nRaw test accuracy {1:.4f} | ".format(epoch_counter + 1, main_test_eval), end='')
        self.logfile.write(
            "Raw test accuracy {1:.4f} | ".format(epoch_counter + 1, main_test_eval))


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
        image_paths = ['test_data/easy/1.png', 'test_data/easy/2.png', 'test_data/easy/3.png', 'test_data/easy/4.png']
        image_paths.sort()
        imgs = [cv.imread(path) for path in image_paths]
        with open('test_data/easy/test_labels.json', "r") as f:
            elems = json.load(f)

        champs_x, items_x, self_x, champs_y, items_y, self_y = [], [], [], [], [], []
        for test_image_x, test_image_y in zip(imgs, elems.items()):
            test_image_y = test_image_y[1]
            champs_y.extend([self.champ_manager.lookup_by("name", champ_name)["img_int"] for champ_name in
                             test_image_y["champs"]])
            items_y.extend(
                [self.item_manager.lookup_by("name", item_name)["img_int"] for item_name in test_image_y["items"]])
            self_y.extend(to_categorical([test_image_y["self"]], nb_classes=10)[0])

            res_cvt = ui_constants.ResConverter(*(test_image_y["res"].split(",")))
            res_cvt.set_res(*(test_image_y["res"].split(",")))
            champ_coords = ChampImgModel.generate_champ_coords(res_cvt.CHAMP_LEFT_X_OFFSET,
                                                               res_cvt.CHAMP_RIGHT_X_OFFSET,
                                                               res_cvt.CHAMP_Y_DIFF,
                                                               res_cvt.CHAMP_Y_OFFSET)

            item_x_offset = res_cvt.ITEM_INNER_OFFSET
            item_y_offset = res_cvt.ITEM_INNER_OFFSET
            if test_image_y["summ_names_displayed"]:
                item_x_offset += res_cvt.SUMM_NAMES_DIS_X_OFFSET
                item_y_offset += res_cvt.SUMM_NAMES_DIS_Y_OFFSET
            item_coords = ItemImgModel.generate_item_coords(res_cvt.ITEM_X_DIFF,
                                                            res_cvt.ITEM_LEFT_X_OFFSET,
                                                            res_cvt.ITEM_RIGHT_X_OFFSET,
                                                            res_cvt.ITEM_Y_DIFF,
                                                            res_cvt.ITEM_Y_OFFSET, item_x_offset, item_y_offset)

            self_coords = ChampImgModel.generate_champ_coords(res_cvt.SELF_INDICATOR_LEFT_X_OFFSET,
                                                              res_cvt.SELF_INDICATOR_RIGHT_X_OFFSET,
                                                              res_cvt.SELF_INDICATOR_Y_DIFF,
                                                              res_cvt.SELF_INDICATOR_Y_OFFSET)

            champ_coords = np.reshape(champ_coords, (-1, 2))
            item_coords = np.reshape(item_coords, (-1, 2))
            self_coords = np.reshape(self_coords, (-1, 2))
            # item_coords = [(coord[0] + 2, coord[1] + 2) for coord in item_coords]

            items_x_raw = [test_image_x[coord[1]:coord[1] + res_cvt.ITEM_SIZE, coord[0]:coord[0] + res_cvt.ITEM_SIZE]
                           for coord
                           in item_coords]
            items_x.extend([cv.resize(img, ui_constants.NETWORK_ITEM_IMG_CROP, cv.INTER_AREA) for img in items_x_raw])
            champs_x_raw = [test_image_x[coord[1]:coord[1] + res_cvt.CHAMP_SIZE, coord[0]:coord[0] + res_cvt.CHAMP_SIZE]
                            for
                            coord in champ_coords]
            champs_x.extend([cv.resize(img, ui_constants.NETWORK_CHAMP_IMG_CROP, cv.INTER_AREA) for img in
                             champs_x_raw])
            self_x_raw = [
                test_image_x[coord[1]:coord[1] + res_cvt.SELF_INDICATOR_SIZE,
                coord[0]:coord[0] + res_cvt.SELF_INDICATOR_SIZE]
                for coord in self_coords]
            self_x.extend([cv.resize(img, ui_constants.NETWORK_SELF_IMG_CROP, cv.INTER_AREA) for img in self_x_raw])

            # self.show_coords(cv.imread('test_data/easy/1.png'), champ_coords, ui_constants.CHAMP_IMG_SIZE[0], item_coords, ui_constants.ITEM_IMG_SIZE[0], self_coords, ui_constants.SELF_IMG_SIZE[0])

        # for i, item in enumerate(items_x):
        #     cv.imshow(str(i), item)
        # cv.waitKey(0)
        self_y = np.array(self_y)[:, np.newaxis]
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
            self.logfile.write(" {0:.4f} ".format(eval_))
            print(" {0:.4f} ".format(eval_), end='')

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
        self.network = ChampImgNetwork()
        self.X_test, _, _, self.Y_test, _, _ = self.load_img_test_data()
        self._build_new_img_model()


    def build_new_item_model(self):
        self.train_path = app_constants.model_paths["train"]["item_imgs"]
        self.best_path = app_constants.model_paths["best"]["item_imgs"]
        self.training_data_generator = lambda: generate.generate_training_data(self.item_manager.get_imgs(), 100,
                                                                               ui_constants.NETWORK_ITEM_IMG_CROP)
        self.network = ItemImgNetwork()
        _, self.X_test, _, _, self.Y_test, _ = self.load_img_test_data()
        self._build_new_img_model()


    def build_new_self_model(self):
        self.train_path = app_constants.model_paths["train"]["self_imgs"]
        self.best_path = app_constants.model_paths["best"]["self_imgs"]
        self.training_data_generator = lambda: generate.generate_training_data(self.self_manager.get_imgs(), 1024,
                                                                               ui_constants.NETWORK_SELF_IMG_CROP, True)
        self.network = SelfImgNetwork()
        _, _, self.X_test, _, _, self.Y_test = self.load_img_test_data()
        self._build_new_img_model()


    def _build_new_img_model(self):
        self.start_generating_train_data(3)
        self.build_new_model()
        self.stop_generating_train_data()


class StaticTrainingDataTrainer(Trainer):



    def determine_best_eval(self, scores):
        # epoch counter is 1 based
        return np.argmax(scores[:,0]) + 1


    def eval_model(self, model, epoch, prior=None):

        y_pred_prob = []
        for chunk in utils.chunks(self.X_test, 1024):
            y_pred_prob.extend(model.predict(np.array(chunk)))
        y_pred_prob = np.array(y_pred_prob)
        if prior:
            y_pred_prob = y_pred_prob / prior


        y_pred = np.argmax(y_pred_prob, axis=1)

        acc = sum(np.equal(y_pred, self.Y_test)) / len(self.Y_test)
        precision, recall, f1, support = precision_recall_fscore_support(self.Y_test, y_pred, average='macro')

        report = classification_report(self.Y_test, y_pred, labels=range(len(self.target_names)),
                                       target_names=self.target_names)
        # confusion = confusion_matrix(self.Y_test, y_pred)
        avg_binary_auc, avg_binary_f1, thresholds = self.get_cum_scores(self.Y_test, y_pred_prob)

        self.log_output(acc, f1, precision, recall, avg_binary_f1, avg_binary_auc,
                        report, thresholds, epoch)

        return avg_binary_f1, avg_binary_auc, acc, precision, recall, f1, thresholds

    def standalone_eval(self):

        with open(app_constants.model_paths["best"]["next_items_early"] + "my_model1_thresholds.json") as f:

            thresholds = json.load(f)
        self.X_test, self.Y_test = data_loader.NextItemsDataLoader(app_constants.train_paths[
                                                                       "next_items_early_processed"]).get_test_data()
        self.network = NextItemEarlyGameNetwork().build()
        self.test_y_distrib = Counter(self.Y_test)
        print("Loading test data")


        self.target_names = [target["name"] for target in sorted(list(ItemManager().get_ints().values()), key=lambda
            x: x["int"])]


        model = tflearn.DNN(self.network)
        model_path = glob.glob(app_constants.model_paths["best"]["next_items_early"] + "my_model*")[0]
        model_path = model_path.rpartition('.')[0]
        model.load(model_path)


        with open("lololo", "w") as self.logfile:
            # thresholds = self.eval_model(model, 0)[-1]
            self.eval_model(model, 0, prior=thresholds)[-1]


    def get_cum_scores(self, Y_true, Y_pred_prob):
        num = Y_pred_prob.shape[1]
        scores = np.array([self.calc_metrics_for_one_class((Y_true == i).astype(int), Y_pred_prob[:,
                                                                                      i], i) for i in range(num)])
        zero_counts = Counter(range(num)) - self.test_y_distrib
        nonzero_counts = Counter(range(num)) - zero_counts
        zero_count_indices = list(zero_counts.keys())
        nonzero_count_indices = list(nonzero_counts.keys())
        num_nonzeros = num - len(zero_count_indices)
        avg_auc = sum(scores[:, 0]) / num_nonzeros
        avg_f1 = sum(scores[:, 1]) / num_nonzeros
        thresholds = scores[:, 2]
        thresholds[zero_count_indices] = min(thresholds[nonzero_count_indices])
        #make sure empty item never appears
        thresholds[0] = 1000
        return avg_auc, avg_f1, thresholds


    def calc_metrics_for_one_class(self, Y_true, Y_pred_prob, j):
        precision, recall, thresholds = precision_recall_curve(Y_true, Y_pred_prob)
        precision = np.nan_to_num(precision)
        recall = np.nan_to_num(recall)
        auc_score = auc(recall, precision)
        auc_score = np.nan_to_num(auc_score)
        precision = precision[:-1]
        recall = recall[:-1]
        f1_scores = 2*precision*recall
        non_zero_f1_scores = f1_scores != 0
        f1_scores[non_zero_f1_scores] /= precision[non_zero_f1_scores] + recall[non_zero_f1_scores]
        max_f1_index = np.argmax(f1_scores)
        return auc_score, f1_scores[max_f1_index], thresholds[max_f1_index]


    def log_output(self, main_test_eval, f1, precision, recall, avg_binary_f1, avg_binary_auc, classification,
                   thresholds,
                   epoch_counter):

        for output in [sys.stdout, self.logfile]:
            output.write("Epoch {0}\n".format(epoch_counter + 1))
            output.write("1. Acc {0:.4f}\n".format(main_test_eval))
            output.write('2. F-1 {0:.4f}\n'.format(f1))
            output.write('3. Precision {0:.4f}\n'.format(precision))
            output.write('4. Recall {0:.4f}\n'.format(recall))
            output.write('5. Avg binary F1 {0:.4f}\n'.format(avg_binary_f1))
            output.write('6. Avg binary auc {0:.4f}\n'.format(avg_binary_auc))
            output.write('7. Classification report \n {} \n'.format(classification))
            output.write("\n\n")
            output.flush()

        with open(self.train_path + self.model_name + str(epoch_counter + 1) + "_thresholds.json", "w") as f:
            f.write(json.dumps(thresholds.tolist()))


    def determine_best_eval(self, scores):
        max_main = -1
        best_model_index = -1
        for i, main in enumerate(scores):
            if main >= max_main:
                max_main = main
                best_model_index = i
        # epoch counter is 1 based
        return best_model_index + 1


    def build_next_items_early_game_model(self):
        self.target_names = [target["name"] for target in sorted(list(ItemManager().get_ints().values()), key=lambda
            x: x["int"])]
        self.network = NextItemEarlyGameNetwork()
        self.train_path = app_constants.model_paths["train"]["next_items_early"]
        self.best_path = app_constants.model_paths["best"]["next_items_early"]

        print("Loading training data")
        dataloader = data_loader.NextItemsDataLoader(app_constants.train_paths["next_items_early_processed"])
        # self.X, self.Y = dataloader.get_train_data()
        self.X, self.Y = dataloader.get_test_data()
        print("Loading test data")
        self.X_test, self.Y_test = self.X, self.Y
        # self.train_y_distrib = Counter(self.Y_test)
        self.test_y_distrib = Counter(self.Y)
        total_y_distrib = self.test_y_distrib
        # total_y_distrib = self.train_y_distrib + self.test_y_distrib
        missing_items = Counter(list(range(len(self.target_names)))) - total_y_distrib
        # assert(missing_items == Counter([0]))
        total_y = sum(list(total_y_distrib.values()))
        total_y_distrib_sorted = np.array([count for count in np.array(sorted(list((total_y_distrib +
                                                                               missing_items).items()),
                                                                      key=lambda x: x[0]))[:,1]])

        self.class_weights = total_y / total_y_distrib_sorted
        #don't include weights for empty item
        self.class_weights[0] = 0
        self.build_new_model()


    def build_next_items_late_game_model(self):
        self.train_path = app_constants.model_paths["train"]["next_items_late"]
        self.best_path = app_constants.model_paths["best"]["next_items_late"]
        self.network = NextItemLateGameNetwork()
        print("Loading training data")
        dataloader = data_loader.NextItemsDataLoader(app_constants.train_paths["next_items_late_processed"])
        self.X, self.Y = dataloader.get_train_data()
        print("Loading test data")
        self.X_test, self.Y_test = dataloader.get_test_data()
        self.build_new_model()


    def build_positions_model(self):
        self.train_path = app_constants.model_paths["train"]["positions"]
        self.best_path = app_constants.model_paths["best"]["positions"]
        self.network = PositionsNetwork()
        print("Loading training data")
        dataloader = data_loader.PositionsDataLoader()
        self.X, self.Y = dataloader.get_train_data()
        print("Loading test data")
        self.X_test, self.Y_test = dataloader.get_test_data()
        self.build_new_model()


if __name__ == "__main__":
    t = StaticTrainingDataTrainer()
    #t.build_next_items_early_game_model()
    t.standalone_eval()
    # s = DynamicTrainingDataTrainer()
    # s.build_new_self_model()

    # res_cvt = ui_constants.ResConverter(1920, 1080)
    # model = ItemImgModel(res_cvt, True)
    #
    #
    # _, X_test, _, _, Y_test, _ = s.load_img_test_data()
    #
    # y = model.model.predict(X_test)
    # y = [np.argmax(y_) for y_ in y]
    # y_test = [np.argmax(y_) for y_ in Y_test]
    # print("Pred Actual")
    # for i in range(len(y)):
    #     a = model.artifact_manager.lookup_by('img_int', y[i])['name']
    #     b = model.artifact_manager.lookup_by('img_int', Y_test[i])['name']
    #     if a != b:
    #         print(f"----->{i}: {a} {b}")
    #     else:
    #         print(f"{i}: {a} {b}")
    # print("Raw test data predictions: {0}".format(y))
    # print("Actual test data  values : {0}".format(Y_test))

    # s = DynamicTrainingDataTrainer()
    # s.build_new_item_model()
