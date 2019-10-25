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
from utils.artifact_manager import ChampManager, ItemManager, SelfManager, CurrentGoldManager, KDAManager
from utils import utils


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

                    # self.eval_model(model, epoch, prior=score[-1])
                    scores.append(score)
        return scores


    def get_train_data(self):
        return self.X, self.Y


    def log_output(self, main_test_eval, epoch_counter):
        print("Epoch {0}:\nRaw test accuracy {1:.4f} | ".format(epoch_counter + 1, main_test_eval), end='')
        self.logfile.write(
            "Raw test accuracy {1:.4f} | ".format(epoch_counter + 1, main_test_eval))


class KDATrainer(DynamicTrainingDataTrainer):
    def __init__(self):
        super().__init__()
        self.kda_manager = KDAManager()
        self.kda_model = KDAImgModel()


    def extract_y_data(self, data):
        kda_y = []
        for row in data:
            for i, section in enumerate(row):
                for digit in section:
                    kda_y.append(self.kda_manager.lookup_by("name", digit)["img_int"])
                if i != 2:
                    kda_y.append(self.kda_manager.lookup_by("name", "slash")["img_int"])


    def load_test_data(self, element):
        with open('test_data/easy/test_labels.json', "r") as f:
            elems = json.load(f)

        base_path = "test_data/easy/"
        result_x, result_y = [],[]

        for key in elems:
            test_image_y = elems[key]
            test_image_x = cv.imread(base_path + test_image_y["filename"])
            res_cvt = ui_constants.ResConverter(*(test_image_y["res"].split(",")))
            res_cvt.set_res(*(test_image_y["res"].split(",")))

            if test_image_y[element] != None:
                result_y.extend(self.extract_y_data(test_image_y[element]))
                result_x.extend(self.kda_model.extract_digit_imgs(test_image_x))

        return result_x, result_y


    def build_new_model(self):
        self.class_weights = 1
        self.train_path = app_constants.model_paths["train"]["kda_imgs"]
        self.best_path = app_constants.model_paths["best"]["kda_imgs"]
        self.training_data_generator = lambda: generate.generate_training_data_rect(self.kda_manager.get_imgs(),
                                                                                    500,
                                                                                    ui_constants.NETWORK_KDA_IMG_CROP)
        self.network = DigitRecognitionNetwork(lambda: KDAManager().get_num("img_int"),
                                               ui_constants.NETWORK_KDA_IMG_CROP)
        self.X_test, self.Y_test = self.load_img_test_data()["kda"]

        self._build_new_img_model()







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
        self.current_gold_manager = CurrentGoldManager()
        self.kda_manager = KDAManager()




    def get_train_data(self):
        x, y = self.queue.get()
        self.queue.task_done()
        return x, y


    def load_img_test_data(self):

        with open('test_data/easy/test_labels.json', "r") as f:
            elems = json.load(f)

        base_path = "test_data/easy/"
        result = dict()
        for artifact in ["champs", "items", "self", "current_gold", "kda"]:
            result[artifact] = ([],[])

        for key in elems:
            test_image_y = elems[key]
            test_image_x = cv.imread(base_path + test_image_y["filename"])
            res_cvt = ui_constants.ResConverter(*(test_image_y["res"].split(",")))
            res_cvt.set_res(*(test_image_y["res"].split(",")))

            if test_image_y["champs"] != None:
                result["champs"][1].extend([self.champ_manager.lookup_by("name", champ_name)["img_int"] for
                                            champ_name in
                                 test_image_y["champs"]])
                champ_coords = ChampImgModel.generate_champ_coords(res_cvt.CHAMP_LEFT_X_OFFSET,
                                                                   res_cvt.CHAMP_RIGHT_X_OFFSET,
                                                                   res_cvt.CHAMP_Y_DIFF,
                                                                   res_cvt.CHAMP_Y_OFFSET)
                champ_coords = np.reshape(champ_coords, (-1, 2))
                champs_x_raw = [test_image_x[coord[1]:coord[1] + res_cvt.CHAMP_SIZE, coord[0]:coord[0] + res_cvt.CHAMP_SIZE]
                                for
                                coord in champ_coords]
                result["champs"][0].extend([cv.resize(img, ui_constants.NETWORK_CHAMP_IMG_CROP, cv.INTER_AREA) for
                                            img in
                                 champs_x_raw])

            if test_image_y["items"] != None:
                result["items"][1].extend(
                    [self.item_manager.lookup_by("name", item_name)["img_int"] for item_name in test_image_y["items"]])
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
                item_coords = np.reshape(item_coords, (-1, 2))
                items_x_raw = [test_image_x[coord[1]:coord[1] + res_cvt.ITEM_SIZE, coord[0]:coord[0] + res_cvt.ITEM_SIZE]
                               for coord
                               in item_coords]
                result["items"][0].extend([cv.resize(img, ui_constants.NETWORK_ITEM_IMG_CROP, cv.INTER_AREA) for img
                                            in items_x_raw])

            if test_image_y["self"] != None:
                result["self"][1].extend(to_categorical([test_image_y["self"]], nb_classes=10)[0])
                self_coords = ChampImgModel.generate_champ_coords(res_cvt.SELF_INDICATOR_LEFT_X_OFFSET,
                                                                  res_cvt.SELF_INDICATOR_RIGHT_X_OFFSET,
                                                                  res_cvt.SELF_INDICATOR_Y_DIFF,
                                                                  res_cvt.SELF_INDICATOR_Y_OFFSET)
                self_coords = np.reshape(self_coords, (-1, 2))
                self_x_raw = [
                    test_image_x[coord[1]:coord[1] + res_cvt.SELF_INDICATOR_SIZE,
                    coord[0]:coord[0] + res_cvt.SELF_INDICATOR_SIZE]
                    for coord in self_coords]
                result["self"][0].extend([cv.resize(img, ui_constants.NETWORK_SELF_IMG_CROP, cv.INTER_AREA) for img
                                            in self_x_raw])

            if test_image_y["current_gold"] != None:

                result["current_gold"][1].extend([self.current_gold_manager.lookup_by("name", digit)["img_int"] for
                                                  digit in test_image_y[
                "current_gold"] ])
                current_gold_coords = [(res_cvt.CURRENT_GOLD_LEFT_X + res_cvt.CURRENT_GOLD_DIGIT_WIDTH * i +
                  res_cvt.CURRENT_GOLD_X_OFFSET,
                  res_cvt.CURRENT_GOLD_TOP_Y + res_cvt.CURRENT_GOLD_Y_OFFSET) for i in range(4)]
                current_gold_coords = np.reshape(current_gold_coords, (-1, 2))
                current_gold_x_raw = [
                    test_image_x[coord[1]:coord[1] + res_cvt.CURRENT_GOLD_SIZE,
                    coord[0]:coord[0] + res_cvt.CURRENT_GOLD_SIZE]
                    for coord in current_gold_coords]
                result["current_gold"][0].extend([cv.resize(img, ui_constants.NETWORK_CURRENT_GOLD_IMG_CROP,
                                                         cv.INTER_AREA) for img in
                                    current_gold_x_raw])

            if test_image_y["kda"] != None:
                kda_y = []
                for row in test_image_y["kda"]:
                    for i, section in enumerate(row):
                        for digit in section:
                            kda_y.append(self.kda_manager.lookup_by("name", digit)["img_int"])
                        if i != 2:
                            kda_y.append(self.kda_manager.lookup_by("name", "slash")["img_int"])

                result["kda"][1].extend(kda_y)
                result["kda"][0].extend()


            if test_image_y["cs"] != None:
                cs_y = []

                for digit in test_image_y["cs"]:
                    cs_y.append(self.kda_manager.lookup_by("name", digit)["img_int"])

                # utils.show_coords(test_image_x, current_kda_coords, res_cvt.KDA_WIDTH, res_cvt.KDA_HEIGHT)

        # for i, item in enumerate(items_x):
        #     cv.imshow(str(i), item)
        # cv.waitKey(0)
        # self_y = np.array(self_y)[:, np.newaxis]


        return result


    def eval_model(self, model, epoch):
        print("now eval")
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





    def build_new_current_gold_model(self):
        self.class_weights = 1
        self.train_path = app_constants.model_paths["train"]["current_gold_imgs"]
        self.best_path = app_constants.model_paths["best"]["current_gold_imgs"]
        self.training_data_generator = lambda: generate.generate_training_data_nonsquare(self.current_gold_manager.get_imgs(),
                                                                               1000,
                                                                               ui_constants.NETWORK_CURRENT_GOLD_IMG_CROP)
        self.network = CurrentGoldImgNetwork()
        self.X_test, self.Y_test = self.load_img_test_data()["current_gold"]
        self._build_new_img_model()


    def build_new_champ_model(self):
        self.class_weights = 1
        self.train_path = app_constants.model_paths["train"]["champ_imgs"]
        self.best_path = app_constants.model_paths["best"]["champ_imgs"]
        self.training_data_generator = lambda: generate.generate_training_data(self.champ_manager.get_imgs(), 100,
                                                                               ui_constants.NETWORK_CHAMP_IMG_CROP)
        self.network = ChampImgNetwork()
        self.X_test, self.Y_test = self.load_img_test_data()["champs"]
        self._build_new_img_model()


    def build_new_item_model(self):
        self.class_weights = 1
        self.train_path = app_constants.model_paths["train"]["item_imgs"]
        self.best_path = app_constants.model_paths["best"]["item_imgs"]
        self.training_data_generator = lambda: generate.generate_training_data(self.item_manager.get_imgs(), 100,
                                                                               ui_constants.NETWORK_ITEM_IMG_CROP)
        self.network = ItemImgNetwork()
        self.X_test, self.Y_test = self.load_img_test_data()["items"]
        self._build_new_img_model()


    def build_new_self_model(self):
        self.class_weights = 1
        self.train_path = app_constants.model_paths["train"]["self_imgs"]
        self.best_path = app_constants.model_paths["best"]["self_imgs"]
        self.training_data_generator = lambda: generate.generate_training_data(self.self_manager.get_imgs(), 1024,
                                                                               ui_constants.NETWORK_SELF_IMG_CROP, True)
        self.network = SelfImgNetwork()
        self.X_test, self.Y_test = self.load_img_test_data()["self"]
        self._build_new_img_model()





    def build_new_cs_model(self):
        self.class_weights = 1
        self.train_path = app_constants.model_paths["train"]["cs_imgs"]
        self.best_path = app_constants.model_paths["best"]["cs_imgs"]
        self.training_data_generator = lambda: generate.generate_training_data_rect(self.cs_manager.get_imgs(),
                                                                                    500,
                                                                                    ui_constants.NETWORK_CS_IMG_CROP)
        self.network = DigitRecognitionNetwork(lambda: CSManager().get_num("img_int"),
                                                       ui_constants.NETWORK_CS_IMG_CROP)
        self.X_test, self.Y_test = self.load_img_test_data()["cs"]

        self._build_new_img_model()


    def build_new_lvl_model(self):
        self.class_weights = 1
        self.train_path = app_constants.model_paths["train"]["lvl_imgs"]
        self.best_path = app_constants.model_paths["best"]["lvl_imgs"]
        self.training_data_generator = lambda: generate.generate_training_data_rect(self.lvl_manager.get_imgs(),
                                                                                         500,
                                                                                         ui_constants.NETWORK_LVL_IMG_CROP)
        self.network = DigitRecognitionNetwork(lambda: LvlManager().get_num("img_int"),
                                                       ui_constants.NETWORK_LVL_IMG_CROP)
        self.X_test, self.Y_test = self.load_img_test_data()["lvl"]

        self._build_new_img_model()


    def _build_new_img_model(self):
        self.start_generating_train_data(3)
        self.build_new_model()
        self.stop_generating_train_data()


class StaticTrainingDataTrainer(Trainer):

    def determine_best_eval(self, scores):
        # epoch counter is 1 based
        return np.argmax(scores[:,0]) + 1


    def weighted_accuracy(self, preds_sparse, targets_sparse, class_weights):
        max_achievable_score = np.sum(class_weights[targets_sparse])
        matching_preds_sparse = targets_sparse[np.equal(targets_sparse, preds_sparse)]
        actually_achieved_score = np.sum(class_weights[matching_preds_sparse])
        return actually_achieved_score / max_achievable_score

    def eval_model(self, model, epoch, prior=None):
        y_pred_prob = []
        for chunk in utils.chunks(self.X_test, 1024):
            y_pred_prob.extend(model.predict(np.array(chunk)))
        y_pred_prob = np.array(y_pred_prob)
        if prior:
            y_pred_prob = y_pred_prob / prior


        y_pred = np.argmax(y_pred_prob, axis=1)

        acc = sum(np.equal(y_pred, self.Y_test)) / len(self.Y_test)
        weighted_acc = self.weighted_accuracy(y_pred, self.Y_test, self.class_weights)
        # weighted_acc = weighted_accuracy(y_pred, self.Y_test, self.class_weights)
        # with tf.Session() as sess:
        #     weighted_acc = sess.run(weighted_acc)
        precision, recall, f1, support = precision_recall_fscore_support(self.Y_test, y_pred, average='macro')

        report = classification_report(self.Y_test, y_pred, labels=range(len(self.target_names)),
                                       target_names=self.target_names)
        # confusion = confusion_matrix(self.Y_test, y_pred)
        avg_binary_auc, avg_binary_f1, thresholds = self.get_cum_scores(self.Y_test, y_pred_prob)

        self.log_output(acc, weighted_acc, f1, precision, recall, avg_binary_f1, avg_binary_auc,
                        report, thresholds, epoch)

        return avg_binary_f1, avg_binary_auc, acc, precision, recall, f1, thresholds

    def standalone_eval(self):

        # with open(app_constants.model_paths["best"]["next_items_early"] + "my_model1_thresholds.json") as f:
        #     thresholds = json.load(f)
        thresholds = 1
        print("Loading training data")
        dataloader = data_loader.NextItemsDataLoader(app_constants.train_paths["next_items_early_processed"])
        self.X, self.Y = dataloader.get_train_data()
        print("Loading test data")
        self.X_test, self.Y_test = dataloader.get_test_data()
        self.train_y_distrib = Counter(self.Y)
        self.test_y_distrib = Counter(self.Y_test)
        self.target_names = [target["name"] for target in sorted(list(ItemManager().get_ints().values()), key=lambda
            x: x["int"])]

        total_y_distrib = self.train_y_distrib + self.test_y_distrib
        missing_items = Counter(list(range(len(self.target_names)))) - total_y_distrib
        print(f"missing items are: {missing_items}")
        # assert(missing_items == Counter([0]))
        total_y = sum(list(total_y_distrib.values()))
        total_y_distrib_sorted = np.array([count for count in np.array(sorted(list((total_y_distrib +
                                                                                    missing_items).items()),
                                                                              key=lambda x: x[0]))[:, 1]])
        self.class_weights = total_y / total_y_distrib_sorted
        # don't include weights for empty item
        self.class_weights[0] = 0

        self.network = NextItemEarlyGameNetwork()
        self.network.network_config["class_weights"] = self.class_weights
        self.network = self.network.build()



        model = tflearn.DNN(self.network)
        model_path = glob.glob(app_constants.model_paths["best"]["next_items_early"] + "my_model*")[0]
        model_path = model_path.rpartition('.')[0]
        model.load(model_path)


        with open("lololo", "w") as self.logfile:
            # thresholds = self.eval_model(model, 0)[-1]
            self.eval_model(model, 0)[-1]


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


    def log_output(self, main_test_eval, weighted_acc, f1, precision, recall, avg_binary_f1, avg_binary_auc, \
                                                    classification,
                   thresholds,
                   epoch_counter):

        for output in [sys.stdout, self.logfile]:
            output.write("Epoch {0}\n".format(epoch_counter + 1))
            output.write("1. Acc {0:.4f}\n".format(main_test_eval))
            output.write("2. Weighted Acc {0:.4f}\n".format(weighted_acc))
            output.write('3. F-1 {0:.4f}\n'.format(f1))
            output.write('4. Precision {0:.4f}\n'.format(precision))
            output.write('5. Recall {0:.4f}\n'.format(recall))
            output.write('6. Avg binary F1 {0:.4f}\n'.format(avg_binary_f1))
            output.write('7. Avg binary auc {0:.4f}\n'.format(avg_binary_auc))
            output.write('8. Classification report \n {} \n'.format(classification))
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
        dataloader = data_loader.SortedNextItemsDataLoader(app_constants.train_paths["next_items_processed_sorted"])
        self.X, self.Y = dataloader.get_train_data()
        print("Loading test data")
        self.X_test, self.Y_test = dataloader.get_test_data()
        self.train_y_distrib = Counter(self.Y)
        self.test_y_distrib = Counter(self.Y_test)

        total_y_distrib = self.train_y_distrib + self.test_y_distrib
        missing_items = Counter(list(range(len(self.target_names)))) - total_y_distrib
        print(f"missing items are: {missing_items}")
        # assert(missing_items == Counter([0]))
        total_y = sum(list(total_y_distrib.values()))
        total_y_distrib_sorted = np.array([count for count in np.array(sorted(list((total_y_distrib +
                                                                               missing_items).items()),
                                                                      key=lambda x: x[0]))[:,1]])
        #self.class_weights = total_y / total_y_distrib_sorted

        effective_num = 1.0 - np.power(0.9999, total_y_distrib_sorted)
        self.class_weights = (1.0 - 0.9999) / np.array(effective_num)
        self.class_weights = self.class_weights / np.sum(self.class_weights) * int(ItemManager().get_num("int"))

        #don't include weights for empty item
        self.class_weights[0] = 0
        self.network.network_config["class_weights"] = self.class_weights
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
    # t = StaticTrainingDataTrainer()
    # t.build_next_items_early_game_model()
    #t.standalone_eval()


    s = DynamicTrainingDataTrainer()
    s.build_new_cs_model()

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
