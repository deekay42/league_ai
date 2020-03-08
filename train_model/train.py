import json
import json
import multiprocessing
import shutil
import sys
from collections import Counter
from multiprocessing import Process, JoinableQueue, Queue
import random

from sklearn.metrics import auc, \
    classification_report, precision_recall_curve, precision_recall_fscore_support

from constants.ui_constants import ResConverter
from train_model import generate, data_loader
from train_model.model import *
from train_model.network import *
from utils import utils
from utils.artifact_manager import ChampManager, ItemManager, SimpleManager
import sklearn
from tflearn.data_utils import to_categorical
from scipy import spatial


class MonitorCallback(tflearn.callbacks.Callback):

    def __init__(self, f):
        super().__init__()
        self.f = f


    def on_epoch_end(self, training_state):
        self.f.write(
            "Epoch {0} accuracy {1:.2f} | loss {2:.2f}\n".format(training_state.epoch, training_state.acc_value,
                                                                 training_state.global_loss))
        self.f.flush()


class MonitorCallbackRegression(MonitorCallback):

    def __init__(self, f):
        super().__init__(f)



    def on_epoch_end(self, training_state):
        self.f.write(
            "Epoch {0}  loss {1:.2f}\n".format(training_state.epoch, training_state.global_loss))
        self.f.flush()


class Trainer(ABC):

    def __init__(self):
        self.num_epochs = 50
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
            with tf.Graph().as_default():
                with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
                    tflearn.is_training(True, sess)
                    self.network = self.network.build()
                    model = tflearn.DNN(self.network, session=sess)
                    sess.run(tf.global_variables_initializer())
                    if self.champ_embs is not None:
                        embeddingWeights = tflearn.get_layer_variables_by_name('my_champ_embs')[0]
                        model.set_weights(embeddingWeights, self.champ_embs)
                        embeddingWeights = tflearn.get_layer_variables_by_name('opp_champ_embs')[0]
                        model.set_weights(embeddingWeights, self.opp_champ_embs)
                    scores = []
                    for epoch in range(self.num_epochs):
                        x, y = self.get_train_data()
                        # x = x[:1000]
                        # y = y[:1000]
                        # x[:, 1:6] = [40, 42, 142, 84, 51]
                        # # lul = model.evaluate(x,y)
                        #
                        # feed_dict = tflearn.utils.feed_dict_builder(x, y, model.inputs,
                        #                                             model.targets)
                        # coord = tf.train.Coordinator()
                        # df = tflearn.data_flow.FeedDictFlow(feed_dict, coord,
                        #                             batch_size=1,
                        #                             dprep_dict=None,
                        #                             daug_dict=None,
                        #                             index_array=None,
                        #                             num_threads=1)
                        # df.reset()
                        # df.start()
                        # res = []
                        # feed_batch = df.next()
                        # tflearn.is_training(True, sess)
                        # while feed_batch:
                        #     r = sess.run(["cond/Merge:0"], feed_batch)[0]
                        #     # r = sess.run(["Reshape_1:0"], feed_batch)[0]
                        #     res.append(r)
                        #     feed_batch = df.next()
                        #
                        # np.savetxt("lulz", np.array(res).reshape((5000, 3)), fmt="%f")

                        model.fit(x, y, n_epoch=1, shuffle=True, validation_set=None,
                                  show_metric=True, batch_size=self.batch_size, run_id='whaddup_glib_globs' + str(epoch),
                                  callbacks=self.monitor_callback)
                        model.save(self.train_path + self.model_name + str(epoch + 1))


                        # # for i, img in enumerate(self.X_test):
                        # #     cv.imshow(str(i), img)
                        # # cv.waitKey(0)
                        # y = model.predict(self.X_test)
                        # y = [np.argmax(y_) for y_ in y]
                        # # y = [np.argmax(np.reshape(y_,(5,5)), axis=1) for y_ in y]
                        # # y_actual = [np.argmax(np.reshape(y_,(5,5)), axis=1) for y_ in self.Y_test]
                        # y_actual = self.Y_test
                        # # print("Pred Actual")
                        # for i in range(len(y_actual)):
                        #     a_text = self.manager.lookup_by('img_int', y[i])['name']
                        #     b_text = self.manager.lookup_by('img_int', self.Y_test[i])['name']
                        #     a = y[i]
                        #     b = y_actual[i]
                        #     if not np.all(np.equal(a,b)):
                        #         print(f"----->{i}: {a_text} {b_text}")
                        #     else:
                        #         print(f"{i}: {a_text} {b_text}")
                        # print("Raw test data predictions: {0}".format(y))
                        # print("Actual test data  values : {0}".format(y_actual))

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

                        score = self.eval_model(model, epoch, self.X_test, self.Y_test)

                        # self.eval_model(model, epoch, prior=score[-1])
                        scores.append(score)
        return scores


    def get_train_data(self):
        return self.X, self.Y


    def get_train_data_balanced(self, size=1e6):
        chunk_per_item = int(size//sum([1 if y != [] else 0 for y in self.Y_indices]))
        print(f"building new epoch with chunksize:{chunk_per_item}")
        indices = []
        for y_indices in self.Y_indices:
            if y_indices:
                indices.extend(np.random.choice(y_indices, size=chunk_per_item))
        print("epoch built")
        return self.X[indices], self.Y[indices]


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
        self.class_weights = 1
        self.train_path = app_constants.model_paths["train"][self.elements]
        self.best_path = app_constants.model_paths["best"][self.elements]
        self.network_crop = ResConverter.network_crop[self.elements]
        self.manager = SimpleManager(self.elements)
        self.training_data_generator = lambda: generate.generate_training_data_rect(self.manager.get_imgs(),
                                                                                    self.epoch_size,
                                                                                    self.network_crop)
        self.network = DigitRecognitionNetwork(lambda: self.manager.get_num("img_int"), self.network_crop)



    def get_train_data(self):
        x, y = self.queue.get()
        self.queue.task_done()
        return x, y


    def load_test_data(self):
        with open('test_data/easy/test_labels.json', "r") as f:
            elems = json.load(f)

        base_path = "test_data/easy/"
        result_x, result_y = [], []

        for key in elems:
            test_image_y = elems[key]
            test_image_x = cv.imread(base_path + test_image_y["filename"])
            res_cvt = ui_constants.ResConverter(*(test_image_y["res"].split(",")), test_image_y.get("hud_scale", None),
                                                test_image_y["summ_names_displayed"])

            if test_image_y[self.elements] is not None:
                result_y.extend(self.extract_y_data(test_image_y[self.elements]))
                result_x.extend(self.model(res_cvt).extract_imgs(test_image_x))

        return result_x, result_y


    def extract_y_data(self, data):
        for elem in data:
            yield self.manager.lookup_by("name", elem)["img_int"]


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


    def build_new_img_model(self):
        self.start_generating_train_data(3)
        self.build_new_model()
        self.stop_generating_train_data()


class KDATrainer(DynamicTrainingDataTrainer):

    def __init__(self):
        self.model = KDAImgModel
        self.elements = "kda"
        self.epoch_size = 500
        super().__init__()
        self.X_test, self.Y_test = self.load_test_data()


    def extract_y_data(self, data):
        for row in data:
            for i, section in enumerate(row):
                for digit in section:
                    yield self.manager.lookup_by("name", digit)["img_int"]
                if i != 2:
                    yield self.manager.lookup_by("name", "slash")["img_int"]
#
#
# class CSTrainer(DynamicTrainingDataTrainer):
#
#     def __init__(self):
#         self.model = CSImgModel
#         self.elements = "cs"
#         self.epoch_size = 500
#         super().__init__()
#
#
#     def extract_y_data(self, data):
#         for number in data:
#             for digit in str(number):
#                 yield self.manager.lookup_by("name", str(digit))["img_int"]
#
#
# class LvlTrainer(DynamicTrainingDataTrainer):
#
#     def __init__(self):
#         self.model = LvlImgModel
#         self.element = "lvl"
#         self.epoch_size = 500
#         super().__init__()
#
#     def extract_y_data(self, data):
#         for number in data:
#             for digit in str(number):
#                 yield self.manager.lookup_by("name", str(digit))["img_int"]
#
#
#
# class CurrentGoldTrainer(DynamicTrainingDataTrainer):
#
#     def __init__(self):
#         self.model = CurrentGoldImgModel
#         self.element = "current_gold"
#         self.epoch_size = 1000
#         super().__init__()
#

class ChampImgTrainer(DynamicTrainingDataTrainer):

    def __init__(self):
        self.model = ChampImgModel
        self.elements = "champs"
        self.epoch_size = 100
        super().__init__()
        self.manager = ChampManager()
        self.training_data_generator = lambda: generate.generate_training_data_champs(self.manager.get_imgs(), self.epoch_size,
                                                                               self.network_crop)
        self.network = ChampImgNetwork()
        self.X_test, self.Y_test = self.load_test_data()


class ItemImgTrainer(DynamicTrainingDataTrainer):

    def __init__(self):
        self.model = ItemImgModel
        self.elements = "items"

        self.epoch_size = 100
        super().__init__()
        self.manager = ItemManager()
        self.training_data_generator = lambda: generate.generate_training_data(self.manager.get_imgs(), self.epoch_size,
                                                                               self.network_crop)
        self.network = ItemImgNetwork()
        self.X_test, self.Y_test = self.load_test_data()


class SelfTrainer(DynamicTrainingDataTrainer):

    def __init__(self):
        self.model = SelfImgModel
        self.elements = "self"
        self.epoch_size = 1000
        super().__init__()

        self.network = SelfImgNetwork()
        self.training_data_generator = lambda: generate.generate_training_data(self.manager.get_imgs(), self.epoch_size,
                                                                               self.network_crop, True)


class PositionsTrainer(Trainer):

    def __init__(self):
        super().__init__()
        self.num_epochs = 30


    def eval_model(self, model, epoch):
        # pred = model.predict(np.array(self.X_test))
        # acc = PositionsNetwork.multi_class_acc_positions(pred, self.Y_test, None)
        # with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        #     acc = sess.run(acc)
        acc = model.evaluate(np.array(self.X_test), np.array(self.Y_test), batch_size=self.batch_size)[0]
        self.log_output(acc, epoch)
        return acc


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


    def train(self):
        self.train_path = app_constants.model_paths["train"]["positions"]
        self.best_path = app_constants.model_paths["best"]["positions"]
        self.network = PositionsNetwork()
        print("Loading training data")
        dataloader = data_loader.PositionsDataLoader()
        self.X, self.Y = dataloader.get_train_data()
        print("Loading test data")
        try:
            self.X_test, self.Y_test = dataloader.get_test_data()
        except FileNotFoundError:
            dataloader.train2test()
            dataloader = data_loader.PositionsDataLoader()
            self.X, self.Y = dataloader.get_train_data()
            self.X_test, self.Y_test = dataloader.get_test_data()
        self.build_new_model()



class NextItemsTrainer(Trainer):

    def __init__(self):
        super().__init__()
        self.manager = ItemManager()
        self.champ_embs = None
        self.opp_champ_embs = None
        self.num_epochs = 200

    def determine_best_eval(self, scores):
        # epoch counter is 1 based
        return np.argmax(scores[:, 0]) + 1


    def weighted_accuracy(self, preds_sparse, targets_sparse, class_weights):
        max_achievable_score = np.sum(class_weights[targets_sparse])
        matching_preds_sparse = targets_sparse[np.equal(targets_sparse, preds_sparse)]
        actually_achieved_score = np.sum(class_weights[matching_preds_sparse])
        return actually_achieved_score / max_achievable_score


    def eval_model(self, model, epoch, x_test, y_test, prior=None):
        y_pred_prob = []
        for chunk in utils.chunks(x_test, 1024):
            y_pred_prob.extend(model.predict(np.array(chunk)))
        y_pred_prob = np.array(y_pred_prob)
        if prior:
            y_pred_prob = y_pred_prob / prior

        y_pred = np.argmax(y_pred_prob, axis=1)

        acc = sum(np.equal(y_pred, y_test)) / len(y_test)
        weighted_acc = self.weighted_accuracy(y_pred, y_test, self.class_weights)
        # weighted_acc = weighted_accuracy(y_pred, self.Y_test, self.class_weights)
        # with tf.Session() as sess:
        #     weighted_acc = sess.run(weighted_acc)
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='macro')

        report = classification_report(y_test, y_pred, labels=range(len(self.target_names)),
                                       target_names=self.target_names)
        # confusion = confusion_matrix(self.Y_test, y_pred)
        avg_binary_auc, avg_binary_f1, thresholds = self.get_cum_scores(y_test, y_pred_prob)

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
        # make sure empty item never appears
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
        f1_scores = 2 * precision * recall
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
            if weighted_acc: output.write("2. Weighted Acc {0:.4f}\n".format(weighted_acc))
            if f1: output.write('3. F-1 {0:.4f}\n'.format(f1))
            if precision: output.write('4. Precision {0:.4f}\n'.format(precision))
            if recall: output.write('5. Recall {0:.4f}\n'.format(recall))
            if avg_binary_f1: output.write('6. Avg binary F1 {0:.4f}\n'.format(avg_binary_f1))
            if avg_binary_auc: output.write('7. Avg binary auc {0:.4f}\n'.format(avg_binary_auc))
            if classification: output.write('8. Classification report \n {} \n'.format(classification))
            output.write("\n\n")
            output.flush()

        # with open(self.train_path + self.model_name + str(epoch_counter + 1) + "_thresholds.json", "w") as f:
        #     f.write(json.dumps(thresholds.tolist()))


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

        my_champ_embs_dst = np.load("my_champ_embs_dst.npy")
        opp_champ_embs_dst = np.load("opp_champ_embs_dst.npy")
        my_champ_embs_dst = np.concatenate([[[0, 0, 0, 0.1]], my_champ_embs_dst], axis=0)
        opp_champ_embs_dst = np.concatenate([[[0, 0, 0, 0.1]], opp_champ_embs_dst], axis=0)

        self.champ_embs = my_champ_embs_dst[:, :3]
        self.opp_champ_embs = opp_champ_embs_dst[:, :3]
        self.network = NextItemEarlyGameNetwork(my_champ_embs_dst[:, -1], opp_champ_embs_dst[:, -1])

        self.train_path = app_constants.model_paths["train"]["next_items_early"]
        self.best_path = app_constants.model_paths["best"]["next_items_early"]

        print("Loading training data")
        dataloader = data_loader.SortedNextItemsDataLoader(app_constants.train_paths[
                                                               "next_items_processed_elite_sorted_inf"])
        dataloader_lower = data_loader.SortedNextItemsDataLoader(app_constants.train_paths[
                                                               "next_items_processed_lower_sorted_inf"])
        self.X, self.Y = dataloader.get_train_data()
        print("Loading test data")
        self.X_test, self.Y_test = dataloader.get_test_data()
        X_test_lower, Y_test_lower = dataloader_lower.get_test_data()

        self.train_y_distrib = Counter(self.Y)
        self.test_y_distrib = Counter(self.Y_test)

        total_y_distrib = self.train_y_distrib + self.test_y_distrib
        missing_items = Counter(list(range(len(self.target_names)))) - total_y_distrib
        print(f"missing items are: {missing_items}")
        # assert(missing_items == Counter([0]))
        total_y = sum(list(total_y_distrib.values()))
        total_y_distrib_sorted = np.array([count for count in np.array(sorted(list((total_y_distrib +
                                                                                    missing_items).items()),
                                                                              key=lambda x: x[0]))[:, 1]])
        # self.class_weights = np.sqrt(total_y / total_y_distrib_sorted)


        effective_num =  1.0 - np.power(0.99, total_y_distrib_sorted)
        self.class_weights = (1.0 - 0.99) / np.array(effective_num)
        self.class_weights = self.class_weights / np.sum(self.class_weights) * int(ItemManager().get_num("int"))
        # self.class_weights = np.array([1.0]*int(ItemManager().get_num("int")))
        #executioners
        self.class_weights[106] *= 3
        #qss
        self.class_weights[113] *= 2
        #cull
        self.class_weights[27] *= 2.5
        #last whisper
        self.class_weights[62] *= 3
        #stopwatch
        self.class_weights[45] *= 2
        #dark seal
        self.class_weights[26] *= 1.5

        # self.class_weights = np.array([1.0]*int(ItemManager().get_num("int")))
        self.network.network_config["class_weights"] = self.class_weights


        self.X = self.X.astype(np.float32)
        self.X_test = np.concatenate([self.X_test, X_test_lower], axis=0)
        self.Y_test = np.concatenate([self.Y_test, Y_test_lower], axis=0)

        self.X_test = self.X_test.astype(np.float32)

        model = NextItemModel("early")
        self.X = model.scale_inputs(np.array(self.X).astype(np.float32))
        self.X_test = model.scale_inputs(np.array(self.X_test).astype(np.float32))

        self.build_new_model()


    def fit_input(self, X, scaler_name):
        min_max_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
        min_max_scaler.fit(np.reshape(X, (-1, 1)))
        # dump(min_max_scaler, self.train_path + scaler_name +"_scaler", compress=True)
        # dump(min_max_scaler, self.best_path + scaler_name + "_scaler", compress=True)
        return min_max_scaler


    @staticmethod
    def no_full_items_completed(data):
        y = data[:, -1]
        starter_ints = ItemManager().get_starter_ints()
        full_item_ints = ItemManager().get_full_item_ints()
        exclude_starters = np.logical_not(np.isin(y, list(starter_ints)))
        exclude_noncompletes = np.isin(y, list(full_item_ints))

        pos = data[:, 1].astype(np.int32)
        data_items = data[:, 12:6 * 12]
        data_items = np.reshape(data_items, (-1, 5, 12))
        target_summ_items = data_items[range(len(pos)), pos, ::2]
        target_summs_full_items_boolean = np.isin(target_summ_items, list(full_item_ints))
        no_full_items_complete = np.logical_not(np.any(target_summs_full_items_boolean, axis=1))
        valid_indices = np.all([no_full_items_complete, exclude_starters, exclude_noncompletes], axis=0)
        return valid_indices


    @staticmethod
    def only_full_items_completed(data):
        y = data[:, -1]
        starter_ints = ItemManager().get_starter_ints()
        exclude_starters = np.logical_not(np.isin(y, list(starter_ints)))

        pos = data[:, 1]
        full_item_ints = ItemManager().get_full_item_ints()
        data_items = data[:, 12:6 * 12]
        data_items = np.reshape(data_items, (-1, 5, 12))
        target_summ_items = data_items[range(len(pos)), pos, ::2]
        target_summs_full_items_boolean = np.isin(target_summ_items, list(full_item_ints))
        only_full_items_complete = np.any(target_summs_full_items_boolean, axis=1)
        valid_indices = np.logical_and(only_full_items_complete, exclude_starters)
        return valid_indices


    def build_next_items_late_game_model(self):
        self.target_names = [target["name"] for target in sorted(list(ItemManager().get_ints().values()), key=lambda
            x: x["int"])]

        my_champ_embs_dst = np.load("my_champ_embs_dst.npy")
        opp_champ_embs_dst = np.load("opp_champ_embs_dst.npy")
        my_champ_embs_dst = np.concatenate([[[0, 0, 0, 0.1]], my_champ_embs_dst], axis=0)
        opp_champ_embs_dst = np.concatenate([[[0, 0, 0, 0.1]], opp_champ_embs_dst], axis=0)

        self.champ_embs = my_champ_embs_dst[:, :3]
        self.opp_champ_embs = opp_champ_embs_dst[:, :3]
        self.network = NextItemLateGameNetwork(my_champ_embs_dst[:,-1], opp_champ_embs_dst[:,-1])
        self.train_path = app_constants.model_paths["train"]["next_items_late"]
        self.best_path = app_constants.model_paths["best"]["next_items_late"]

        print("Loading training data")
        dataloader_elite = data_loader.SortedNextItemsDataLoader(app_constants.train_paths[
                                                               "next_items_processed_elite_sorted_complete"])
        dataloader_lower = data_loader.SortedNextItemsDataLoader(app_constants.train_paths[
                                                                     "next_items_processed_lower_sorted_complete"])
        X_elite, Y_elite = dataloader_elite.get_train_data(NextItemsTrainer.only_full_items_completed)
        print("Loading test data")
        X_test_elite, Y_test_elite = dataloader_elite.get_test_data(NextItemsTrainer.only_full_items_completed)

        X_lower, Y_lower = dataloader_lower.get_train_data(NextItemsTrainer.only_full_items_completed)
        print("Loading test data")
        X_test_lower, Y_test_lower = dataloader_lower.get_test_data(NextItemsTrainer.only_full_items_completed)

        self.X = np.concatenate([X_elite, X_lower], axis=0)
        self.Y = np.concatenate([Y_elite, Y_lower], axis=0)
        self.X_test = np.concatenate([X_test_elite, X_test_lower], axis=0)
        self.Y_test = np.concatenate([Y_test_elite, Y_test_lower], axis=0)

        # self.X = self.X[:1000]
        # self.Y = self.Y[:1000]
        # self.X_test = self.X[:1000]
        # self.Y_test = self.Y[:1000]

        self.train_y_distrib = Counter(self.Y)
        self.test_y_distrib = Counter(self.Y_test)

        total_y_distrib = self.train_y_distrib + self.test_y_distrib
        missing_items = Counter(list(range(len(self.target_names)))) - total_y_distrib
        print(f"missing items are: {missing_items}")
        # assert(missing_items == Counter([0]))
        total_y = sum(list(total_y_distrib.values()))
        total_y_distrib_sorted = np.array([count for count in np.array(sorted(list((total_y_distrib +
                                                                                    missing_items).items()),
                                                                              key=lambda x: x[0]))[:, 1]])
        # self.class_weights = np.sqrt(total_y / total_y_distrib_sorted)

        effective_num =  1.0 - np.power(0.99, total_y_distrib_sorted)
        self.class_weights = (1.0 - 0.99) / np.array(effective_num)
        non_complete_ints = (ItemManager().get_ints().keys() - ItemManager().get_completes().keys())
        self.class_weights = [0 if index in non_complete_ints else weight for index, weight in enumerate(
            self.class_weights)]
        self.class_weights = self.class_weights / np.sum(self.class_weights) * int(len(ItemManager().get_completes()))
        # self.class_weights = np.array([1.0]*int(ItemManager().get_num("int")))
        self.class_weights[106] *= 3


        # self.class_weights = np.array([1.0]*int(ItemManager().get_num("int")))
        self.network.network_config["class_weights"] = self.class_weights
        self.X = self.X.astype(np.float32)
        self.X_test = self.X_test.astype(np.float32)
        model = NextItemModel("late")
        self.X = model.scale_inputs(np.array(self.X).astype(np.float32))
        self.X_test = model.scale_inputs(np.array(self.X_test).astype(np.float32))
        self.build_new_model()


    def build_next_items_starter_model(self):
        self.target_names = [target["name"] for target in sorted(list(ItemManager().get_ints().values()), key=lambda
            x: x["int"])]


        def condition(data):
            pos = data[:, 1]
            all_starter_items_ints = ItemManager().get_starter_ints()
            starter_items = np.isin(data[:, -1], list(all_starter_items_ints))
            data_items = data[:, 12:6*12]
            data_items = np.reshape(data_items, (-1, 5,12))
            empty_items = data_items[range(len(pos)), pos, 1] == 6
            return np.logical_and(empty_items, starter_items)


        my_champ_embs_normed = np.load("my_champ_embs_normed.npy")
        opp_champ_embs_normed = np.load("opp_champ_embs_normed.npy")
        my_champ_embs_normed = np.concatenate([[[0, 0, 0]], my_champ_embs_normed], axis=0)
        opp_champ_embs_normed = np.concatenate([[[0, 0, 0]], opp_champ_embs_normed], axis=0)

        self.champ_embs = my_champ_embs_normed
        self.opp_champ_embs = opp_champ_embs_normed
        self.network = NextItemStarterNetwork()
        self.train_path = app_constants.model_paths["train"]["next_items_starter"]
        self.best_path = app_constants.model_paths["best"]["next_items_starter"]

        print("Loading training data")
        dataloader_elite = data_loader.SortedNextItemsDataLoader(app_constants.train_paths[
                                                                     "next_items_processed_elite_sorted_uninf"])
        dataloader_lower = data_loader.SortedNextItemsDataLoader(app_constants.train_paths[
                                                                     "next_items_processed_lower_sorted_uninf"])
        X_elite, Y_elite = dataloader_elite.get_train_data(condition)
        print("Loading test data")
        X_test_elite, Y_test_elite = dataloader_elite.get_test_data(condition)

        X_lower, Y_lower = dataloader_lower.get_train_data(condition)
        print("Loading test data")
        X_test_lower, Y_test_lower = dataloader_lower.get_test_data(condition)

        self.X = np.concatenate([X_elite, X_lower], axis=0)
        self.Y = np.concatenate([Y_elite, Y_lower], axis=0)
        self.X_test = np.concatenate([X_test_elite, X_test_lower], axis=0)
        self.Y_test = np.concatenate([Y_test_elite, Y_test_lower], axis=0)

        # self.X = np.tile(self.X[:20], (10000,1))
        # self.Y = np.tile(self.Y[:20], 10000)
        # self.X_test = self.X
        # self.Y_test = self.Y

        self.train_y_distrib = Counter(self.Y)
        self.test_y_distrib = Counter(self.Y_test)
        self.class_weights = np.array([1.0]*int(ItemManager().get_num("int")))
        self.network.network_config["class_weights"] = self.class_weights
        self.X = self.X.astype(np.float32)
        self.X_test = self.X_test.astype(np.float32)
        model = NextItemModel("starter")
        self.X = model.scale_inputs(np.array(self.X).astype(np.float32))
        self.X_test = model.scale_inputs(np.array(self.X_test).astype(np.float32))
        self.build_new_model()





class FirstItemsTrainer(NextItemsTrainer):

    def build_aux_test_data(self):
        with open('test_data/first_items_test.json', "r") as f:
            self.aux_test_raw = json.load(f)
        self.X_test_aux = []
        self.Y_test_aux = []
        for i, test_case in self.aux_test_raw.items():
            my_team_champs = [0,0,0,0,0]
            my_team_champs[test_case["role"]] = ChampManager().lookup_by("name", test_case["target_summ"])["int"]
            opp_team_champs = [ChampManager().lookup_by("name", champ_name)["int"] for champ_name in test_case[
                "opp_team"]]
            complete_example = [0]*221
            complete_example[0] = test_case["role"]
            complete_example[1:11] = my_team_champs + opp_team_champs

            targets = [ItemManager().lookup_by("name", item_name)["int"] for item_name in test_case["target"]]
            self.X_test_aux.append(complete_example)
            self.Y_test_aux.append(targets)


    def train(self):
        self.target_names = [target["name"] for target in sorted(list(ItemManager().get_ints().values()), key=lambda
            x: x["int"])]

        self.build_aux_test_data()

        my_champ_embs_normed = np.load("my_champ_embs_normed.npy")
        opp_champ_embs_normed = np.load("opp_champ_embs_normed.npy")
        my_champ_embs_normed = np.concatenate([[[0, 0, 0]], my_champ_embs_normed], axis=0)
        opp_champ_embs_normed = np.concatenate([[[0, 0, 0]], opp_champ_embs_normed], axis=0)

        self.champ_embs = my_champ_embs_normed
        self.opp_champ_embs = opp_champ_embs_normed
        self.network = NextItemFirstItemNetwork()

        self.train_path = app_constants.model_paths["train"]["next_items_first_item"]
        self.best_path = app_constants.model_paths["best"]["next_items_first_item"]

        print("Loading training data")
        dataloader_elite = data_loader.SortedNextItemsDataLoader(app_constants.train_paths[
                                                                     "next_items_processed_elite_sorted_complete"])
        dataloader_lower = data_loader.SortedNextItemsDataLoader(app_constants.train_paths[
                                                                     "next_items_processed_lower_sorted_complete"])
        print("Loading elite train data first item")
        X_elite, Y_elite = dataloader_elite.get_train_data(NextItemsTrainer.no_full_items_completed)
        print("Loading elite test data first item")
        X_test_elite, Y_test_elite = dataloader_elite.get_test_data(NextItemsTrainer.no_full_items_completed)
        print("Loading lower train data first item")
        X_lower, Y_lower = dataloader_lower.get_train_data(NextItemsTrainer.no_full_items_completed)
        print("Loading lower test data first item")
        X_test_lower, Y_test_lower = dataloader_lower.get_test_data(NextItemsTrainer.no_full_items_completed)

        self.X = np.concatenate([X_elite, X_lower], axis=0)
        self.Y = np.concatenate([Y_elite, Y_lower], axis=0)
        self.X_test = np.concatenate([X_test_elite, X_test_lower], axis=0)
        self.Y_test = np.concatenate([Y_test_elite, Y_test_lower], axis=0)

        # print("calculating indices per class")
        # num_items = ItemManager().get_num("int")
        # self.Y_indices = [[] for _ in range(num_items)]
        # for x_index, y in enumerate(self.Y):
        #     self.Y_indices[y].append(x_index)
        #
        # print("got all indices")
        # print(len(self.Y_indices))
        # blacklist = []
        # #don't want super minor occurrences
        # for i, y_indices in enumerate(self.Y_indices):
        #     print(len(y_indices))
        #
        #     if len(y_indices) < 0.0018 * len(self.X):
        #         self.Y_indices[i] = []
        #         blacklist.append(i)
        #
        # valid_test_indices = np.logical_not(np.isin(self.Y_test, blacklist))
        # self.X_test = self.X_test[valid_test_indices]
        # self.Y_test = self.Y_test[valid_test_indices]
        # print("corrected indices")


        # self.X = np.tile(self.X[:20], (10000,1))
        # self.Y = np.tile(self.Y[:20], 10000)
        # self.X_test = self.X
        # self.Y_test = self.Y

        # self.train_y_distrib = Counter(self.Y)
        # self.test_y_distrib = Counter(self.Y_test)
        #
        # total_y_distrib = self.train_y_distrib + self.test_y_distrib
        # missing_items = Counter(list(range(len(self.target_names)))) - total_y_distrib
        # print(f"missing items are: {missing_items}")
        # # assert(missing_items == Counter([0]))
        # total_y = sum(list(total_y_distrib.values()))
        #
        # total_y_distrib_sorted = np.array([count for count in np.array(sorted(list((total_y_distrib +
        #                                                                             missing_items).items()),
        #                                                                       key=lambda x: x[0]))[:, 1]])
        # print("This is the class distrib:")
        # print(total_y_distrib_sorted)
        # self.class_weights = total_y_distrib_sorted / total_y
        #
        # print("These are the class weights:")
        # print(self.class_weights)
        #
        # # self.class_weights = np.array([1.0]*int(ItemManager().get_num("int")))
        # self.network.network_config["class_weights"] = self.class_weights

        self.train_y_distrib = Counter(self.Y)
        self.test_y_distrib = Counter(self.Y_test)
        self.class_weights = np.array([1.0]*int(ItemManager().get_num("int")))
        self.network.network_config["class_weights"] = self.class_weights


        self.X = self.X.astype(np.float32)
        self.X_test = self.X_test.astype(np.float32)
        model = NextItemModel("first_item")
        self.X = model.scale_inputs(np.array(self.X).astype(np.float32))
        self.X_test = model.scale_inputs(np.array(self.X_test).astype(np.float32))

        self.build_new_model()


    def eval_model_extra(self, model, epoch, x_test, y_test):
        y_pred_prob = []
        for chunk in utils.chunks(x_test, 1024):
            y_pred_prob.extend(model.predict(np.array(chunk)))
        y_pred_prob = np.array(y_pred_prob)
        y_preds = np.argmax(y_pred_prob, axis=1)



        for y_pred, y_actual, test_case in zip(y_preds, y_test, self.aux_test_raw.values()):
            if y_pred not in y_actual:
                print(f"test_case: {test_case}")
                print(ItemManager().lookup_by('img_int', y_pred)['name'])
                print("\n\n")

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

        acc = sum([y_pred in y_actual for y_pred, y_actual in zip(y_preds, y_test)])/len(y_test)
        self.log_output(acc, None, None, None, None, None, None,
                        None, None, epoch)

        return acc


    def train_neural_network(self):
        with tf.device("/gpu:0"):
            with tf.Graph().as_default():
                with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
                    tflearn.is_training(True, sess)
                    self.network = self.network.build()
                    model = tflearn.DNN(self.network, session=sess)
                    sess.run(tf.global_variables_initializer())
                    if self.champ_embs is not None:
                        embeddingWeights = tflearn.get_layer_variables_by_name('my_champ_embs')[0]
                        model.set_weights(embeddingWeights, self.champ_embs)
                        embeddingWeights = tflearn.get_layer_variables_by_name('opp_champ_embs')[0]
                        model.set_weights(embeddingWeights, self.opp_champ_embs)
                    scores = []
                    for epoch in range(self.num_epochs):
                        x, y = self.get_train_data()

                        model.fit(x, y, n_epoch=1, shuffle=True, validation_set=None,
                                  show_metric=True, batch_size=self.batch_size, run_id='whaddup_glib_globs' + str(epoch),
                                  callbacks=self.monitor_callback)
                        model.save(self.train_path + self.model_name + str(epoch + 1))

                        score = self.eval_model(model, epoch, self.X_test, self.Y_test)
                        score = self.eval_model_extra(model, epoch, self.X_test_aux, self.Y_test_aux)




                        scores.append(score)
        return scores


class ChampsEmbeddingTrainer(Trainer):

    def __init__(self):
        super().__init__()
        self.manager = ItemManager()
        self.num_epochs = 2000
        self.q = None
        self.network = ChampEmbeddings()


    def determine_best_eval(self, scores):
        # epoch counter is 1 based
        return np.argmax(scores[:, 0]) + 1


    #     def get_train_data(self):
    #         return np.array(self.X)[:,1:], np.array(self.X)[:,1:]
    def get_train_data(self, reps=10000):
        return np.tile(np.array(self.X)[:,1:], [reps, 1]), np.tile(np.array(self.X)[:,0], reps)




    # def get_train_data(self, num=100):
    #     if not self.q:
    #         self.q = Queue()
    #         p = Process(target=self.generate_train_data, args=(self.q,))
    #         p.start()
    #
    #     X, Y = None, None
    #     for i in range(num):
    #         print(f"Waiting for q. {i}")
    #         x,y = self.q.get()
    #         x = x[:,1:]
    #         if X is None:
    #             X = x
    #             Y = np.array(self.X)[:,1:]
    #         else:
    #             X = np.concatenate([X, x], axis=0)
    #             Y = np.concatenate([Y, np.array(self.X)[:,1:]], axis=0)
    #     return X, Y

    def get_train_data_st(self, num=100):
        X, Y = None, None
        for i in range(num):
            x,y = self.generate_train_data_st()
            x = x[:,1:]
            if X is None:
                X = x
                Y = np.array(self.X)[:,1:]
            else:
                X = np.concatenate([X, x], axis=0)
                Y = np.concatenate([Y, np.array(self.X)[:,1:]], axis=0)
        return X, Y


    # def get_train_data(self, num=10000):
    #     if not self.q:
    #         self.q = Queue()
    #         p = Process(target=self.generate_train_data, args=(self.q,))
    #         p.start()
    #
    #     X, Y = [], []
    #     for i in range(num):
    #         x,y = self.q.get()
    #         X.extend(x)
    #         Y.extend(y)
    #     return np.array(X),np.array(Y)


    # def generate_train_data(self, q):
    #     while True:
    #         X, Y = [], []
    #         for example in self.X:
    #             newitems = list(set(random.choices(range(177), weights=example[1:], k=random.randint(5, 7))))
    #             newitems = np.sum(to_categorical(newitems, nb_classes=ItemManager().get_num("int")), axis=0)
    #             X.append(np.concatenate([[example[0]], newitems], axis=0))
    #             Y.append(1)
    #             nonmatching_champs = list(set(range(ChampManager().get_num("int"))) - {example[0]})
    #             nc = random.choice(nonmatching_champs)
    #             X.append(np.concatenate([[nc], newitems], axis=0))
    #             Y.append(0)
    #         q.put([np.array(X), np.reshape(Y, (-1, 1))])

    def generate_train_data(self, q):
        while True:
            X, Y = [], []
            for example in self.X:
                newitems = list(set(random.choices(range(177), weights=example[1:], k=7)))
                newitems = np.sum(to_categorical(newitems, nb_classes=ItemManager().get_num("int")), axis=0)
                X.append(np.concatenate([[example[0]], newitems], axis=0))
                Y.append(1)
            q.put([np.array(X), np.reshape(Y, (-1, 1))])


    def generate_train_data_st(self):
            X, Y = [], []
            for example in self.X:
                newitems = list(set(random.choices(range(177), weights=example[1:], k=random.randint(5, 7))))
                newitems = np.sum(to_categorical(newitems, nb_classes=ItemManager().get_num("int")), axis=0)
                X.append(np.concatenate([[example[0]], newitems], axis=0))
                Y.append(1)
            return np.array(X), np.reshape(Y, (-1, 1))


    def eval_model(self, model, epoch, prior=None):
        X, Y = self.get_train_data(reps=1000)
        main_eval = model.evaluate(X, Y, batch_size=self.batch_size)[0]
        print(f"Test eval: {main_eval}")
        self.log_output(main_eval, epoch)
        return main_eval




    # def build_champ_embeddings_model(self):
    #
    #     self.network = ChampEmbeddings()
    #     self.train_path = app_constants.model_paths["train"]["next_items_starter"]
    #     self.best_path = app_constants.model_paths["best"]["next_items_starter"]
    #
    #     print("Loading training data")
    #     dataloader_elite = data_loader.SortedNextItemsDataLoader(app_constants.train_paths[
    #                                                            "next_items_processed_elite_sorted_complete"])
    #     # dataloader_lower = data_loader.SortedNextItemsDataLoader(app_constants.train_paths[
    #     #                                                              "next_items_processed_lower_sorted_complete"])
    #     champ_ints, items = dataloader_elite.get_item_distrib_by_champ()
    #     self.X = []
    #     self.Y = []
    #     for champ_int, item in zip(champ_ints, items):
    #         self.X.append(np.concatenate([[champ_int], item], axis=0))
    #
    #     self.build_new_model()


    def build_new_model(self):
        utils.remove_old_files(self.train_path)
        with open(self.train_path + self.acc_file_name, "w") as self.logfile:
            self.monitor_callback = MonitorCallbackRegression(self.logfile)
            scores = self.train_neural_network()
        best_model_index = self.determine_best_eval(scores)
        self.save_best_model(best_model_index)


    def build_champ_embeddings_model(self, distrib):
        self.train_path = app_constants.model_paths["train"]["next_items_starter"]
        self.best_path = app_constants.model_paths["best"]["next_items_starter"]
        print("Loading training data")

        # dataloader_elite = data_loader.SortedNextItemsDataLoader(app_constants.train_paths[
        #                                                              "next_items_processed_elite_sorted_complete"])
        #
        # champ_ints, items = dataloader_elite.get_item_distrib_by_champ()
        #
        # self.X = []
        # self.Y = []
        # for champ_int, item in zip(champ_ints, items):
        #     self.X.append(np.concatenate([[champ_int], item], axis=0))

        self.X = distrib
        # self.X = np.load("champ_item_distrib.npy")
        self.build_new_model()


    def train_neural_network(self):
        with tf.device("/gpu:0"):
            with tf.Graph().as_default():
                with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
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
                        self.eval_model(model, epoch)

        return scores

    def extract_embeddings(self, model, layer_name):
        # num_examples = 200
        # x, y = self.get_train_data_st(num=num_examples)
        # model.evaluate(x, y)

        x, y = self.get_train_data(1)

        feed_dict = tflearn.utils.feed_dict_builder(x, y, model.inputs,
                                                    model.targets)
        embeddings = model.predictor.evaluate(feed_dict, [layer_name], x.shape[0])[0]
        # tree = spatial.KDTree(embeddings)
        d = tree.query(embeddings, k=2)
        d = d[0][:,1]
        return embeddings, d




    def get_embedding_for_model(self, path, distrib, out_path, out_path_normed):
        self.X = distrib
        with tf.device("/gpu:0"):
            with tf.Graph().as_default():
                with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
                    tflearn.is_training(False)
                    self.network = self.network.build()
                    model = tflearn.DNN(self.network, session=sess)
                    sess.run(tf.global_variables_initializer())
                    model.load(path)
                    embs, dst = self.extract_embeddings(model, 'my_embedding/MatMul:0')
                    np.save(out_path, np.concatenate([embs, np.expand_dims(dst, axis=-1)], axis=1))
                    embs_normed = embs / np.expand_dims(np.linalg.norm(embs, axis=-1), axis=-1)
                    np.save(out_path_normed, embs_normed)




if __name__ == "__main__":
    # t = ChampsEmbeddingTrainer()
    # t.build_champ_embeddings_model(np.load("vs_champ_item_distrib.npy"))
    # t.get_embedding_for_model('models/best/next_items/starter/my_model17', np.load("vs_champ_item_distrib.npy"),
    #                           "opp_champ_embs_dst")


    t = FirstItemsTrainer()
    t.train()
    #
    # t.build_champ_embeddings_model()


    # try:
    #     t.build_next_items_early_game_model()
    # except Exception as e:
    #     print(e)
    # print("NOW TRAINING LATE GAME")
    # try:
    #     t.build_next_items_late_game_model()
    # except Exception as e:
    #     print(e)


    # t.standalone_eval()



    # s = PositionsTrainer()
    # s.train()

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
    #
    # s = ChampImgTrainer()
    # s.build_new_img_model()


    # from mpl_toolkits.mplot3d import Axes3D
    # import matplotlib.pyplot as plt
    #
    #
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # data = np.loadtxt("lulz")
    # data = data[:1000]
    #
    # #Fizz Sona Neeko Zilean Kayle
    #
    # ax.scatter(data[0::5, 0], data[0::5, 1], data[0::5, 2], c='red', marker='o')
    # # ax.scatter(data[1::5, 0], data[1::5, 1], data[1::5, 2], c='orange', marker='o')
    # ax.scatter(data[2::5, 0], data[2::5, 1], data[2::5, 2], c='blue', marker='o')
    # # ax.scatter(data[3::5, 0], data[3::5, 1], data[3::5, 2], c='purple', marker='o')
    # ax.scatter([-11.58291245, -12.81385612] ,  [1.67422855, 3.79949546] ,[-4.85478115,-4.1356039 ], c='yellow',
    #            marker='x')
    #
    # # [-11.58291245   1.67422855 - 4.85478115   2.55914012]
    # # [-12.81385612   3.79949546 - 4.1356039    2.37586738]
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    #
    # plt.show()




