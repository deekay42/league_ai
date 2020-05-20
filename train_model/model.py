import time
starttime = time.time()
import glob
import threading
from abc import ABC, abstractmethod
import cv2 as cv
import numpy as np
from constants import game_constants, app_constants, ui_constants
from utils.artifact_manager import ChampManager, ItemManager, SimpleManager
from utils import utils
import json
import itertools
from collections import Counter
import os
import platform
from sklearn import preprocessing
from tesserocr import PyTessBaseAPI
import pathlib
import ctypes
from numpy.ctypeslib import ndpointer
import importlib
import logging
import sys
logger = logging.getLogger("main")
from train_model.input_vector import Input

# if platform.system() == "Windows":
#     pytesseract.pytesseract.tesseract_cmd = os.path.abspath('Tesseract-OCR/tesseract.exe')
tflearn = None
tf = None
network = None

class Model(ABC):

    def __init__(self, dll_hook=None):
        self.output_node_name = None
        self.artifact_manager = None
        self.dll_hook = dll_hook
        if not dll_hook:
            global tf, tflearn, network
            tf = importlib.import_module("tensorflow")
            tflearn = importlib.import_module("tflearn")
            network = importlib.import_module("train_model.network")
            self.network = None
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            self.model = None
            self.model_path = None
            self.champ_embs = None
            self.opp_champ_embs = None


    def load_model(self):

        if self.dll_hook:
            return
        with self.graph.as_default():
            tflearn.is_training(False, session=self.session)
            
            self.network = self.network.build()
            model = tflearn.DNN(self.network, session=self.session)
            self.session.run(tf.global_variables_initializer())
            try:
                self.model_path = glob.glob(app_constants.model_paths["best"][self.elements] + "my_model*")[0]
                self.model_path = self.model_path.rpartition('.')[0]
                model.load(self.model_path, create_new_session=False)
                if self.champ_embs is not None:
                    embeddingWeights = tflearn.get_layer_variables_by_name('my_champ_embs')[0]
                    model.set_weights(embeddingWeights, self.champ_embs)
                    embeddingWeights = tflearn.get_layer_variables_by_name('opp_champ_embs')[0]
                    model.set_weights(embeddingWeights, self.opp_champ_embs)
            except Exception as e:
                print("Unable to open best model files")
                raise e
            self.model = model


    def export(self):
        from tensorflow.python.framework import graph_util
        with self.graph.as_default():
            tflearn.is_training(False, session=self.session)
        gd = self.session.graph.as_graph_def()
        for node in gd.node:
             if node.op == 'RefSwitch':
                   node.op = 'Switch'
                   for index in range(len(node.input)):
                           if 'moving_' in node.input[index]:
                                 node.input[index] = node.input[index] + '/read'
             elif node.op == 'AssignSub':
                   node.op = 'Sub'
                   if 'use_locking' in node.attr: del node.attr['use_locking']
             elif node.op == 'AssignAdd':
                   node.op = 'Add'
                   if 'use_locking' in node.attr: del node.attr['use_locking']
             elif node.op == 'Assign':
                   node.op = 'Identity'
                   if 'use_locking' in node.attr: del node.attr['use_locking']
                   if 'validate_shape' in node.attr: del node.attr['validate_shape']
                   if len(node.input) == 2:
                           # input0: ref: Should be from a Variable node. May be uninitialized.
                           # input1: value: The value to be assigned to the variable.
                           node.input[0] = node.input[1]
                           del node.input[1]


        converted_graph_def = graph_util.convert_variables_to_constants(self.session, gd, [self.output_node_name])
        tf.train.write_graph(converted_graph_def, app_constants.model_paths["best"][self.elements], "model.pb", as_text=False)


        # gd = self.session.graph.as_graph_def()
        # for node in gd.node:
        #     if node.op == 'RefSwitch':
        #         node.op = 'Switch'
        #         for index in range(len(node.input)):
        #             if 'moving_' in node.input[index]:
        #                 node.input[index] = node.input[index] + '/read'
        #     elif node.op == 'AssignSub':
        #         node.op = 'Sub'
        #         if 'use_locking' in node.attr: 
        #             del node.attr['use_locking']
        #     elif node.op == 'AssignAdd':
        #         node.op = 'Add'
        #         if 'use_locking' in node.attr: 
        #             del node.attr['use_locking']
        #     elif node.op == 'Assign':
        #         node.op = 'Identity'
        #         if 'use_locking' in node.attr: 
        #             del node.attr['use_locking']
        #         if 'validate_shape' in node.attr: 
        #             del node.attr['validate_shape']
        #         if len(node.input) == 2:
        #             node.input[0] = node.input[1]
        #             del node.input[1]
        # converted_graph_def = graph_util.convert_variables_to_constants(self.session, gd, [self.output_node_name])
        # tf.train.write_graph(converted_graph_def, app_constants.model_paths["best"][self.elements], "model.pb", as_text=False)


    # def output_logs(self, in_vec):
    #     sess = tf.InteractiveSession()
    #     sess.run(tf.global_variables_initializer())
    #     tflearn.is_training(True, session=sess)
    #     is_training = tflearn.get_training_mode()
    #     game_config = \
    #         {
    #             "champs_per_game": game_constants.CHAMPS_PER_GAME,
    #             "champs_per_team": game_constants.CHAMPS_PER_TEAM,
    #             "total_num_champs": ChampManager().get_num("int"),
    #             "total_num_items": ItemManager().get_num("int"),
    #             "items_per_champ": game_constants.MAX_ITEMS_PER_CHAMP
    #         }

    #     network_config = \
    #         {
    #             "learning_rate": 0.00025,
    #             "champ_emb_dim": 3,
    #             "all_items_emb_dim": 6,
    #             "champ_all_items_emb_dim": 8,
    #             "class_weights": [1]
    #         }
    #     my_champ_embs_dst = np.load("my_champ_embs_dst.npy")
    #     opp_champ_embs_dst = np.load("opp_champ_embs_dst.npy")
    #     my_champ_embs_dst = np.concatenate([[[0, 0, 0, 0]], my_champ_embs_dst], axis=0)

    #     self.champ_embs = my_champ_embs_dst[:, :3]
    #     self.opp_champ_embs = opp_champ_embs_dst[:, :3]

    #     network_config["my_champ_emb_scales"] = (np.repeat(my_champ_embs_dst[:, -1], network_config[
    #         "champ_emb_dim"])/2).astype(np.float32)
    #     network_config["opp_champ_emb_scales"] = (np.repeat(opp_champ_embs_dst[:, -1], network_config[
    #         "champ_emb_dim"])/2).astype(np.float32)

    #     champs_per_game = game_config["champs_per_game"]
    #     total_num_champs = game_config["total_num_champs"]
    #     total_num_items = game_config["total_num_items"]
    #     items_per_champ = game_config["items_per_champ"]
    #     champs_per_team = game_config["champs_per_team"]

    #     learning_rate = network_config["learning_rate"]
    #     champ_emb_dim = network_config["champ_emb_dim"]

    #     all_items_emb_dim = network_config["all_items_emb_dim"]
    #     champ_all_items_emb_dim = network_config["champ_all_items_emb_dim"]

    #     total_champ_dim = champs_per_game
    #     total_item_dim = champs_per_game * items_per_champ

    #     pos_start = 0
    #     pos_end = pos_start + 1
    #     champs_start = pos_end
    #     champs_end = champs_start + champs_per_game
    #     items_start = champs_end
    #     items_end = items_start + items_per_champ * 2 * champs_per_game
    #     total_gold_start = items_end
    #     total_gold_end = total_gold_start + champs_per_game
    #     cs_start = total_gold_end
    #     cs_end = cs_start + champs_per_game
    #     neutral_cs_start = cs_end
    #     neutral_cs_end = neutral_cs_start + champs_per_game
    #     xp_start = neutral_cs_end
    #     xp_end = xp_start + champs_per_game
    #     lvl_start = xp_end
    #     lvl_end = lvl_start + champs_per_game
    #     kda_start = lvl_end
    #     kda_end = kda_start + champs_per_game * 3
    #     current_gold_start = kda_end
    #     current_gold_end = current_gold_start + champs_per_game

    #     # in_vec = input_data(shape=[None, 221], name='input')
    #     #  1 elements long
    #     pos = in_vec[:, 0]
    #     pos = tf.cast(pos, tf.int32)

    #     n = tf.shape(in_vec)[0]
    #     batch_index = tf.range(n)
    #     pos_index = tf.transpose([batch_index, pos], (1, 0))
    #     opp_index = tf.transpose([batch_index, pos + champs_per_team], (1, 0))


    #     item_ints = in_vec[:, items_start:items_end]

    #     champ_ints = tf.cast(in_vec[:, champs_start:champs_end], tf.int32)
    #     my_team_champ_ints = champ_ints[:, :5]
    #     opp_team_champ_ints = champ_ints[:, 5:]

    #     my_team_champs_embedded = tf.cast(tf.gather(self.champ_embs, my_team_champ_ints), tf.float32)
    #     my_team_emb_noise_dist = tf.distributions.Normal(loc=[0.] * (total_num_champs) * champ_emb_dim,
    #                                                      scale=network_config["my_champ_emb_scales"])
    #     my_team_emb_noise = my_team_emb_noise_dist.sample([1])
    #     my_team_emb_noise = tf.reshape(my_team_emb_noise, (-1, 3))
    #     my_team_champs_embedded_noise = tf.cast(tf.gather(my_team_emb_noise,
    #                                               tf.cast(my_team_champ_ints, tf.int32)), tf.float32)
    #     my_team_champs_embedded_noised = tf.cond(is_training, lambda: my_team_champs_embedded +
    #                                                              my_team_champs_embedded_noise,
    #                                       lambda: my_team_champs_embedded)




    #     net = batch_normalization(fully_connected(my_team_champ_ints, 16, bias=False,
    #                                               activation='relu',
    #                                               regularizer="L2"))
    #     logits = fully_connected(net, total_num_items, activation='linear')


    #     inference_output = tf.nn.softmax(logits)

    #     net = tf.cond(is_training, lambda: logits, lambda: inference_output)

    #     output = regression(net, optimizer='adam', to_one_hot=True,
    #                       n_classes=total_num_items,
    #                       shuffle_batches=True,
    #                       learning_rate=learning_rate,
    #                       loss='softmax_categorical_crossentropy',
    #                       name='target')






    def predict2int(self, x):
        if self.dll_hook:
            flat_x = np.ravel(x)
            y = self.dll_hook.predict(flat_x, self.elements, (x.shape[0], self.artifact_manager.get_num("img_int")))
        else:     
            with self.graph.as_default():
                y = self.model.predict(x)
        y_int = np.argmax(y, axis=len(y.shape) - 1)
        return y_int


    @abstractmethod
    def predict(self, x):
        pass


class CPredict:
    def __init__(self):
        models = [NextItemModel("standard", dll_hook=self),
            NextItemModel("late", dll_hook=self),
            NextItemModel("starter", dll_hook=self),
            NextItemModel("first_item", dll_hook=self), 
            NextItemModel("boots", dll_hook=self), 
            ChampImgModel(dll_hook=self),
            ItemImgModel(dll_hook=self), 
            SelfImgModel(dll_hook=self), 
            KDAImgModel(ui_constants.ResConverter(1024,768), dll_hook=self)]
        model_ids = [model.elements for model in models]
        model_paths = [app_constants.model_paths["best"][model]+"model.pb" for model in model_ids]
        model_output_nodes = [model.output_node_name for model in models]
        libname = pathlib.Path().absolute() / "cpredict"

        self.cpredict = ctypes.CDLL(str(libname))
        self.cpredict.initialize(utils.strlist2cstrlist(model_paths), utils.strlist2cstrlist(model_ids), utils.strlist2cstrlist(model_output_nodes), len(models))


    def predict(self, x, model_id, dims):
        result_len = 1
        for i in dims:
            result_len *= i
        self.cpredict.predict.restype = ndpointer(dtype=ctypes.c_float, shape=(result_len,))
        result = self.cpredict.predict((ctypes.c_float * len(x))(*x), len(x), utils.str2cstr(model_id))
        return np.reshape(result, dims)
        

class ImgModel(Model):

    def __init__(self, res_converter=None, dll_hook=None):
        super().__init__(dll_hook)
        self.output_node_name = "FullyConnected_1/Softmax"
        if res_converter:
            self.network_crop = res_converter.network_crop[self.elements]
            self.res_converter = res_converter


    def predict(self, img):
        x = self.extract_imgs(img)
        img_ints = self.predict2int(np.array(x))
        predicted_artifacts = (self.artifact_manager.lookup_by("img_int", artifact_int) for artifact_int in img_ints)
        return predicted_artifacts


    def get_coords(self):
        return list(self.res_converter.generate_std_coords(self.elements))


    def extract_imgs(self, whole_img):
        coords = self.get_coords()
        coords = np.reshape(coords, (-1, 2))
        # from utils import utils
        # utils.show_coords(whole_img, coords, self.res_converter.lookup(self.elements, "x_crop"),
        #                   self.res_converter.lookup(self.elements, "y_crop"))

        sub_imgs = [whole_img[int(round(coord[1])):int(round(coord[1] + self.res_converter.lookup(self.elements,
                                                                                                  "y_crop"))),
                    int(round(coord[0])):int(round(coord[0] + self.res_converter.lookup(self.elements, "x_crop")))]
                    for coord in
                    coords]
        sub_imgs = [cv.resize(img, self.network_crop, cv.INTER_AREA) for img in sub_imgs]
        # for i, img in enumerate(sub_imgs):
        #     cv.imshow(str(i), img)
        # cv.waitKey(0)
        return np.array(sub_imgs)


class MultiTesseractModel:
    def __init__(self, tesseractmodels):
        self.tess = PyTessBaseAPI(path=app_constants.tess_path, lang='eng', psm=6, oem=1)
        self.tess.SetVariable("tessedit_char_whitelist", "@0123456789")
        # self.tess.SetVariable("classify_bln_numeric_mode", "1")

        self.tesseractmodels = tesseractmodels
    

    def __del__(self): 
        if self.tess:
            self.tess.End()


    def predict(self, whole_img):
        slide_imgs = []
        for model in self.tesseractmodels:
            slide_imgs.extend(model.extract_all_slide_imgs(whole_img))
        y_heights = [slide_img.shape[0] for slide_img in slide_imgs]

        y_height = Counter(y_heights).most_common(1)[0][0]


        slide_imgs = [cv.resize(slide_img, None, fx=y_height/slide_img.shape[0], fy=y_height/slide_img.shape[
            0],
                                interpolation=cv.INTER_CUBIC) for slide_img in slide_imgs]

        y_widths = [slide_img.shape[1] for slide_img in slide_imgs]
        y_longest_width = max(y_widths)
        slide_imgs_vert = [cv.copyMakeBorder(slide_img, 0, 5, 0, y_longest_width - slide_img.shape[1], cv.BORDER_CONSTANT,
                                             value=(255, 255, 255)) for slide_img in slide_imgs]
        img_vert = np.concatenate(slide_imgs_vert, axis=0)
        img = cv.copyMakeBorder(img_vert, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=(255, 255, 255))
        # cv.imshow("f", img)
        # cv.waitKey(0)
        # img = np.concatenate(slide_imgs, axis=1)
        # img = cv.copyMakeBorder(img, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=(255, 255, 255))
        self.tess.SetImageBytes(np.ravel(img).tostring(), *img.shape[::-1], 1, 1 * img.shape[1])
        text = self.tess.GetUTF8Text()
        # result = text[10:]
        # print(text)
        str_result = [s[10:] for s in text.split("\n")][:-1]
        for res in str_result:
            yield model.convert(res)


class TesseractModel:

    def __init__(self, res_converter):
        self.res_converter = res_converter
        sep_imgs = [cv.imread(app_constants.asset_paths["kda"]+str(i)+".png", cv.IMREAD_GRAYSCALE) for i in range(10)]
        sep_img = np.concatenate(sep_imgs, axis=1)
        sep_img = sep_img[2:-2]
        sep_img = cv.bitwise_not(sep_img)
        sep_img = cv.resize(sep_img, None, fx=5, fy=5, interpolation=cv.INTER_CUBIC)

        # self.separator = cv.imread( app_constants.asset_paths["tesseract_separator"], cv.IMREAD_GRAYSCALE)
        # self.left_separator = cv.copyMakeBorder(self.separator, 0, 0, 0, 0, cv.BORDER_CONSTANT, value=(255, 255, 255))
        # self.right_separator = cv.copyMakeBorder(self.separator, 0, 0, 0, 0, cv.BORDER_CONSTANT, value=(255, 255, 255))
        self.left_separator = sep_img
        self.right_separator = sep_img
        kernel = np.ones((2, 2 ), np.uint8)


        # self.left_separator = cv.erode(self.left_separator,kernel,iterations = 1)
        # self.right_separator = cv.erode(self.right_separator, kernel, iterations=1)
        _, self.right_separator = cv.threshold(self.right_separator, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        _, self.left_separator = cv.threshold(self.left_separator, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)


    def extract_slide_img(self, slide_img):

        # x_pad = y_pad = 20
        # img_bordered = cv.copyMakeBorder(slide_img, x_pad, x_pad, y_pad, y_pad, cv.BORDER_CONSTANT, value=(0, 0, 0))
        scale_factor = 3
        img_bordered = cv.resize(slide_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_CUBIC)
        gray = cv.cvtColor(img_bordered, cv.COLOR_BGR2GRAY)
        # cv.imshow("gray", gray)

        ret, thresholded = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # cv.imshow("thresholded", thresholded)
        # cutoff_ratio = 2/5
        # sorted_thresholds = sorted(gray[thresholded > 0])
        # thresh = sorted_thresholds[int(len(sorted_thresholds) * cutoff_ratio)]
        # gray[gray < thresh] = 0
        # cv.imshow("gray_thresholded", gray)
        contours, hier = cv.findContours(thresholded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        bboxes = [cv.boundingRect(contour) for contour in contours]
        bboxes = np.array([[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]] for bbox in bboxes])
        # sorted_bboxes = np.array(sorted([cv.boundingRect(contour) for contour in contours], key=lambda a: a[0]))
        # x_l, y_l, w_l, h_l = sorted_bboxes[0]
        # x_r, y_r, w_r, h_r = sorted_bboxes[-1]
        x_left = np.min(bboxes[:, 0])
        x_right = np.max(bboxes[:, 2])
        y_top = np.min(bboxes[:, 1])
        y_bot = np.max(bboxes[:, 3])
        ratio = (y_bot - y_top) / self.left_separator.shape[0]
        left_separator = cv.resize(self.left_separator, None, fx=ratio, fy=ratio, interpolation=cv.INTER_AREA)
        right_separator = cv.resize(self.right_separator, None, fx=ratio, fy=ratio, interpolation=cv.INTER_AREA)
        blob_contour = gray[y_top:y_bot, x_left:x_right]
        ret, thresholded = cv.threshold(blob_contour, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        inv = cv.bitwise_not(thresholded)
        # inv = cv.copyMakeBorder(inv, 0, left_separator.shape[0]-inv.shape[0], 0, 0, cv.BORDER_CONSTANT,
        #                         value=(255, 255, 255))
        border = np.ones(shape=(left_separator.shape[0], 3), dtype=np.uint8)*255
        img = np.concatenate([left_separator, border, inv, border],
                             axis=1)

        # img = cv.copyMakeBorder(img, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=(255, 255, 255))
        _, img_binarized = cv.threshold(img, 180, 255, cv.THRESH_BINARY)

        # cv.waitKey(0)
        return img_binarized


    def get_coords(self):
        return list(self.res_converter.generate_std_coords(self.elements))


    def get_raw_slide_imgs(self, whole_img):
        coords = self.get_coords()
        coords = np.reshape(coords, (-1, 2))
        # from utils import utils
        # utils.show_coords(whole_img, coords, self.res_converter.lookup(self.elements, "x_width"),self.res_converter.lookup(self.elements, "y_height"))
        slide_imgs = [
            whole_img[int(round(coord[1])):int(round(coord[1] + self.res_converter.lookup(self.elements, "y_height"))),
            int(round(coord[0])):int(round(coord[0] + self.res_converter.lookup(self.elements, "x_width")))]
            for coord in coords]
        return slide_imgs


    def extract_all_slide_imgs(self, whole_img):

        slide_imgs = self.get_raw_slide_imgs(whole_img)
        # for img in slide_imgs:
        #     cv.imshow("f", img)
        #     cv.waitKey(0)
        result = [self.extract_slide_img(slide_img) for slide_img in slide_imgs]
        # for img in result:
        #     cv.imshow("f", img)
        #     cv.waitKey(0)
        return result


    def convert(self, tesseract_result):
        try:
            return int(tesseract_result)
        except ValueError as e:
            return -1


class CSImgModel(TesseractModel):

    def __init__(self, res_converter):
        self.elements = "cs"
        super().__init__(res_converter)


class LvlImgModel(TesseractModel):

    def __init__(self, res_converter):
        self.elements = "lvl"
        super().__init__(res_converter)


    def extract_slide_img(self, slide_img):
        bw_pixels = np.logical_and(slide_img[:,:,0] == slide_img[:,:,1], slide_img[:,:,1] == slide_img[:,:,
                                                                                             2]).astype(np.uint8)
        slide_img = cv.bitwise_and(slide_img, slide_img, mask=bw_pixels)
        return super().extract_slide_img(slide_img)


class KDAImgModel(ImgModel):

    def __init__(self, res_converter=None, dll_hook=None):
        self.elements = "kda"
        super().__init__(res_converter, dll_hook)
        self.artifact_manager = SimpleManager(self.elements)
        if dll_hook:
            return
        self.network = network.DigitRecognitionNetwork(lambda: self.artifact_manager.get_num("img_int"),
                                                       self.network_crop)
        self.model_path = app_constants.model_paths["best"][self.elements]



    def predict(self, whole_img):
        imgs, num_elements = self.extract_imgs(whole_img)
        img_ints = self.predict2int(np.array(imgs))
        predicted_artifacts = [self.artifact_manager.lookup_by("img_int", artifact_int)["name"] for artifact_int in
                               img_ints]
        index = 0
        for number in num_elements:
            kda_section = predicted_artifacts[index:index+number]
            kda_string = "".join(kda_section)
            kda_string = kda_string.split("slash")
            if len(kda_string) == 3:
                try:
                    kills = int(kda_string[0])
                    deaths = int(kda_string[1])
                    assists = int(kda_string[2])
                    yield [kills, deaths, assists]
                except ValueError as e:
                    yield [-1,-1,-1]
            else:
                yield [-1,-1,-1]
            index += number


    def break_up_into_sub_imgs(self, slide_img):

        x_pad = y_pad = 20
        img_bordered = cv.copyMakeBorder(slide_img, x_pad, x_pad, y_pad, y_pad, cv.BORDER_CONSTANT, value=(0, 0, 0))
        scale_factor = 5
        img_bordered = cv.resize(img_bordered, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_CUBIC)

        gray = cv.cvtColor(img_bordered, cv.COLOR_BGR2GRAY)


        cutoff_ratio = 2/5
        ret, thresholded = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        sorted_thresholds = sorted(gray[thresholded > 0])
        thresh = sorted_thresholds[int(len(sorted_thresholds) * cutoff_ratio)]
        gray[gray < thresh] = 0


        contours, hier = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        sorted_bboxes = sorted([cv.boundingRect(contour) for contour in contours], key=lambda a: a[0])

        for x,y,w,h in sorted_bboxes:
            # get the bounding rect
            # x += x_l
            # y += y_top

            # cv.rectangle(img_bordered, (x, y), (x + w, y + h), (0, 0, 255), 1)

            center_x = x + w / 2
            center_y = y + h / 2
            crop_start_x = round(center_x - scale_factor * self.res_converter.lookup(self.elements, "x_crop") / 2)
            crop_start_y = round(center_y - scale_factor * self.res_converter.lookup(self.elements, "y_crop") / 2)
            if crop_start_x < 0 or crop_start_y < 0:
                print("less than 0")

            y_height = scale_factor * self.res_converter.lookup(self.elements, "y_crop")
            x_width = scale_factor * self.res_converter.lookup(self.elements, "x_crop")
            sub_image = img_bordered[crop_start_y:int(round(crop_start_y + y_height)),
                        crop_start_x:int(round(crop_start_x + x_width))]

            # cv.rectangle(img_bordered, (crop_start_x, crop_start_y), (int(round(crop_start_x + x_width)), int(round(crop_start_y +
            #                                                           y_height))), (0, 255,0), 1)
            yield sub_image
        # cv.imshow("f", img_bordered)
        # cv.waitKey(0)


    def extract_imgs(self, whole_img):
        coords = list(self.get_coords())
        coords = np.reshape(coords, (-1, 2))
        # from utils import utils
        # utils.show_coords(whole_img, coords, self.res_converter.lookup(self.elements, "x_width"),self.res_converter.lookup(self.elements, "y_height"))

        slide_imgs = [
            whole_img[coord[1]:int(round(coord[1] + self.res_converter.lookup(self.elements, "y_height"))),
            coord[0]:int(round(coord[0] + self.res_converter.lookup(self.elements, "x_width")))]
            for coord in coords]

        sub_imgs = []
        num_elements = []
        for img in slide_imgs:
            next_sub_imgs = list(self.break_up_into_sub_imgs(img))
            num_elements.append(len(next_sub_imgs))
            sub_imgs.extend(next_sub_imgs)

        sub_imgs = [cv.resize(img, (self.res_converter.network_crop[self.elements][1],
                                         self.res_converter.network_crop[self.elements][0]), cv.INTER_AREA) for img in
                         sub_imgs]

        return sub_imgs, num_elements


class CurrentGoldImgModel(TesseractModel):

    def __init__(self, res_converter):
        self.elements = "current_gold"
        super().__init__(res_converter)


    def get_raw_slide_imgs(self, whole_img):
        x,y,w,h = self.res_converter.generate_current_gold_coords()
        slide_imgs = [whole_img[int(round(y)):int(round(y + h)), int(round(x)):int(round(x+w))]]
        return slide_imgs


class ChampImgModel(ImgModel):

    def __init__(self, res_converter=None, dll_hook=None):
        self.elements = "champs"
        super().__init__(res_converter, dll_hook)
        
        self.artifact_manager = ChampManager()
        if dll_hook:
            return
        self.network = network.ChampImgNetwork()



class ItemImgModel(ImgModel):

    def __init__(self, res_converter=None, dll_hook=None):
        self.elements = "items"
        super().__init__(res_converter, dll_hook)
        self.artifact_manager = ItemManager()
        if dll_hook:
            return
        self.network = network.ItemImgNetwork()


    def get_coords(self):
        return list(self.res_converter.generate_item_coords())


class SelfImgModel(ImgModel):

    def __init__(self, res_converter=None, dll_hook=None):
        self.elements = "self"
        super().__init__(res_converter, dll_hook)
        self.output_node_name = "FullyConnected/Sigmoid"
        self.artifact_manager = SimpleManager(self.elements)
        if dll_hook:
            return
        self.network = network.SelfImgNetwork()


    def predict(self, img):
        x = self.extract_imgs(img)
        if self.dll_hook:
            flat_x = np.ravel(x)
            y = self.dll_hook.predict(flat_x, self.elements, (x.shape[0],))
        else:
            with self.graph.as_default():
                y = self.model.predict(x)
        role_index = np.argmax(y)
        return role_index


class NextItemModel(Model):

    def __init__(self, early_or_late, dll_hook=None):
        super().__init__(dll_hook)


        self.early_or_late = early_or_late
        if early_or_late == "standard":
            self.model_path = app_constants.model_paths["best"]["next_items_standard"]
            self.elements = "next_items_standard"
        elif early_or_late == "late":
            self.model_path = app_constants.model_paths["best"]["next_items_late"]
            self.elements = "next_items_late"
        elif early_or_late == "starter":
            self.model_path = app_constants.model_paths["best"]["next_items_starter"]
            self.elements = "next_items_starter"
        elif early_or_late == "first_item":
            self.model_path = app_constants.model_paths["best"]["next_items_first_item"]
            self.elements = "next_items_first_item"
        elif early_or_late == "boots":
            self.model_path = app_constants.model_paths["best"]["next_items_boots"]
            self.elements = "next_items_boots"

        self.artifact_manager = ItemManager()

        # self.load_model()

        # with open(glob.glob('models/best/next_items/early/thresholds*')[0]) as f:
        #     self.thresholds = json.load(f)
        self.thresholds = 1

        champs_per_game = game_constants.CHAMPS_PER_GAME
        items_per_champ = game_constants.MAX_ITEMS_PER_CHAMP


        self.cont_slices_by_name = {'total_gold': np.s_[:, Input.total_gold_start:Input.total_gold_end],
                                    'cs': np.s_[:, Input.cs_start:Input.cs_end],
                                    'neutral_cs': np.s_[:, Input.neutral_cs_start:Input.neutral_cs_end],
                                    'xp': np.s_[:, Input.xp_start:Input.xp_end],
                                    'lvl': np.s_[:, Input.lvl_start:Input.lvl_end],
                                    'kda': np.s_[:, Input.kda_start:Input.kda_end],
                                    'cg': np.s_[:, Input.current_gold_start:Input.current_gold_end],
                                    'turrets': np.s_[:, Input.turrets_start:Input.turrets_end], }
        
        self.input_len = Input.len
        
        self.output_node_name = "Softmax"
        
        if dll_hook:
            return

        if early_or_late == "standard":
            self.network = network.StandardNextItemNetwork()
        elif early_or_late == "late":
            self.network = network.NextItemLateGameNetwork()
        elif early_or_late == "starter":
            self.network = network.NextItemStarterNetwork()
        elif early_or_late == "first_item":
            self.network = network.NextItemFirstItemNetwork()
        elif early_or_late == "boots":
            self.network = network.NextItemBootsNetwork()
        

        my_champ_embs_normed = np.load(app_constants.asset_paths["my_champs_embeddings"])
        opp_champ_embs_normed = np.load(app_constants.asset_paths["opp_champs_embeddings"])
        my_champ_embs_normed = np.concatenate([[[0, 0, 0]], my_champ_embs_normed], axis=0)
        opp_champ_embs_normed = np.concatenate([[[0, 0, 0]], opp_champ_embs_normed], axis=0)

        self.champ_embs = my_champ_embs_normed
        self.opp_champ_embs = opp_champ_embs_normed


        


    def predict_easy(self, role, champs_int, items_id, cs, lvl, kda, current_gold, blackout_indices):
        x = np.zeros(shape=self.input_len, dtype=np.int32)
        x[Input.pos_start:Input.pos_end] = [role]
        x[Input.champs_start:Input.champs_end] = champs_int
        encoded_items = np.ravel(self.encode_items(items_id, self.artifact_manager))
        x[Input.items_start:Input.items_end] = encoded_items
        x[Input.cs_start:Input.cs_end] = cs
        x[Input.lvl_start:Input.lvl_end] = lvl
        x[Input.kda_start:Input.kda_end] = np.ravel(kda)
        num_increments = 20
        granularity = 10
        start = -150
        zero_offset = -start // granularity
        current_gold_list = np.zeros((num_increments,10))
        current_gold_list[:,role] = np.array([current_gold]*num_increments) + np.array(range(start,
                                                                                             start+num_increments*granularity,
                                                                                             granularity))
 
        logger.info(current_gold_list[:,role])
        x = np.tile(x,num_increments).reshape((num_increments,-1))
        x[:, Input.current_gold_start:Input.current_gold_end] = current_gold_list
        x = self.scale_inputs(np.array(x).astype(np.float32))
        lul = False
        if lul:
            self.output_logs(x)
        result, probabilities = self.predict(x, blackout_indices)
        return result[zero_offset], result, probabilities[zero_offset]


    def scale_inputs(self, X):
        for slice_name in self.cont_slices_by_name:
            slice = self.cont_slices_by_name[slice_name]
            if slice_name == 'cs' or slice_name == 'neutral_cs':
                scaler = self.fit_input(np.array([[0.0, 300.0]]), slice_name)
            elif slice_name == 'lvl':
                scaler = self.fit_input(np.array([[0.0, 18.0]]), slice_name)
            elif slice_name == 'kda':
                scaler = self.fit_input(np.array([[0.0, 15]]), slice_name)
            elif slice_name == 'cg':
                scaler = self.fit_input(np.array([[0.0, 2000.0]]), slice_name)
            elif slice_name == 'total_gold':
                scaler = self.fit_input(np.array([[500.0, 50000.0]]), slice_name)
            elif slice_name == 'xp':
                scaler = self.fit_input(np.array([[0.0, 50000.0]]), slice_name)
            elif slice_name == 'turrets':
                scaler = self.fit_input(np.array([[0.0, 11.0]]), slice_name)
            else:
                print("WTFFFFFFFF")
            X[slice] = scaler.transform(X[slice])
        return X


    def fit_input(self, X, scaler_name):
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        min_max_scaler.fit(np.reshape(X, (-1, 1)))
        return min_max_scaler


    @staticmethod
    def encode_items(items, artifact_manager):
        items_at_time_x = []
        for player_items in items:
            player_items_dict = Counter(player_items)
            player_items_dict_items = []
            processed_player_items = []
            for item in player_items_dict:
                # these items can fit multiple instances into one item slot
                if item == 2055 or item == 2003:
                    added_item = [artifact_manager.lookup_by('id', str(item))['int'],
                                  player_items_dict[
                                      item]]
                    processed_player_items.append(added_item)
                    player_items_dict_items.append(added_item)
                elif item == 2138 or item == 2139 or item == 2140:
                    continue
                else:
                    added_item = artifact_manager.lookup_by('id', str(item))['int']
                    processed_player_items.extend([[added_item, 1]] * player_items_dict[item])
                    player_items_dict_items.append((added_item, player_items_dict[item]))

            if processed_player_items == []:
                processed_player_items = [[0, 6], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]]
            else:
                empties_length = game_constants.MAX_ITEMS_PER_CHAMP - len(processed_player_items)
                padding_length = game_constants.MAX_ITEMS_PER_CHAMP - len(player_items_dict_items)

                try:
                    if empties_length < 0:
                        raise ValueError()

                    if padding_length == 0:
                        empties = np.array([]).reshape((0, 2)).astype(int)
                        padding = np.array([]).reshape((0, 2)).astype(int)
                    if padding_length == 1:
                        empties = [[0, empties_length]]
                        padding = np.array([]).reshape((0, 2)).astype(int)
                    elif padding_length > 1:
                        empties = [[0, empties_length]]
                        padding = [[-1, -1]] * (padding_length - 1)


                except ValueError as e:
                    raise e

                processed_player_items = np.concatenate([player_items_dict_items, empties, padding],
                                                        axis=0).tolist()

            items_at_time_x.append(processed_player_items)

        return np.array(items_at_time_x)


    def predict(self, x, blackout_indices=None):
        # with self.graph.as_default(), tf.Session() as sess:
        #     tflearn.is_training(False, session=sess)
        #     X = tf.placeholder("float", [None, 71])
        #     # X = input_data(shape=[None, 71], name='lolol')
        #     log = self.output_logs(X)
        #     sess.run(tf.global_variables_initializer())
        #     log = sess.run(log, feed_dict = {X: np.array(x)})
        #     for i, _ in enumerate(log):
        #         print(f"{i}: {log[i]}")
        # x = [[3,1,73,142,38,130,110,6,123,139,127,42,0,0,0,0,0,15,41,0,0,0,0,42,0,0,0,0,0,37,23,12,2,0,0,151,0,0,0,0,
        #       0,23,37,0,0,0,0,15,41,0,0,0,0,3,3,3,37,0,0,23,0,0,0,0,0,150,0,0,0,0,0]]
        # self.output_logs(x)

        # np.savetxt('lolfile.out', np.ravel(x[14]), delimiter=',', fmt='%1.4f')
        if self.dll_hook:
            flat_x = np.ravel(x).astype(np.float)
            y = self.dll_hook.predict(flat_x, self.elements, (x.shape[0], self.artifact_manager.get_num("int")))
        else:
            with self.graph.as_default():
                y = self.model.predict(x)
            # print(y[14])
        if blackout_indices:
            y[:, blackout_indices] = 0
        item_ints = np.argmax(y, axis=len(y.shape) - 1)
        logger.info(f"Confidence: {np.max(y, axis=1)}")
        items = [self.artifact_manager.lookup_by("int", item_int) for item_int in item_ints]
        logger.info([item["name"] for item in items])
        logger.info("\n\n")
        return items, np.max(y, axis=1)


class PositionsModel(Model):

    def __init__(self):
        super().__init__()
        self.network = network.PositionsNetwork()
        self.model_path = app_constants.model_paths["best"]["positions"]
        keys = [(1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 1, 0), (0, 0, 0, 0, 1)]
        self.roles = dict(zip(keys, game_constants.ROLE_ORDER))
        self.permutations = dict(zip(keys, [0, 1, 2, 3, 4]))
        self.champ_manager = ChampManager()
        self.spell_manager = SimpleManager("spells")
        self.elements = "positions"
        self.load_model()
        self.lock = threading.Lock()


    def predict(self, x):
        with self.lock:
            with self.graph.as_default():
                pred = self.model.predict([x])
                with tf.Session() as sess:
                    final_pred = network.PositionsNetwork.best_permutations_one_hot(pred)
                    final_pred = sess.run(final_pred)[0]
            champ_roles = [self.roles[tuple(role)] for role in final_pred]

            # the champ ids need to be ints, otherwise jq fails
            champ_ids = [int(self.champ_manager.lookup_by("int", champ_int)["id"]) for champ_int in x[
                                                                                                    :game_constants.CHAMPS_PER_TEAM]]
            return dict(zip(champ_roles, champ_ids))


    def multi_predict_perm(self, x):
        with self.lock:
            with self.graph.as_default():
                with tf.Session() as sess:
                    x = network.PositionsNetwork.permutate_inputs(x)

                    chunk_len = 1000
                    x = tf.reshape(x, (-1,120,55))
                    i = 0
                    final_pred = []
                    while i < int(x.shape[0]):
                        print(i/int(x.shape[0]))
                        next_chunk = x[i:i+chunk_len]
                        next_chunk = tf.reshape(next_chunk, (-1,55))
                        chunk_pred = self.model.predict(sess.run(next_chunk))
                        i += chunk_len
                        best_perms = network.PositionsNetwork.select_best_input_perm(np.array(chunk_pred))
                        final_pred.extend(sess.run(best_perms).tolist())

        result = []
        for sorted_team in final_pred:
            sorted_team_perm = [0] * 5
            for i, pos in enumerate(sorted_team):
                sorted_team_perm[self.permutations[tuple(pos)]] = i
            result.append(sorted_team_perm)
        return result



    def multi_predict(self, x):
        with self.lock:
            with self.graph.as_default():
                pred = self.model.predict(x)
                with Session() as sess:
                    final_pred = network.PositionsNetwork.best_permutations_one_hot(pred)
                    final_pred = sess.run(final_pred)
        result = []
        for sorted_team, unsorted_team in zip(final_pred, x):
            sorted_team_perm = [0] * 5
            for i, pos in enumerate(sorted_team):
                sorted_team_perm[self.permutations[tuple(pos)]] = i
            result.append(sorted_team_perm)
            # champ_roles = [self.roles[tuple(role)] for role in sorted_team]

            # the champ ids need to be ints, otherwise jq fails
            # champ_ids = [int(self.champ_manager.lookup_by("int", champ_int)["id"]) for champ_int in unsorted_team[
            #                                                                             :game_constants.CHAMPS_PER_TEAM]]
            # result.append(dict(zip(champ_roles, champ_ids)))

        return result



def export_models():
    models = [NextItemModel("standard"),
        NextItemModel("late"),
        NextItemModel("starter"),
        NextItemModel("first_item"), 
        NextItemModel("boots"), 
        ChampImgModel(),
        ItemImgModel(), 
        SelfImgModel(), 
        KDAImgModel(ui_constants.ResConverter(1024,768))]
    for model in models:
        model.load_model()
        model.export()
