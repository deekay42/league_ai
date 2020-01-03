import glob
import threading
from abc import ABC, abstractmethod

import cv2 as cv
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.merge_ops import merge
from tflearn.layers.core import fully_connected, input_data, dropout

from constants import game_constants, app_constants, ui_constants
from train_model import network
from utils.artifact_manager import ChampManager, ItemManager, SimpleManager
import pytesseract
import json
import itertools
from tflearn.layers.embedding_ops import embedding
from collections import Counter
import os
import platform
import sklearn
from sklearn.externals.joblib import dump, load

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = os.path.abspath('Tesseract-OCR/tesseract.exe')

class Model(ABC):

    def __init__(self):
        self.network = None
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.model = None
        self.model_path = None
        self.artifact_manager = None


    def load_model(self):
        with self.graph.as_default():
            tflearn.is_training(False, session=self.session)
            self.network = self.network.build()
            model = tflearn.DNN(self.network, session=self.session)
            self.session.run(tf.global_variables_initializer())
            try:
                self.model_path = glob.glob(app_constants.model_paths["best"][self.elements] + "my_model*")[0]
                self.model_path = self.model_path.rpartition('.')[0]
                model.load(self.model_path, create_new_session=False)
            except Exception as e:
                print("Unable to open best model files")
                raise e
            self.model = model


    def output_logs(self, in_vec):
        sess = tf.InteractiveSession()
        tflearn.is_training(True, session=sess)
        game_config = \
            {
                "champs_per_game": game_constants.CHAMPS_PER_GAME,
                "champs_per_team": game_constants.CHAMPS_PER_TEAM,
                "total_num_champs": ChampManager().get_num("int"),
                "total_num_items": ItemManager().get_num("int"),
                "items_per_champ": game_constants.MAX_ITEMS_PER_CHAMP
            }

        network_config = \
            {
                "learning_rate": 0.00025,
                "champ_emb_dim": 6,
                "item_emb_dim": 7,
                "all_items_emb_dim": 10,
                "champ_all_items_emb_dim": 12,
                "target_summ": 1
            }

        champs_per_game = game_config["champs_per_game"]
        total_num_champs = game_config["total_num_champs"]
        total_num_items = game_config["total_num_items"]
        items_per_champ = game_config["items_per_champ"]
        champs_per_team = game_config["champs_per_team"]

        learning_rate = network_config["learning_rate"]
        champ_emb_dim = network_config["champ_emb_dim"]

        all_items_emb_dim = network_config["all_items_emb_dim"]
        champ_all_items_emb_dim = network_config["champ_all_items_emb_dim"]

        total_champ_dim = champs_per_game
        total_item_dim = champs_per_game * items_per_champ

        pos_start = 0
        pos_end = pos_start + 1
        champs_start = pos_end
        champs_end = champs_start + champs_per_game
        items_start = champs_end
        items_end = items_start + items_per_champ * 2 * champs_per_game
        total_gold_start = items_end
        total_gold_end = total_gold_start + champs_per_game
        cs_start = total_gold_end
        cs_end = cs_start + champs_per_game
        neutral_cs_start = cs_end
        neutral_cs_end = neutral_cs_start + champs_per_game
        xp_start = neutral_cs_end
        xp_end = xp_start + champs_per_game
        lvl_start = xp_end
        lvl_end = lvl_start + champs_per_game
        kda_start = lvl_end
        kda_end = kda_start + champs_per_game * 3
        current_gold_start = kda_end
        current_gold_end = current_gold_start + champs_per_game

        # in_vec = input_data(shape=[None, 221], name='input')
        #  1 elements long
        pos = in_vec[:, 0]
        pos = tf.cast(pos, tf.int32)

        n = tf.shape(in_vec)[0]
        batch_index = tf.range(n)
        pos_index = tf.transpose([batch_index, pos], (1, 0))
        opp_index = tf.transpose([batch_index, pos + champs_per_team], (1, 0))

        # Make tensor of indices for the first dimension

        #  10 elements long
        champ_ints = in_vec[:, champs_start:champs_end]
        #  10 elements long

        champ_ints = dropout(champ_ints, 0.8)
        # 60 elements long
        item_ints = in_vec[:, items_start:items_end]
        cs = in_vec[:, cs_start:cs_end]
        neutral_cs = in_vec[:, neutral_cs_start:neutral_cs_end]
        lvl = in_vec[:, lvl_start:lvl_end]
        kda = in_vec[:, kda_start:kda_end]
        current_gold = in_vec[:, current_gold_start:current_gold_end]
        total_cs = cs + neutral_cs

        target_summ_current_gold = tf.expand_dims(tf.gather_nd(current_gold, pos_index), 1)
        target_summ_cs = tf.expand_dims(tf.gather_nd(total_cs, pos_index), 1)
        target_summ_kda = tf.gather_nd(tf.reshape(kda, (-1, champs_per_game, 3)), pos_index)
        target_summ_lvl = tf.expand_dims(tf.gather_nd(lvl, pos_index), 1)




        champs_one_hot = tf.one_hot(tf.cast(champ_ints, tf.int32), depth=total_num_champs)
        opp_champs_one_hot = champs_one_hot[:, champs_per_team:]
        opp_champs_k_hot = tf.reduce_sum(opp_champs_one_hot, axis=1)
        # champs_one_hot_flat = tf.reshape(champs_one_hot, [-1, champs_per_game * total_num_champs])
        target_summ_champ = tf.gather_nd(champs_one_hot, pos_index)
        opp_summ_champ = tf.gather_nd(champs_one_hot, opp_index)


        items_by_champ = tf.reshape(item_ints, [-1, champs_per_game, items_per_champ, 2])
        items_by_champ_flat = tf.reshape(items_by_champ, [-1])

        batch_indices = tf.reshape(tf.tile(tf.expand_dims(tf.range(n), 1), [1, champs_per_game * items_per_champ]),
                                   (-1,))
        champ_indices = tf.reshape(tf.tile(tf.tile(tf.expand_dims(tf.range(champs_per_game), 1), [1,
                                                                                                  items_per_champ]),
                                           [n, 1]),
                                   (-1,))

        index_shift = tf.cast(tf.reshape(items_by_champ[:, :, :, 0] + 1, (-1,)), tf.int32)

        item_one_hot_indices = tf.cast(tf.transpose([batch_indices, champ_indices, index_shift], [1, 0]),
                                       tf.int64)

        items = tf.SparseTensor(indices=item_one_hot_indices, values=tf.reshape(items_by_champ[:, :, :, 1], (-1,)),
                                dense_shape=(n, champs_per_game, total_num_items + 1))
        items = tf.sparse.to_dense(items, validate_indices=False)
        items_by_champ_k_hot = items[:, :, 1:]



        target_summ_items_sparse = tf.gather_nd(items_by_champ, pos_index)
        target_summ_items = tf.gather_nd(items_by_champ_k_hot, pos_index)
        opp_summ_items = tf.gather_nd(items_by_champ_k_hot, opp_index)

        pos = tf.one_hot(pos, depth=champs_per_team)
        final_input_layer = merge(
            [pos, target_summ_champ, target_summ_items, opp_summ_champ],
            mode='concat', axis=1)
        lol = dropout(final_input_layer, 0.8)
        return items


    def predict2int(self, x):
        with self.graph.as_default():
            y = self.model.predict(x)
            y_int = np.argmax(y, axis=len(y.shape) - 1)
            return y_int


    @abstractmethod
    def predict(self, x):
        pass


class ImgModel(Model):

    def __init__(self, res_converter):
        super().__init__()
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
        sub_imgs = [whole_img[int(round(coord[1])):int(round(coord[1] + self.res_converter.lookup(self.elements,
                                                                                                  "y_crop"))),
                    int(round(coord[0])):int(round(coord[0] + self.res_converter.lookup(self.elements, "x_crop")))]
                    for coord in
                    coords]
        sub_imgs = [cv.resize(img, self.network_crop, cv.INTER_AREA) for img in sub_imgs]
        # for i, img in enumerate(sub_imgs):
        #     cv.imshow(str(i), img)
        # cv.waitKey(0)
        return sub_imgs


class MultiTesseractModel:
    def __init__(self, tesseractmodels):
        self.tesseractmodels = tesseractmodels
        self.config = "-l eng --oem 1 --psm 7 -c tessedit_char_whitelist=@0123456789"

    def predict(self, whole_img):
        slide_imgs = [model.extract_all_slide_imgs(whole_img) for model in self.tesseractmodels]
        with open(app_constants.asset_paths["tesseract_list_file"], "w") as f:
            index = 0
            for imgs in slide_imgs:
                for slide_img in imgs:
                    cv.imwrite(os.path.join(app_constants.asset_paths["tesseract_tmp_files"], str(index+1) + ".png"), slide_img)
                    f.write(os.path.join(app_constants.asset_paths["tesseract_tmp_files"], str(index+1) + ".png\n"))
                    index += 1

        text = pytesseract.image_to_string(app_constants.asset_paths["tesseract_list_file"], config=self.config,
                                           output_type=pytesseract.Output.BYTES
                                           ).decode('utf-8').split('\f')
        text = [token.strip() for token in text]
        offset = 0
        for i in range(len(slide_imgs)):
            yield np.array([self.tesseractmodels[i].convert(result) if result != '' else None for result in text[
                                                                                                  offset:offset+len(slide_imgs[
                                                                                                             i])]])
            offset += len(slide_imgs[i])


class TesseractModel:

    def __init__(self, res_converter):
        self.config = ("-l eng --oem 1 --psm 7 -c tessedit_char_whitelist=@0123456789")
        self.res_converter = res_converter
        self.separator = cv.imread( app_constants.asset_paths["tesseract_separator"], cv.IMREAD_GRAYSCALE)
        self.separator = cv.copyMakeBorder(self.separator, 0, 0, 5, 5, cv.BORDER_CONSTANT, value=(255, 255, 255))
        _, self.separator = cv.threshold(self.separator, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)


    def extract_slide_img(self, slide_img):

        x_pad = y_pad = 20
        img_bordered = cv.copyMakeBorder(slide_img, x_pad, x_pad, y_pad, y_pad, cv.BORDER_CONSTANT, value=(0, 0, 0))
        scale_factor = 5
        img_bordered = cv.resize(img_bordered, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_CUBIC)
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
        ratio = (y_bot - y_top) / self.separator.shape[0]
        separator = cv.resize(self.separator, None, fx=ratio, fy=ratio, interpolation=cv.INTER_AREA)
        blob_contour = gray[y_top:y_bot, x_left:x_right]
        ret, thresholded = cv.threshold(blob_contour, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        inv = cv.bitwise_not(thresholded)
        img = np.concatenate([separator, separator, inv, separator, separator], axis=1)
        img = cv.copyMakeBorder(img, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=(255, 255, 255))

        return img

    def get_coords(self):
        return list(self.res_converter.generate_std_coords(self.elements))

    def get_raw_slide_imgs(self, whole_img):
        coords = self.get_coords()
        coords = np.reshape(coords, (-1, 2))
        slide_imgs = [
            whole_img[int(round(coord[1])):int(round(coord[1] + self.res_converter.lookup(self.elements, "y_height"))),
            int(round(coord[0])):int(round(coord[0] + self.res_converter.lookup(self.elements, "x_width")))]
            for coord in coords]
        return slide_imgs


    def extract_all_slide_imgs(self, whole_img):
        slide_imgs = self.get_raw_slide_imgs(whole_img)
        result = [self.extract_slide_img(slide_img) for slide_img in slide_imgs]
        return result


    def predict(self, whole_img):

        imgs = self.extract_all_slide_imgs(whole_img)
        for i, img in enumerate(imgs):
            cv.imwrite(app_constants.asset_paths["tesseract_tmp_files"] + str(i+1)+".png", img)
        text = pytesseract.image_to_string(app_constants.asset_paths["tesseract_list_file"], config=self.config)
        return self.convert(text)



    def convert(self, tesseract_result):
        try:
            return int(tesseract_result[2:-2])
        except ValueError as e:
            return 0



# class SlideImgModel(ImgModel):
#
#     def __init__(self, res_converter):
#         super().__init__(res_converter)
#
#         self.manager = SimpleManager(self.elements)
#         self.network = network.DigitRecognitionNetwork(lambda: self.manager.get_num("img_int"), self.network_crop)
#         self.model_path = app_constants.model_paths["best"][self.elements]
#
#
#     def predict(self, x):
#         img = x
#         predicted_artifact_ints = [it["img_int"] for it in self.predict(img)]
#         return self.list2decimal(predicted_artifact_ints[::-1])
#
#
#     def list2decimal(self, list_):
#         result = 0
#         for power_ten, artifact in enumerate(list_):
#             result += artifact * 10 ** power_ten
#         return result
#     #
#     # def break_up_into_sub_imgs(self, slide_img):
#     #
#     #     x_pad = y_pad = 20
#     #     img_bordered = cv.copyMakeBorder(slide_img, x_pad, x_pad, y_pad, y_pad, cv.BORDER_CONSTANT, value=(0, 0, 0))
#     #
#     #     gray = cv.cvtColor(img_bordered, cv.COLOR_BGR2GRAY)
#     #
#     #     # cv.imshow("gray", gray)
#     #
#     #     cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU, gray)
#     #
#     #     # cv.imshow("gray2", gray)
#     #     contours, hier = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
#     #     x_l, y_l, w_l, h_l = cv.boundingRect(contours[-1])
#     #     x_r, y_r, w_r, h_r = cv.boundingRect(contours[0])
#     #     y_top = min(y_l, y_r)
#     #     height = max(h_l, h_r)
#     #
#     #     width = x_r - x_l + w_r
#     #
#     #     blob_contour = gray[y_top:y_top+height, x_l:x_r + w_r]
#     #
#     #     cv.imshow("blob", blob_contour)
#     #
#     #     num_digits = 1
#     #     single_digit_width = width
#     #     if width >= self.res_converter.lookup(self.elements, "triple_digit_x_width"):
#     #         single_digit_width /= 3
#     #         num_digits = 3
#     #     elif width >= self.res_converter.lookup(self.elements, "double_digit_x_width"):
#     #         single_digit_width /= 2
#     #         num_digits = 2
#     #
#     #     x = x_l
#     #     for i in range(num_digits):
#     #         # get the bounding rect
#     #         w = single_digit_width
#     #         h = height
#     #         y = y_top
#     #
#     #         # cv.rectangle(img_bordered, (x, y), (x + w, y + h), (0, 0, 255), 1)
#     #
#     #         center_x = x + w / 2
#     #         center_y = y + h / 2
#     #         crop_start_x = int(round(center_x - self.res_converter.lookup(self.elements, "x_crop") / 2))
#     #         crop_start_y = int(round(center_y - self.res_converter.lookup(self.elements, "y_crop") / 2))
#     #         if crop_start_x < 0 or crop_start_y < 0:
#     #             print("less than 0")
#     #
#     #         y_height = self.res_converter.lookup(self.elements, "y_crop")
#     #         x_width = self.res_converter.lookup(self.elements, "x_crop")
#     #         sub_image = img_bordered[crop_start_y:int(round(crop_start_y + y_height)),
#     #                     crop_start_x:int(round(crop_start_x + x_width))]
#     #
#     #         cv.rectangle(img_bordered, (crop_start_x, crop_start_y), (int(round(crop_start_x + x_width)),
#     #                                                                   int(round(crop_start_y +
#     #                                                                   y_height))), (0, 255,0), 1)
#     #         yield sub_image
#     #         x += single_digit_width
#     #     cv.imshow("fds", img_bordered)
#     #     cv.waitKey(0)
#
#
#     def break_up_into_sub_imgs(self, slide_img):
#
#         x_pad = y_pad = 20
#         img_bordered = cv.copyMakeBorder(slide_img, x_pad, x_pad, y_pad, y_pad, cv.BORDER_CONSTANT, value=(0, 0, 0))
#         scale_factor = 5
#         img_bordered = cv.resize(img_bordered, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_CUBIC)
#
#         gray = cv.cvtColor(img_bordered, cv.COLOR_BGR2GRAY)
#
#         cv.imshow("gray", gray)
#         cutoff_ratio = 2/5
#         ret, thresholded = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
#         sorted_thresholds = sorted(gray[thresholded > 0])
#         thresh = sorted_thresholds[int(len(sorted_thresholds) * cutoff_ratio)]
#         gray[gray < thresh] = 0
#
#         # ret, thresholded = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
#         # thresh = np.mean(gray[thresholded > 0])
#         # gray[gray < thresh] = 0
#
#         # cv.imshow("gray2", gray)
#         contours, hier = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
#         sorted_bboxes = sorted([cv.boundingRect(contour) for contour in contours], key=lambda a: a[0])
#
#         x_l, y_l, w_l, h_l = sorted_bboxes[0]
#         x_r, y_r, w_r, h_r = sorted_bboxes[-1]
#         y_top = min(y_l, y_r)
#         height = max(h_l, h_r)
#
#         width = x_r - x_l + w_r
#
#         blob_contour = gray[y_top:y_top + height, x_l:x_r + w_r]
#         blob_contour = cv.copyMakeBorder(blob_contour, 10, 10, 3, 3, cv.BORDER_CONSTANT, value=(0, 0, 0))
#
#         blob_contour = cv.bitwise_not(blob_contour)
#         blob_contour = np.concatenate([blob_contour for i in range(5)], axis=1)
#
#         cv.imshow("blob", blob_contour)
#
#         config = ("-l eng --oem 1 --psm 7 -c tessedit_char_whitelist=0123456789")
#         text = pytesseract.image_to_string(blob_contour, config=config)
#         print(text)
#
#         cv.waitKey(0)
#
#     #
#     # def break_up_into_sub_imgs(self, slide_img):
#     #
#     #     x_pad = y_pad = 20
#     #     img_bordered = cv.copyMakeBorder(slide_img, x_pad, x_pad, y_pad, y_pad, cv.BORDER_CONSTANT, value=(0, 0, 0))
#     #     scale_factor = 5
#     #     img_bordered = cv.resize(img_bordered, None,fx=scale_factor,fy=scale_factor, interpolation=cv.INTER_CUBIC)
#     #
#     #     gray = cv.cvtColor(img_bordered, cv.COLOR_BGR2GRAY)
#     #
#     #     # cv.imshow("gray", gray)
#     #     # cutoff_ratio = 2/5
#     #     # ret, thresholded = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
#     #     # sorted_thresholds = sorted(gray[thresholded > 0])
#     #     # thresh = sorted_thresholds[int(len(sorted_thresholds) * cutoff_ratio)]
#     #     # gray[gray < thresh] = 0
#     #
#     #
#     #     ret, thresholded = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
#     #     thresh = np.mean(gray[thresholded>0])
#     #     gray[gray < thresh] = 0
#     #
#     #
#     #
#     #     # cv.imshow("gray2", gray)
#     #     contours, hier = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
#     #     sorted_bboxes = sorted([cv.boundingRect(contour) for contour in contours], key=lambda a:a[0])
#     #
#     #     x_l, y_l, w_l, h_l = sorted_bboxes[0]
#     #     x_r, y_r, w_r, h_r = sorted_bboxes[-1]
#     #     y_top = min(y_l, y_r)
#     #     height = max(h_l, h_r)
#     #
#     #     width = x_r - x_l + w_r
#     #     blob_contour = gray[y_top:y_top+height, x_l:x_r + w_r]
#     #
#     #     cv.imshow("blob", blob_contour)
#     #
#     #     if width > height:
#     #         line_threshold = .20
#     #     else:
#     #         line_threshold = 0
#     #
#     #     vert_sum = np.sum(blob_contour, axis=0)
#     #     vert_sum_indexed = np.transpose([vert_sum, range(len(vert_sum))], (1,0))
#     #     vert_sum_sorted = np.array(sorted(vert_sum_indexed, key=lambda a: a[0]), dtype=np.int32)
#     #     blob_contour[:,vert_sum_sorted[:int(round(len(vert_sum)*line_threshold)), 1]] = 0
#     #
#     #     cv.imshow("blob_vertical_lines", blob_contour)
#     #
#     #
#     #     contours, hier = cv.findContours(blob_contour, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
#     #
#     #     for c in contours[::-1]:
#     #         # get the bounding rect
#     #         x, y, w, h = cv.boundingRect(c)
#     #         x += x_l
#     #         y += y_top
#     #
#     #         # cv.rectangle(img_bordered, (x, y), (x + w, y + h), (0, 0, 255), 1)
#     #
#     #         center_x = x + w / 2
#     #         center_y = y + h / 2
#     #         crop_start_x = round(center_x - scale_factor * self.res_converter.lookup(self.elements, "x_crop") / 2)
#     #         crop_start_y = round(center_y - scale_factor * self.res_converter.lookup(self.elements, "y_crop") / 2)
#     #         if crop_start_x < 0 or crop_start_y < 0:
#     #             print("less than 0")
#     #
#     #         y_height = scale_factor * self.res_converter.lookup(self.elements, "y_crop")
#     #         x_width = scale_factor * self.res_converter.lookup(self.elements, "x_crop")
#     #         sub_image = img_bordered[crop_start_y:int(round(crop_start_y + y_height)),
#     #                     crop_start_x:int(round(crop_start_x + x_width))]
#     #
#     #         cv.rectangle(img_bordered, (crop_start_x, crop_start_y), (int(round(crop_start_x + x_width)), int(round(crop_start_y +
#     #                                                                   y_height))), (0, 255,0), 1)
#     #         yield sub_image
#     #     cv.imshow("fds", img_bordered)
#     #     cv.waitKey(0)
#
#
#     def extract_imgs(self, whole_img):
#         coords = list(self.generate_coords())
#         coords = np.reshape(coords, (-1, 2))
#         slide_imgs = [
#             whole_img[coord[1]:int(round(coord[1] + self.res_converter.lookup(self.elements, "y_height"))),
#             coord[0]:int(round(coord[0] + self.res_converter.lookup(self.elements, "x_width")))]
#             for coord in coords]
#
#         sub_imgs = []
#         for img in slide_imgs:
#             # sub_imgs.extend(list(self.break_up_into_sub_imgs(img)))
#             self.break_up_into_sub_imgs(img)
#
#         sub_imgs = [cv.resize(img, (self.res_converter.network_crop[self.elements][1],
#                               self.res_converter.network_crop[self.elements][0]), cv.INTER_AREA) for img in
#                     sub_imgs]
#         return sub_imgs



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

    def __init__(self, res_converter):
        self.elements = "kda"

        super().__init__(res_converter)
        self.artifact_manager = SimpleManager(self.elements)
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
                    yield [0,0,0]
            else:
                yield [0,0,0]
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

    def __init__(self, res_converter):
        self.elements = "champs"
        super().__init__(res_converter)
        self.network = network.ChampImgNetwork()
        self.artifact_manager = ChampManager()



class ItemImgModel(ImgModel):

    def __init__(self, res_converter):
        self.elements = "items"
        super().__init__(res_converter)
        self.network = network.ItemImgNetwork()
        self.artifact_manager = ItemManager()



    def get_coords(self):
        return list(self.res_converter.generate_item_coords())


class SelfImgModel(ImgModel):

    def __init__(self, res_converter):
        self.elements = "self"
        super().__init__(res_converter)
        self.network = network.SelfImgNetwork()
        self.artifact_manager = SimpleManager(self.elements)


    def predict(self, img):
        x = self.extract_imgs(img)
        with self.graph.as_default():
            y = self.model.predict(x)
            role_index = np.argmax(y)
        return role_index


class NextItemEarlyGameModel(Model):

    def __init__(self):
        super().__init__()
        self.network = network.NextItemEarlyGameNetwork()
        self.model_path = app_constants.model_paths["best"]["next_items_early"]
        self.artifact_manager = ItemManager()
        self.elements = "next_items_early"
        # self.load_model()

        # with open(glob.glob('models/best/next_items/early/thresholds*')[0]) as f:
        #     self.thresholds = json.load(f)
        self.thresholds = 1

        champs_per_game = game_constants.CHAMPS_PER_GAME
        items_per_champ = game_constants.MAX_ITEMS_PER_CHAMP

        self.cont_slices_by_name = {'total_gold': np.s_[:, -90:-80],
         'cs': np.s_[:, -80:-70],
         'neutral_cs': np.s_[:, -70:-60],
         'xp': np.s_[:, -60:-50],
         'lvl': np.s_[:, -50:-40],
         'kda': np.s_[:, -40:-10],
         'cg': np.s_[:, -10:]}

        self.pos_start = 0
        self.pos_end = self.pos_start + 1
        self.champs_start = self.pos_end
        self.champs_end = self.champs_start + champs_per_game
        self.items_start = self.champs_end
        self.items_end = self.items_start + items_per_champ * 2 * champs_per_game
        self.total_gold_start = self.items_end
        self.total_gold_end = self.total_gold_start + champs_per_game
        self.cs_start = self.total_gold_end
        self.cs_end = self.cs_start + champs_per_game
        self.neutral_cs_start = self.cs_end
        self.neutral_cs_end = self.neutral_cs_start + champs_per_game
        self.xp_start = self.neutral_cs_end
        self.xp_end = self.xp_start + champs_per_game
        self.lvl_start = self.xp_end
        self.lvl_end = self.lvl_start + champs_per_game
        self.kda_start = self.lvl_end
        self.kda_end = self.kda_start + champs_per_game * 3
        self.current_gold_start = self.kda_end
        self.current_gold_end = self.current_gold_start + champs_per_game


    def predict_easy(self, role, champs_int, items_id, cs, lvl, kda, current_gold):
        x = np.zeros(shape=self.current_gold_end, dtype=np.int32)
        x[self.pos_start:self.pos_end] = [role]
        x[self.champs_start:self.champs_end] = champs_int
        encoded_items = np.ravel(self.encode_items(items_id, self.artifact_manager))
        x[self.items_start:self.items_end] = encoded_items
        x[self.cs_start:self.cs_end] = cs
        x[self.lvl_start:self.lvl_end] = lvl
        x[self.kda_start:self.kda_end] = np.ravel(kda)
        num_increments = 20
        current_gold_list = np.zeros((num_increments,10))


        current_gold_list[:,role] = np.array([current_gold]*num_increments) + np.array(range(0,num_increments*100,100))

        print(current_gold_list[:,role])
        x = np.tile(x,num_increments).reshape((num_increments,-1))
        x[:, self.current_gold_start:self.current_gold_end] = current_gold_list
        x = self.scale_inputs(np.array(x).astype(np.float32))
        return self.predict(x)


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
            else:
                print("WTFFFFFFFF")
            X[slice] = scaler.transform(X[slice])
        return X

    def fit_input(self, X, scaler_name):
        min_max_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
        min_max_scaler.fit(np.reshape(X, (-1, 1)))
        return min_max_scaler


    @staticmethod
    def num_itemslots(items):
        if not items:
            return 0
        wards = ItemManager().lookup_by("name", "Control Ward")["int"]
        hpots = ItemManager().lookup_by("name", "Health Potion")["int"]
        num_single_slot_items = int(items.get(wards, 0)>0) + int(items.get(hpots, 0)>0)
        reg_item_keys = (set(items.keys()) - {hpots, wards})
        num_reg_items = sum([items[key] for key in reg_item_keys])
        return num_single_slot_items + num_reg_items

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

    def predict(self, x):
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

        with self.graph.as_default():
            y = self.model.predict(x)
            item_ints = np.argmax(y, axis=len(y.shape) - 1)
            print(f"Confidence: {np.max(y, axis=1)}")
        items = [self.artifact_manager.lookup_by("int", item_int) for item_int in item_ints]
        print([item["name"] for item in items])
        print("\n\n")
        return items


class NextItemLateGameModel(Model):

    def __init__(self):
        super().__init__()
        self.network = network.NextItemLateGameNetwork()
        self.model_path = app_constants.model_paths["best"]["next_items_late"]
        self.artifact_manager = ItemManager()


    def predict2int_blackouts(self, x, blackout_indices):
        with self.graph.as_default():
            y = self.model.predict(x)[0]
            y = np.array(list(zip(y, range(len(y)))))
            blackout_indices = list(blackout_indices)
            y = np.delete(y, blackout_indices, axis=0)
            logits = y[:, 0]
            old_indices = y[:, 1]
            y_int = np.argmax(logits, axis=0)
            y_int = old_indices[y_int]
            return [int(y_int)]


    def predict(self, x, blackout_indices):
        # with self.graph.as_default(), tf.Session() as sess:
        #     tflearn.is_training(False, session=sess)
        #     X = tf.placeholder("float", [None, 71])
        #     # X = input_data(shape=[None, 71], name='lolol')
        #     log = self.output_logs(X)
        #     sess.run(tf.global_variables_initializer())
        #     log = sess.run(log, feed_dict = {X: np.array(x)})
        #     for i, _ in enumerate(log):
        #         print(f"{i}: {log[i]}")
        item_int = self.predict2int_blackouts(x, blackout_indices)
        item = self.artifact_manager.lookup_by("int", item_int[0])
        return item


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
                with tf.Session() as sess:
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

#