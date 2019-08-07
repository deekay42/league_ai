import numpy as np
import tensorflow as tf
import tflearn
import glob
from abc import ABC, abstractmethod
import cv2 as cv

from train_model import network
from utils.artifact_manager import ChampManager, ItemManager, SelfManager, SpellManager
from constants import ui_constants, game_constants, app_constants
import threading
from utils import utils


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
                self.model_path = glob.glob(self.model_path + "my_model*")[0]
                self.model_path = self.model_path.rpartition('.')[0]
                model.load(self.model_path, create_new_session=False)
            except Exception as e:
                print("Unable to open best model files")
                raise e
            self.model = model


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
        self.coords = None
        self.network_crop = None
        self.img_size = None
        self.res_converter = res_converter
        self.coords = self.generate_coords()
        self.coords = np.reshape(self.coords, (-1, 2))


    def classify_img(self, img):
        # utils.show_coords(img, self.coords, self.img_size)
        x = [img[coord[1]:coord[1] + self.img_size, coord[0]:coord[0] + self.img_size] for coord in self.coords]
        x = [cv.resize(img, self.network_crop, cv.INTER_AREA) for img in x]
        return self.predict2int(np.array(x))


    def predict(self, x):
        img = x
        predicted_artifact_ints = self.classify_img(img)
        predicted_artifacts = (self.artifact_manager.lookup_by("img_int", artifact_int) for artifact_int in
                               predicted_artifact_ints)
        return predicted_artifacts


    @abstractmethod
    def generate_coords(self):
        pass


class ChampImgModel(ImgModel):

    def __init__(self, res_converter):
        super().__init__(res_converter)
        self.network = network.ChampImgNetwork()
        self.network_crop = ui_constants.NETWORK_CHAMP_IMG_CROP
        self.img_size = res_converter.CHAMP_SIZE
        self.model_path = app_constants.model_paths["best"]["champ_imgs"]
        self.artifact_manager = ChampManager()
        self.load_model()


    def generate_coords(self):
        return ChampImgModel.generate_champ_coords(self.res_converter.CHAMP_LEFT_X_OFFSET,
                                                   self.res_converter.CHAMP_RIGHT_X_OFFSET,
                                                   self.res_converter.CHAMP_Y_DIFF,
                                                   self.res_converter.CHAMP_Y_OFFSET)


    @staticmethod
    def generate_champ_coords(left_x, right_x, y_diff, top_left_spell_y):
        champ_slots_coordinates = np.zeros((2, 5, 2), dtype=np.int64)
        x_offsets = (left_x, right_x)
        for x_offset, team in zip(x_offsets, range(2)):
            for player in range(5):
                champ_slots_coordinates[team][player] = (
                    round(x_offset), round(top_left_spell_y + player * y_diff))
        return champ_slots_coordinates



class ItemImgModel(ImgModel):

    def __init__(self, res_converter, summ_names_displayed):
        self.summ_names_displayed = summ_names_displayed
        super().__init__(res_converter)
        self.network = network.ItemImgNetwork()
        self.network_crop = ui_constants.NETWORK_ITEM_IMG_CROP
        self.img_size = res_converter.ITEM_SIZE
        self.model_path = app_constants.model_paths["best"]["item_imgs"]
        self.artifact_manager = ItemManager()
        self.load_model()


    def generate_coords(self):
        item_x_offset = self.res_converter.ITEM_INNER_OFFSET
        item_y_offset = self.res_converter.ITEM_INNER_OFFSET
        if self.summ_names_displayed:
            item_x_offset += self.res_converter.SUMM_NAMES_DIS_X_OFFSET
            item_y_offset += self.res_converter.SUMM_NAMES_DIS_Y_OFFSET
        return ItemImgModel.generate_item_coords(self.res_converter.ITEM_X_DIFF, self.res_converter.ITEM_LEFT_X_OFFSET,
                                                 self.res_converter.ITEM_RIGHT_X_OFFSET, self.res_converter.ITEM_Y_DIFF,
                                                 self.res_converter.ITEM_Y_OFFSET, item_x_offset, item_y_offset)


    @staticmethod
    def generate_item_coords(box_size, left_x, right_x, y_diff, top_left_trinket_y, x_offset, y_offset):
        item_slots_coordinates = np.zeros((2, 5, 7, 2), dtype=np.int64)
        total_x_offsets = (left_x + x_offset, right_x + x_offset)
        for total_x_offset, team in zip(total_x_offsets, range(2)):
            for player in range(5):
                for item in range(7):
                    item_slots_coordinates[team][player][item] = (
                        round(total_x_offset + item * box_size), round(top_left_trinket_y + y_offset + player * y_diff))
        return item_slots_coordinates


class SelfImgModel(ImgModel):

    def __init__(self, res_converter):
        super().__init__(res_converter)
        self.network = network.SelfImgNetwork()
        self.network_crop = ui_constants.NETWORK_SELF_IMG_CROP
        self.img_size = res_converter.SELF_INDICATOR_SIZE
        self.model_path = app_constants.model_paths["best"]["self_imgs"]
        self.artifact_manager = SelfManager()
        self.load_model()


    def generate_coords(self):
        return ChampImgModel.generate_champ_coords(self.res_converter.SELF_INDICATOR_LEFT_X_OFFSET,
                                                   self.res_converter.SELF_INDICATOR_RIGHT_X_OFFSET,
                                                   self.res_converter.SELF_INDICATOR_Y_DIFF,
                                                   self.res_converter.SELF_INDICATOR_Y_OFFSET)


    def predict(self, img):
        x = [img[coord[1]:coord[1] + self.img_size, coord[0]:coord[0] + self.img_size] for coord in self.coords]
        x = [cv.resize(img, self.network_crop, cv.INTER_AREA) for img in x]
        with self.graph.as_default():
            y = self.model.predict(x).flatten()
        role_index = np.argmax(y)
        return role_index


class NextItemsModel(Model):

    def __init__(self):
        super().__init__()
        self.network = network.NextItemNetwork()
        self.model_path = app_constants.model_paths["best"]["next_items"]
        self.artifact_manager = ItemManager()
        self.load_model()


    def predict(self, x):
        item_int = self.predict2int(x)
        item = self.artifact_manager.lookup_by("int", item_int[0])
        return item


class PositionsModel(Model):

    def __init__(self):
        super().__init__()
        self.network = network.PositionsNetwork()
        self.model_path = app_constants.model_paths["best"]["positions"]
        keys = [(1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 1, 0), (0, 0, 0, 0, 1)]
        self.roles = dict(zip(keys, game_constants.ROLE_ORDER))
        self.champ_manager = ChampManager()
        self.spell_manager = SpellManager()
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


    def multi_predict(self, x):
        with self.lock:
            with self.graph.as_default():
                pred = self.model.predict(x)
                with tf.Session() as sess:
                    final_pred = network.PositionsNetwork.best_permutations_one_hot(pred)
                    final_pred = sess.run(final_pred)
        result = []
        for sorted_team, unsorted_team in zip(final_pred, x):
            champ_roles = [self.roles[tuple(role)] for role in sorted_team]
            # the champ ids need to be ints, otherwise jq fails
            champ_ids = [int(self.champ_manager.lookup_by("int", champ_int)["id"]) for champ_int in unsorted_team[
                                                                                        :game_constants.CHAMPS_PER_TEAM]]
            result.append(dict(zip(champ_roles, champ_ids)))

        return result
