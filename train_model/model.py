import glob
import threading
from abc import ABC, abstractmethod

import cv2 as cv
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.merge_ops import merge

from constants import ui_constants, game_constants, app_constants
from train_model import network
from utils.artifact_manager import ChampManager, ItemManager, SelfManager, SpellManager, CurrentGoldManager, \
    KDAManager, CSManager, LvlManager


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


    def output_logs(self, in_vec):
        sess = tf.InteractiveSession()
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
        item_emb_dim = network_config["item_emb_dim"]

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

        items_by_champ_k_hot_flat = tf.reshape(items_by_champ_k_hot, [-1, champs_per_game * total_num_items])

        items_by_champ_k_hot_rep = tf.reshape(items_by_champ_k_hot, [-1, total_num_items])

        target_summ_items_sparse = tf.gather_nd(items_by_champ, pos_index)
        target_summ_items = tf.gather_nd(items_by_champ_k_hot, pos_index)
        opp_summ_items = tf.gather_nd(items_by_champ_k_hot, opp_index)

        pos = tf.one_hot(pos, depth=champs_per_team)
        pos = tf.cast(pos, tf.int64)
        final_input_layer = merge(
            [pos, target_summ_items, target_summ_current_gold,
             target_summ_cs, target_summ_lvl, target_summ_kda,
             lvl, kda, total_cs],
            mode='concat', axis=1)

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


class SlideImgModel(ImgModel):

    def __init__(self, res_converter):
        super().__init__(res_converter)
        self.slide_img_height =
        self.slide_img_width
        self.sub_img_height
        self.sub_img_width
        self.x_crop res_cvt.KDA_X_CROP,
        self.y_crop res_cvt.KDA_Y_CROP
        self.network_size_x ui_constants.NETWORK_KDA_IMG_CROP[1],
                                 ui_constants.NETWORK_KDA_IMG_CROP[0]
        self.network_size_y


    def classify_img(self, img):
        # utils.show_coords(img, self.coords, self.img_size)
        x = [img[coord[1]:coord[1] + self.img_height, coord[0]:coord[0] + self.img_width] for coord in self.coords]
        x = [cv.resize(img, self.network_crop, cv.INTER_AREA) for img in x]
        return self.predict2int(np.array(x))


    def predict(self, x):
        img = x
        predicted_artifact_ints = self.classify_img(img)
        # 10 is the blank element
        for blank_index, element in enumerate(predicted_artifact_ints):
            if element == 10:
                break
        # cutoff the trailing empty digits
        predicted_artifact_ints = predicted_artifact_ints[:blank_index]

        result = 0
        for power_ten, artifact in enumerate(predicted_artifact_ints[::-1]):
            result += artifact * 10 ** power_ten
        return result


    def generate_coords(self, x_diff, y_diff, x_start, y_start):
        for team_offset in [0, x_diff]:
            for row in range(5):
                yield x_start + team_offset, row * y_diff + y_start


    def break_into_sub_imgs(self, slide_img):
        self.break_up_into_sub_imgs(slide_img, self.x_crop, self.y_crop)


    @staticmethod
    def break_up_into_sub_imgs(slide_img, X_CROP, Y_CROP):

        x_pad = y_pad = 20
        img_bordered = cv.copyMakeBorder(slide_img, x_pad, x_pad, y_pad, y_pad, cv.BORDER_CONSTANT, value=(0, 0, 0))

        gray = cv.cvtColor(img_bordered, cv.COLOR_BGR2GRAY)
        cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU, gray)
        contours, hier = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        for c in contours[::-1]:
            # get the bounding rect
            x, y, w, h = cv.boundingRect(c)
            cv.rectangle(slide_img, (x, y), (x + w, y + h), (0, 0, 255), 1)

            cv.imshow('g', slide_img)
            cv.waitKey(0)

            center_x = x + w / 2
            center_y = y + h/2
            crop_start_x = round(center_x - X_CROP/2)
            crop_start_y = round(center_y - Y_CROP/2)
            if crop_start_x < 0 or crop_start_y < 0:
                print("less than 0")
            sub_image = img_bordered[crop_start_y:crop_start_y + Y_CROP,
                        crop_start_x:crop_start_x + X_CROP]
            yield sub_image


    def extract_digit_imgs(self, whole_img):

        coords = list(self.generate_coords(res_cvt))

        coords = np.reshape(coords, (-1, 2))

        slide_imgs = [
            whole_img[coord[1]:coord[1] + self.slide_img_height,
            coord[0]:coord[0] + self.slide_img_width]
            for coord in coords]

        sub_imgs = []
        for img in slide_imgs:
            sub_imgs.extend(list(KDAImgModel.break_up_into_sub_imgs(img, self.x_crop, self.y_crop)))

        sub_imgs = [cv.resize(img, (self.network_size_x, self.network_size_y ),
                           cv.INTER_AREA) for img in sub_imgs]
        return sub_imgs




class CSImgModel(SlideImgModel):

    def __init__(self, res_converter):
        super().__init__(res_converter)

        self.network = network.DigitRecognitionNetwork(lambda: CSManager().get_num("img_int"),
                                                       ui_constants.NETWORK_CS_IMG_CROP)
        self.network_crop = ui_constants.NETWORK_CS_IMG_CROP
        self.img_height = res_converter.CS_HEIGHT
        self.img_width = res_converter.CS_WIDTH
        self.x_crop = res_converter.CS_X_CROP
        self.y_crop = res_converter.CS_Y_CROP

        self.model_path = app_constants.model_paths["best"]["cs_imgs"]
        self.artifact_manager = CSManager()



    def generate_coords(self, res_converter):
        return self.generate_coords(res_converter.CS_X_DIFF, res_converter.CS_Y_DIFF, res_converter.CS_X_START,
                                       res_converter.CS_Y_START)


class LvlImgModel(SlideImgModel):

    def __init__(self, res_converter):
        super().__init__(res_converter)

        self.network = network.DigitRecognitionNetwork(lambda: LvlManager().get_num("img_int"),
                                                       ui_constants.NETWORK_LVL_IMG_CROP)
        self.network_crop = ui_constants.NETWORK_LVL_IMG_CROP
        self.img_height = res_converter.LVL_HEIGHT
        self.img_width = res_converter.LVL_WIDTH
        self.x_crop = res_converter.LVL_X_CROP
        self.y_crop = res_converter.LVL_Y_CROP

        self.model_path = app_constants.model_paths["best"]["lvl_imgs"]
        self.artifact_manager = LvlManager()



    def generate_coords(self, res_converter):
        return SlideImgModel.generate_coords(res_converter.LVL_X_DIFF, res_converter.LVL_Y_DIFF,
                                             res_converter.LVL_X_START,
                                       res_converter.LVL_Y_START)


class KDAImgModel(SlideImgModel):

    def __init__(self, res_converter):
        super().__init__(res_converter)

        self.network = network.DigitRecognitionNetwork(lambda: KDAManager().get_num("img_int"),
                                                       ui_constants.NETWORK_KDA_IMG_CROP)
        self.network_crop = ui_constants.NETWORK_KDA_IMG_CROP
        self.img_height = res_converter.KDA_HEIGHT
        self.img_width = res_converter.KDA_WIDTH
        self.x_crop = res_converter.KDA_X_CROP
        self.y_crop = res_converter.KDA_Y_CROP

        self.model_path = app_constants.model_paths["best"]["kda_imgs"]
        self.artifact_manager = KDAManager()


    def generate_coords(self, res_converter):
        return self.generate_coords(res_converter.KDA_X_DIFF, res_converter.KDA_Y_DIFF, res_converter.KDA_X_START,
                                 res_converter.KDA_Y_START)



    def classify_img(self, img):
        # utils.show_coords(img, self.coords, self.img_size)
        x = [img[coord[1]:coord[1] + self.img_height, coord[0]:coord[0] + self.img_width] for coord in self.coords]
        x = [cv.resize(img, self.network_crop, cv.INTER_AREA) for img in x]
        return self.predict2int(np.array(x))


    def predict(self, x):
        img = x
        predicted_artifact_ints = self.classify_img(img)
        # 10 is the blank element
        for blank_index, element in enumerate(predicted_artifact_ints):
            if element == 10:
                break
        # cutoff the trailing empty digits
        predicted_artifact_ints = predicted_artifact_ints[:blank_index]

        result = 0
        for power_ten, artifact in enumerate(predicted_artifact_ints[::-1]):
            result += artifact * 10 ** power_ten
        return result


class CurrentGoldImgModel(ImgModel):

    def __init__(self, res_converter):
        super().__init__(res_converter)

        self.network = network.DigitRecognitionNetwork(lambda: CurrentGoldManager().get_num("img_int"),
                                                       ui_constants.NETWORK_CURRENT_GOLD_IMG_CROP)
        print(f"def get_num_elements(self): {self.network.get_num_elements()}")
        self.network_crop = ui_constants.NETWORK_CURRENT_GOLD_IMG_CROP
        self.img_size = res_converter.CURRENT_GOLD_DIGIT_SIZE
        self.model_path = app_constants.model_paths["best"]["current_gold_imgs"]
        self.artifact_manager = CurrentGoldManager()



    def predict(self, x):
        img = x
        predicted_artifact_ints = self.classify_img(img)
        # 10 is the blank element
        for blank_index, element in enumerate(predicted_artifact_ints):
            if element == 10:
                break
        # cutoff the trailing empty digits
        predicted_artifact_ints = predicted_artifact_ints[:blank_index]

        result = 0
        for power_ten, artifact in enumerate(predicted_artifact_ints[::-1]):
            result += artifact * 10 ** power_ten
        return result


    def generate_coords(self):
        return [(self.res_converter.CURRENT_GOLD_LEFT_X + self.res_converter.CURRENT_GOLD_DIGIT_WIDTH * i +
                 self.res_converter.CURRENT_GOLD_X_OFFSET,
                 self.res_converter.CURRENT_GOLD_TOP_Y + self.res_converter.CURRENT_GOLD_Y_OFFSET) for i in range(4)]



class ChampImgModel(ImgModel):

    def __init__(self, res_converter):
        super().__init__(res_converter)

        self.network = network.ChampImgNetwork()
        print(f"def get_num_elements(self): {self.network.get_num_elements()}")
        self.network_crop = ui_constants.NETWORK_CHAMP_IMG_CROP
        self.img_size = res_converter.CHAMP_SIZE
        self.model_path = app_constants.model_paths["best"]["champ_imgs"]
        self.artifact_manager = ChampManager()



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


class NextItemEarlyGameModel(Model):

    def __init__(self):
        super().__init__()
        self.network = network.NextItemEarlyGameNetwork()
        self.model_path = app_constants.model_paths["best"]["next_items_early"]
        self.artifact_manager = ItemManager()
        # self.load_model()

        # with open(glob.glob('models/best/next_items/early/thresholds*')[0]) as f:
        #     self.thresholds = json.load(f)
        self.thresholds = 1


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
        item_int = self.predict_with_prior(x)
        item = self.artifact_manager.lookup_by("int", item_int[0])
        return item


    # first item is the empty item... don't want to include that

    def predict_with_prior(self, x):
        with self.graph.as_default():
            y = self.model.predict(x)
            y_priored = y / self.thresholds
            y_int = np.argmax(y_priored, axis=len(y.shape) - 1)
            return y_int


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


