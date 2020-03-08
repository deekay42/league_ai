from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
import tflearn
from tflearn.activations import relu
from tflearn.layers.conv import conv_2d, conv_1d, max_pool_2d
from tflearn.layers.core import fully_connected, input_data, dropout
from tflearn.layers.embedding_ops import embedding
from tflearn.layers.estimator import regression, regression_custom
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import batch_normalization

from constants import game_constants
from constants.ui_constants import ResConverter
from utils.artifact_manager import ChampManager, ItemManager, SimpleManager


class Network(ABC):

    @abstractmethod
    def build(self):
        pass


class ImgNetwork(Network):

    def __init__(self):
        self.num_elements = self.get_num_elements()
        self.network_config = \
            {
                "learning_rate": 0.001
            }


    @abstractmethod
    def get_num_elements(self):
        pass


class DigitRecognitionNetwork(ImgNetwork):

    def __init__(self, num_func, img_size):
        self.num_func = num_func
        self.img_size = img_size
        super().__init__()


    def get_num_elements(self):
        return self.num_func()


    def build(self):
        is_training = tflearn.get_training_mode()
        # input
        network = input_data(shape=[None, self.img_size[0], self.img_size[1], 3], name='input')

        conv1 = relu(batch_normalization(
            conv_2d(network, 16, 3, bias=False, activation=None, regularizer="L2"), trainable=is_training))
        maxpool1 = max_pool_2d(conv1, 3)

        conv2 = relu(batch_normalization(
            conv_2d(maxpool1, 32, 3, bias=False, activation=None, regularizer="L2"), trainable=is_training))
        maxpool2 = max_pool_2d(conv2, 3)

        net = fully_connected(maxpool2, 64, activation='relu', regularizer="L2")
        net = fully_connected(net, self.num_elements, activation='softmax')

        return regression(net, optimizer='adam', learning_rate=self.network_config["learning_rate"], to_one_hot=True,
                          n_classes=self.num_elements, shuffle_batches=True,
                          loss='categorical_crossentropy', name='target')


class ChampImgNetwork(ImgNetwork):

    def __init__(self):
        super().__init__()
        self.img_size = ResConverter.network_crop["champs"]


    def get_num_elements(self):
        return ChampManager().get_num("img_int")


    def build(self):
        is_training = tflearn.get_training_mode()
        # input
        network = input_data(shape=[None, self.img_size[0], self.img_size[1], 3], name='input')

        conv1 = relu(batch_normalization(
            conv_2d(network, 16, 5, bias=False, activation=None, regularizer="L2"), trainable=is_training))
        maxpool1 = max_pool_2d(conv1, 3)

        conv2 = relu(batch_normalization(
            conv_2d(maxpool1, 32, 5, bias=False, activation=None, regularizer="L2"), trainable=is_training))
        maxpool2 = max_pool_2d(conv2, 3)

        net = fully_connected(maxpool2, 64, activation='relu', regularizer="L2")
        # net = fully_connected(net, 128, activation='relu')
        net = fully_connected(net, self.num_elements, activation='softmax')

        return regression(net, optimizer='adam', learning_rate=self.network_config["learning_rate"], to_one_hot=True,
                          n_classes=self.num_elements, shuffle_batches=True,
                          loss='categorical_crossentropy', name='target')


class ItemImgNetwork(ImgNetwork):

    def __init__(self):
        super().__init__()
        self.img_size = ResConverter.network_crop["items"]


    def get_num_elements(self):
        return ItemManager().get_num("img_int")


    def build(self):
        is_training = tflearn.get_training_mode()
        # input
        network = input_data(shape=[None, self.img_size[0], self.img_size[1], 3], name='input')

        conv1 = relu(batch_normalization(
            conv_2d(network, 32, 5, bias=False, activation=None, regularizer="L2"), trainable=is_training))
        maxpool1 = max_pool_2d(conv1, 3)

        conv2 = relu(batch_normalization(
            conv_2d(maxpool1, 64, 5, bias=False, activation=None, regularizer="L2"), trainable=is_training))
        maxpool2 = max_pool_2d(conv2, 3)

        net = fully_connected(maxpool2, 128, activation='relu', regularizer="L2")
        # net = fully_connected(net, 128, activation='relu')
        net = fully_connected(net, self.num_elements, activation='softmax')

        return regression(net, optimizer='adam', learning_rate=self.network_config["learning_rate"], to_one_hot=True,
                          n_classes=self.num_elements, shuffle_batches=True,
                          loss='categorical_crossentropy', name='target')


class SelfImgNetwork(ImgNetwork):

    def __init__(self):
        super().__init__()
        self.img_size = ResConverter.network_crop["self"]


    def get_num_elements(self):
        return 1


    # binary_acc doesn't really show accuracy here and will converge at 0.5... even though the network learns the function.
    # tf.one_hot([0], 1) will give 1 and tf.one_hot([1], 1) will give 0... wtf?
    def build(self):
        is_training = tflearn.get_training_mode()
        # input
        network = input_data(shape=[None, self.img_size[0], self.img_size[1], 3], name='input')

        conv1 = relu(batch_normalization(
            conv_2d(network, 8, 5, bias=False, activation=None, regularizer="L2"), trainable=is_training))
        maxpool1 = max_pool_2d(conv1, 2)

        # conv2 = relu(batch_normalization(
        #     conv_2d(maxpool1, 16, 5, bias=False, activation=None, regularizer="L2"), trainable=is_training))
        # maxpool2 = max_pool_2d(conv2, 2)

        # net = fully_connected(maxpool2, 16, activation='relu', regularizer="L2")
        net = fully_connected(maxpool1, self.num_elements, activation='sigmoid')

        return regression(net, optimizer='adam', learning_rate=self.network_config["learning_rate"],
                          loss='binary_crossentropy', name='target', metric=SelfImgNetwork.bin_acc)


    @staticmethod
    def bin_acc(preds, targets, input_):
        preds = tf.round(preds)
        correct_prediction = tf.equal(preds, targets)
        all_correct = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), 1)
        acc = tf.reduce_mean(all_correct)
        return acc


class PositionsNetwork(Network):

    def __init__(self):
        self.game_config = \
            {
                "champs_per_team": game_constants.CHAMPS_PER_TEAM,
                "total_num_champs": ChampManager().get_num("int"),
                "spells_per_summ": game_constants.SPELLS_PER_CHAMP,
                "total_num_spells": SimpleManager("spells").get_num("int"),
                "rest_dim": 8
            }

        self.network_config = \
            {
                "learning_rate": 0.00025,
                "champ_emb_dim": 6,
            }


    @staticmethod
    def permute(a, l, r):
        if l == r:
            yield list(zip([0, 1, 2, 3, 4], a))
        else:
            for i in range(l, r + 1):
                a[l], a[i] = a[i], a[l]
                yield from PositionsNetwork.permute(a, l + 1, r)
                a[l], a[i] = a[i], a[l]


    @staticmethod
    def permutate_inputs(x):
        perm_indices = tf.constant(
            np.array(list(PositionsNetwork.permute([0, 1, 2, 3, 4], 0, 4)))[:, :, 1][:, :, tf.newaxis])
        x = np.transpose(x, (1, 2, 0))
        x = tf.gather_nd(np.array(x), perm_indices)
        x = tf.transpose(x, (3, 0, 1, 2))
        x = tf.reshape(x, (-1, 55))
        return x


    # expects already permutated pred input vector
    @staticmethod
    def select_best_input_perm(pred):

        perm_indices = tf.constant(
            np.array(list(PositionsNetwork.permute([0, 1, 2, 3, 4], 0, 4)))[:, :, 1][:, :, tf.newaxis])
        num_perms = perm_indices.shape[0]
        batch_size = pred.shape[0] // num_perms
        pred = PositionsNetwork.best_permutations_one_hot(pred)
        pred_5x5_one_hot_by_permed_example = tf.reshape(pred, (-1, num_perms, 5, 5))
        batch_range = tf.tile(tf.range(batch_size), [num_perms * 5])
        batch_range = tf.reshape(batch_range, (num_perms * 5, -1))
        batch_range = tf.transpose(batch_range, (1, 0))
        batch_range = tf.reshape(batch_range, (-1, 5, 1))
        perm_range = tf.tile(tf.range(num_perms), [5])
        perm_range = tf.reshape(perm_range, (5, -1))
        perm_range = tf.transpose(perm_range, (1, 0))
        perm_range = tf.tile(perm_range, [batch_size, 1])
        perm_range = tf.reshape(perm_range, (-1, 5, 1))
        repeat_perms = tf.tile(perm_indices, [batch_size, 1, 1])
        repeat_perms = tf.cast(repeat_perms, tf.int32)
        inv_perm_indices = tf.concat([batch_range, perm_range, repeat_perms],
                                     axis=2)
        inv_perm_indices_shaped = tf.reshape(inv_perm_indices, (batch_size, num_perms, 5, 3))
        x_perms_inv = tf.gather_nd(pred_5x5_one_hot_by_permed_example, inv_perm_indices_shaped)
        x_perms_inv_summed = tf.reduce_sum(x_perms_inv, axis=1)
        best_perm = PositionsNetwork.best_permutations_one_hot(x_perms_inv_summed)
        return best_perm


    @staticmethod
    def best_permutations_one_hot(pred):
        pred_5x5 = tf.reshape(pred, [-1, 5, 5])
        pred_5x5_T = tf.transpose(pred_5x5, (1, 2, 0))
        all_perms = tf.constant(list(PositionsNetwork.permute([0, 1, 2, 3, 4], 0, 4)))
        selected_elements_per_example = tf.gather_nd(pred_5x5_T, all_perms)
        sums_per_example = tf.reduce_sum(selected_elements_per_example, axis=1)
        best_perm_per_example_index = tf.argmax(sums_per_example, axis=0)
        best_perms = tf.gather_nd(all_perms, best_perm_per_example_index[:, tf.newaxis])[:, :, 1]
        pred_5x5_one_hot = tf.reshape(tf.one_hot(best_perms, depth=5), (-1, 5, 5))
        return pred_5x5_one_hot


    @staticmethod
    def multi_class_acc_positions(pred, target, input_):
        pred_5x5_one_hot = PositionsNetwork.best_permutations_one_hot(pred)
        target_5x5 = tf.reshape(target, [-1, 5, 5])
        correct_prediction = tf.equal(tf.argmax(pred_5x5_one_hot, axis=2), tf.argmax(target_5x5, axis=2))
        all_correct = tf.reduce_min(tf.cast(correct_prediction, tf.float32), 1)
        acc = tf.reduce_mean(all_correct)
        return acc


    def build(self):
        champs_per_team = self.game_config["champs_per_team"]
        total_num_champs = self.game_config["total_num_champs"]
        spells_per_summ = self.game_config["spells_per_summ"]
        total_num_spells = self.game_config["total_num_spells"]
        rest_dim = self.game_config["rest_dim"]
        total_sum_dim = 1 + spells_per_summ + rest_dim

        learning_rate = self.network_config["learning_rate"]
        champ_emb_dim = self.network_config["champ_emb_dim"]

        in_vec = input_data(shape=[None, champs_per_team + champs_per_team * (spells_per_summ + rest_dim)],
                            name='input')

        champ_ints = in_vec[:, ::total_sum_dim]
        champs = embedding(champ_ints, input_dim=total_num_champs, output_dim=champ_emb_dim, reuse=tf.AUTO_REUSE,
                           scope="champ_scope_emb")
        champs = tf.reshape(champs, [-1, champs_per_team * champ_emb_dim])

        spell_ints = [in_vec[:, i * total_sum_dim + 1:i * total_sum_dim + 1 + spells_per_summ] for i in
                      range(champs_per_team)]
        spell_ints = tf.transpose(spell_ints, (1, 0, 2))
        spell_ints = tf.reshape(spell_ints, [-1, champs_per_team, spells_per_summ])

        spells_one_hot_i = tf.one_hot(tf.cast(spell_ints, tf.int32), depth=total_num_spells)
        spells_one_hot = tf.reduce_sum(spells_one_hot_i, axis=2)
        spells_one_hot = tf.reshape(spells_one_hot, [-1, champs_per_team * total_num_spells])

        rest = [in_vec[:, i * total_sum_dim + 1 + spells_per_summ:i * total_sum_dim + 1 + spells_per_summ + rest_dim]
                for i in
                range(champs_per_team)]
        rest = tf.transpose(rest, (1, 0, 2))
        rest = tf.reshape(rest, [-1, champs_per_team * rest_dim])

        final_input_layer = merge([champs, spells_one_hot, rest], mode='concat', axis=1)

        # net = dropout(final_input_layer, 0.8)
        net = final_input_layer
        net = relu(
            batch_normalization(fully_connected(net, 256, bias=False, activation=None, regularizer="L2")))
        # net = dropout(net, 0.6)

        net = fully_connected(net, champs_per_team * champs_per_team, activation=None)

        return regression(net, optimizer='adam', to_one_hot=False, shuffle_batches=True,
                          learning_rate=learning_rate,
                          loss='binary_crossentropy', name='target', metric=PositionsNetwork.multi_class_acc_positions)


class NextItemNetwork(Network):

    def __init__(self, my_champ_emb_scales=None, opp_champ_emb_scales=None):
        self.game_config = \
            {
                "champs_per_game": game_constants.CHAMPS_PER_GAME,
                "champs_per_team": game_constants.CHAMPS_PER_TEAM,
                "total_num_champs": ChampManager().get_num("int"),

                "total_num_items": ItemManager().get_num("int"),
                "items_per_champ": game_constants.MAX_ITEMS_PER_CHAMP
            }
        self.network_config = \
            {
                "learning_rate": 0.00025,
                "champ_emb_dim": 3,
                "all_items_emb_dim": 6,
                "champ_all_items_emb_dim": 8,
                "class_weights": [1]
            }

        if my_champ_emb_scales is not None:
            self.network_config["my_champ_emb_scales"] = (np.repeat(my_champ_emb_scales, self.network_config[
                "champ_emb_dim"])).astype(np.float32)
            self.network_config["opp_champ_emb_scales"] = (np.repeat(opp_champ_emb_scales, self.network_config[
                "champ_emb_dim"])).astype(np.float32)
        else:
            self.network_config["my_champ_emb_scales"] = (np.repeat([0.]*ChampManager().get_num("int"), self.network_config[
                "champ_emb_dim"])).astype(np.float32)
            self.network_config["opp_champ_emb_scales"] = (np.repeat([0.]*ChampManager().get_num("int"), self.network_config[
                "champ_emb_dim"])).astype(np.float32)

        self.pos_start = 0
        self.pos_end = self.pos_start + 1
        self.champs_start = self.pos_end
        self.champs_end = self.champs_start + self.game_config["champs_per_game"]
        self.items_start = self.champs_end
        self.items_end = self.items_start + self.game_config["items_per_champ"] * 2 * self.game_config[
            "champs_per_game"]
        self.total_gold_start = self.items_end
        self.total_gold_end = self.total_gold_start + self.game_config["champs_per_game"]
        self.cs_start = self.total_gold_end
        self.cs_end = self.cs_start + self.game_config["champs_per_game"]
        self.neutral_cs_start = self.cs_end
        self.neutral_cs_end = self.neutral_cs_start + self.game_config["champs_per_game"]
        self.xp_start = self.neutral_cs_end
        self.xp_end = self.xp_start + self.game_config["champs_per_game"]
        self.lvl_start = self.xp_end
        self.lvl_end = self.lvl_start + self.game_config["champs_per_game"]
        self.kda_start = self.lvl_end
        self.kda_end = self.kda_start + self.game_config["champs_per_game"] * 3
        self.current_gold_start = self.kda_end
        self.current_gold_end = self.current_gold_start + self.game_config["champs_per_game"]


    @abstractmethod
    def build(self):
        pass


    def get_champ_embeddings(self, team_champ_ints, emb_name, scale_name, start, end, pos_index, n, dropout_rate):
        emb_range = end - start
        team_champ_embs = [self.noise_embeddings(team_champ_ints, scale_name,
                                                 emb_name, 1 / i) for i in range(start, end)]
        team_champ_embs = tf.transpose(tf.reshape(team_champ_embs, (emb_range, -1, self.game_config["champs_per_team"],
                                                                    self.network_config[
                                                                        "champ_emb_dim"]
                                                                    )), (1, 2, 0, 3))
        target_summ_champ_emb = tf.gather_nd(team_champ_embs, pos_index)
        target_summ_champ_emb_dropout = dropout(target_summ_champ_emb, dropout_rate, noise_shape=[n, emb_range, 1])
        target_summ_champ_emb_dropout_flat = tf.reshape(target_summ_champ_emb_dropout,
                                                        (-1, emb_range * self.network_config[
                                                            "champ_emb_dim"]))

        team_champ_embs_dropout = dropout(team_champ_embs, dropout_rate, noise_shape=[n, self.game_config["champs_per_team"],
                                                                             emb_range, 1])
        team_champ_embs_dropout_flat = tf.reshape(team_champ_embs_dropout, (-1, self.game_config[
            "champs_per_team"] * emb_range * self.network_config["champ_emb_dim"]))

        return target_summ_champ_emb_dropout_flat, team_champ_embs_dropout_flat


    def get_champ_embeddings_v2(self, team_champ_ints, emb_name, angles, pos_index, n, dropout_rate):

        team_champ_embs = [self.noise_embeddings_v2(team_champ_ints,
                                                 emb_name, i) for i in angles]
        team_champ_embs = tf.transpose(tf.reshape(team_champ_embs, (len(angles), -1, self.game_config[
            "champs_per_team"],
                                                                    self.network_config[
                                                                        "champ_emb_dim"]
                                                                    )), (1, 2, 0, 3))
        target_summ_champ_emb = tf.gather_nd(team_champ_embs, pos_index)
        target_summ_champ_emb_dropout = dropout(target_summ_champ_emb, dropout_rate, noise_shape=[n, len(angles), 1])
        target_summ_champ_emb_dropout_flat = tf.reshape(target_summ_champ_emb_dropout,
                                                        (-1, len(angles) * self.network_config[
                                                            "champ_emb_dim"]))

        team_champ_embs_dropout = dropout(team_champ_embs, dropout_rate, noise_shape=[n, self.game_config["champs_per_team"],
                                                                             len(angles), 1])
        team_champ_embs_dropout_flat = tf.reshape(team_champ_embs_dropout, (-1, self.game_config[
            "champs_per_team"] * len(angles) * self.network_config["champ_emb_dim"]))

        return target_summ_champ_emb_dropout_flat, team_champ_embs_dropout_flat


    def calc_noise(self, noise_name, noisyness):
        my_team_emb_noise_dist = tf.distributions.Normal(loc=[0.] * self.game_config["total_num_champs"] *
                                                             self.network_config["champ_emb_dim"],
                                                         scale=self.network_config[noise_name]*noisyness)
        my_team_emb_noise = my_team_emb_noise_dist.sample([1])
        my_team_emb_noise = tf.reshape(my_team_emb_noise, (-1, 3))
        return my_team_emb_noise

    def apply_noise(self, champ_ints, my_team_emb_noise, my_team_champs_embedded):
        is_training = tflearn.get_training_mode()
        my_team_champs_embedded_noise = tf.cast(tf.gather(my_team_emb_noise,
                                                          tf.cast(champ_ints, tf.int32)), tf.float32)
        my_team_champs_embedded = tf.cond(is_training, lambda: my_team_champs_embedded + my_team_champs_embedded_noise,
                                          lambda: my_team_champs_embedded)
        return my_team_champs_embedded


    def apply_noise_v2(self, my_team_emb_noised, my_team_champs_embedded):
        is_training = tflearn.get_training_mode()
        my_team_champs_embedded = tf.cond(is_training, lambda: my_team_emb_noised,
                                          lambda: my_team_champs_embedded)
        return my_team_champs_embedded


    def noise_embeddings(self, champ_ints, noise_name, emb_name, noisyness):
        my_team_champs_embedded = embedding(champ_ints, input_dim=self.game_config["total_num_champs"],
                                            output_dim=self.network_config["champ_emb_dim"], trainable=False,
                                            name=emb_name)

        noise = self.calc_noise(noise_name, noisyness)
        my_team_champs_embedded = self.apply_noise(champ_ints, noise, my_team_champs_embedded)

        return my_team_champs_embedded


    def noise_embeddings_v2(self, champ_ints, emb_name, noisyness):
        my_team_champs_embedded = embedding(champ_ints, input_dim=self.game_config["total_num_champs"],
                                            output_dim=self.network_config["champ_emb_dim"], trainable=False,
                                            name=emb_name)

        my_team_champs_embedded_noised = self.rand_cos_sim(my_team_champs_embedded, noisyness)
        my_team_champs_embedded = self.apply_noise_v2(my_team_champs_embedded_noised, my_team_champs_embedded)

        return my_team_champs_embedded


    @staticmethod
    def rand_cos_sim(v, dev):
        costheta = 1-tf.expand_dims(tf.abs(tf.random.normal((tf.shape(v)[:-1]), 0, dev)), (-1))
        u = v / tf.sqrt(tf.reduce_sum(tf.square(v), axis=1, keep_dims=True) + 1e-8)
        r = tf.random.normal((tf.shape(v)))
        uperp = r - tf.expand_dims(tf.reduce_sum(u*r, axis=-1), axis=-1) * u
        uperp = uperp / tf.sqrt(tf.reduce_sum(tf.square(uperp), axis=1, keep_dims=True) + 1e-8)
        w = costheta * u + tf.sqrt(1 - costheta ** 2) * uperp
        return w


    @staticmethod
    def multi_class_acc(pred, target, input_):
        pred = tf.reshape(pred, [-1, 5, 204])
        target = tf.reshape(target, [-1, 5, 204])
        correct_prediction = tf.equal(tf.argmax(pred, axis=2), tf.argmax(target, axis=2))
        # all_labels_true = tf.reduce_min(tf.cast(correct_prediction, tf.float32), 2)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return acc


    @staticmethod
    def multi_class_top_k_acc(preds, targets, input_):
        preds = tf.reshape(preds, [-1, 204])
        targets = tf.reshape(targets, [-1, 204])
        targets = tf.cast(targets, tf.int32)
        correct_pred = tf.nn.in_top_k(preds, tf.argmax(targets, 1), 4)
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return acc


    def filter_invalid_indices(self, ets_mag, ets_dir, team_emb_dim, n):
        valid_mag_idx = tf.reshape(tf.greater_equal(ets_mag, 1e-7), (-1,))
        valid_mag_idx_i = tf.where(valid_mag_idx)
        ets_magnitude = tf.boolean_mask(ets_mag, valid_mag_idx)
        ets_direction = tf.boolean_mask(ets_dir, valid_mag_idx)
        ets_magnitude = tf.scatter_nd(valid_mag_idx_i, ets_magnitude, (n, 1))
        ets_direction = tf.scatter_nd(valid_mag_idx_i, ets_direction, (n, team_emb_dim))
        return [ets_direction, ets_magnitude]


    def calc_norm_mag(self, emb, norm_type, team_emb_dim, n):
        if norm_type == "l2":
            ets_magnitude = tf.sqrt(tf.reduce_sum(tf.square(emb), axis=1, keep_dims=True) + 1e-8)
        elif norm_type == "l1":
            ets_magnitude = tf.reduce_sum(tf.abs(emb), axis=1, keep_dims=True) + 1e-8
        elif norm_type == "max":
            ets_magnitude = tf.reduce_max(tf.abs(emb), axis=1, keep_dims=True)
        ets_direction = tf.math.divide_no_nan(emb, ets_magnitude)

        return self.filter_invalid_indices(ets_magnitude, ets_direction, team_emb_dim, n)


    def calc_enemy_team_strength(self, enemy_summs_strength_output, dim, champ_ints, total_num_champs, n):
        champs_embedded = embedding(champ_ints, input_dim=total_num_champs, output_dim=dim)
        opp_champ_emb = champs_embedded[:, 5:10]
        enemy_summs_strength_output = tf.tile(enemy_summs_strength_output, multiples=[1, 1, dim])
        enemy_team_strength = enemy_summs_strength_output * opp_champ_emb
        enemy_team_strength = tf.reduce_sum(enemy_team_strength, axis=1)
        norm_types = ["l2", "l1", "max"]
        for norm_type in norm_types:
            yield self.calc_norm_mag(enemy_team_strength, norm_type, dim, n)


    def build_team_convs(self, opp_champ_emb):
        et_lane_conv = batch_normalization(conv_1d(opp_champ_emb, 32, 2, bias=False, activation='relu',
                                                   padding='valid'))
        et_lane_conv = batch_normalization(conv_1d(et_lane_conv, 64, 2, bias=False, activation='relu',
                                                   padding='valid'))
        et_lane_conv = batch_normalization(conv_1d(et_lane_conv, 128, 2, bias=False, activation='relu',
                                                   padding='valid'))
        et_lane_conv = batch_normalization(conv_1d(et_lane_conv, 256, 2, bias=False, activation='relu',
                                                   padding='valid'))

        return tf.reshape(et_lane_conv, (-1, 256))


    # def multi_class_acc_positions(pred, target, input):
    #     pred = tf.reshape(pred, [-1, 5, 5])
    #     target = tf.reshape(target, [-1, 5, 5])
    #     correct_prediction = tf.equal(tf.argmax(pred, axis=2), tf.argmax(target, axis=2))
    #     all_correct = tf.reduce_min(tf.cast(correct_prediction, tf.float32), 1)
    #     acc = tf.reduce_mean(all_correct)
    #     return acc

    # calculates penalty if prediction is an item we already have, or an item with an item effect we already have
    def class_weighted_sm_ce_loss(self, y_pred, y_true, target_summ_items_sparse):
        # sess = tf.get_default_session()
        # y_pred = np.zeros((6,self.game_config["total_num_items"]), dtype=np.float32)
        # y_pred[0][ItemManager().lookup_by("name", "Ninja Tabi")["int"]] = 1.0
        # y_pred[1][ItemManager().lookup_by("name", "Ninja Tabi")["int"]] = 1.0
        # y_pred[2][ItemManager().lookup_by("name", "Boots of Speed")["int"]] = 1.0
        # y_pred[3][ItemManager().lookup_by("name", "Hexdrinker")["int"]] = 1.0
        # y_pred[4][ItemManager().lookup_by("name", "Stalker's Blade: Cinderhulk")["int"]] = 1.0
        # y_pred[5][ItemManager().lookup_by("name", "Trinity Force")["int"]] = 1.0
        # y_true = np.zeros((6,self.game_config["total_num_items"]), dtype=np.float32)
        # y_true[0][ItemManager().lookup_by("name", "The Dark Seal")["int"]] = 1.0
        # y_true[1][ItemManager().lookup_by("name", "The Dark Seal")["int"]] = 1.0
        # y_true[2][ItemManager().lookup_by("name", "The Dark Seal")["int"]] = 1.0
        # y_true[3][ItemManager().lookup_by("name", "The Dark Seal")["int"]] = 1.0
        # y_true[4][ItemManager().lookup_by("name", "The Dark Seal")["int"]] = 1.0
        # y_true[4][ItemManager().lookup_by("name", "The Dark Seal")["int"]] = 1.0
        # target_summ_items_sparse = np.array(
        #     [[ItemManager().lookup_by("name", "Sorcerer's Shoes")["int"], 91, 59, 17, 45, 40]
        #     ,[ItemManager().lookup_by("name", "Boots of Speed")["int"], 91, 59, 17, 45, 40]
        #     ,[ItemManager().lookup_by("name", "Ninja Tabi")["int"], 91, 59, 17, 45, 40]
        #     ,[ItemManager().lookup_by("name", "Hexdrinker")["int"], 91, 59, 17, 45, 40]
        #     ,[ItemManager().lookup_by("name", "Sunfire Cape")["int"], 91, 59, 17, 45, 40]
        #     ,[ItemManager().lookup_by("name", "The Black Cleaver")["int"], 91, 59, 17, 45, 40]]
        #     , dtype=np.float32)

        class_weights = tf.reduce_sum(tf.multiply(y_true, tf.constant(self.network_config[
                                                                          "class_weights"], dtype=tf.float32)), 1)
        # sparse_pred = tf.argmax(y_pred, axis=1)
        # items_by_int_list_sorted = sorted([artifact for artifact in ItemManager().get_ints().values()], key=lambda
        #     artifact: artifact["int"])
        # items_by_int_list_sorted_unique = [artifact["unique"] if "unique" in artifact else [] for artifact in
        #                             items_by_int_list_sorted]
        # items_by_int_list_sorted_unique = np.array(list(itertools.zip_longest(*items_by_int_list_sorted_unique,
        #                                                                       fillvalue=-1))).T
        # items_by_int = tf.constant(items_by_int_list_sorted_unique, dtype=tf.int32)
        # target_summ_items_sparse_flat = tf.reshape(target_summ_items_sparse, (-1,))
        # target_summ_item_effects = tf.gather(items_by_int, tf.cast(target_summ_items_sparse_flat, tf.int32))
        # pred_item_effects = tf.gather(items_by_int, tf.cast(sparse_pred, tf.int32))
        # pred_item_effects_one_hot = tf.one_hot(pred_item_effects, depth=self.game_config["total_num_items"])
        # pred_item_effects_k_hot = tf.reduce_sum(pred_item_effects_one_hot, axis=-2)
        #
        #
        # target_summ_item_effects_k_hot = tf.reduce_sum(tf.one_hot(target_summ_item_effects, depth=self.game_config[
        #     "total_num_items"]), axis=-2)
        # target_summ_item_effects_k_hot = tf.reshape(target_summ_item_effects_k_hot, (-1,self.game_config[
        #     "items_per_champ"], self.game_config["total_num_items"]))
        # pred_item_effects_k_hot = tf.tile(pred_item_effects_k_hot, multiples=[1,self.game_config["items_per_champ"]])
        # pred_item_effects_k_hot = tf.reshape(pred_item_effects_k_hot, (-1,self.game_config["items_per_champ"], self.game_config["total_num_items"]))
        #
        # # penalize if intersection and (not superset or is_equal)
        # # == (intersection and not superset) or (intersection and is_equal)
        #
        # new_and_old_item_has_effect_overlap = tf.math.logical_and(tf.cast(target_summ_item_effects_k_hot, tf.bool),
        #                                    tf.cast(pred_item_effects_k_hot, tf.bool))
        #
        # new_and_old_item_has_effect_overlap_sparse = tf.math.reduce_any(new_and_old_item_has_effect_overlap, axis=-1)
        #
        # new_item_is_superset = tf.equal(new_and_old_item_has_effect_overlap, tf.cast(target_summ_item_effects_k_hot, tf.bool))
        # new_item_is_superset_sparse = tf.math.reduce_all(new_item_is_superset, axis=-1)
        #
        # new_item_equal_old_item = tf.equal(tf.cast(target_summ_item_effects_k_hot, tf.bool),
        #                                    tf.cast(pred_item_effects_k_hot, tf.bool))
        # new_item_equal_old_item_sparse = tf.math.reduce_all(new_item_equal_old_item, axis=-1)
        #
        # bad_predictions_by_batch_index_by_item = tf.math.logical_or(tf.math.logical_and(
        #     new_and_old_item_has_effect_overlap_sparse, tf.math.logical_not(
        #     new_item_is_superset_sparse)), tf.math.logical_and(new_and_old_item_has_effect_overlap_sparse, new_item_equal_old_item_sparse))
        # bad_predictions_by_batch_index = tf.math.reduce_any(bad_predictions_by_batch_index_by_item, axis = -1)
        #
        #
        # class_weights = class_weights + tf.cast(bad_predictions_by_batch_index, tf.float32) * max(self.network_config["class_weights"])

        return tf.losses.softmax_cross_entropy(y_true, y_pred, weights=class_weights)


    def weighted_accuracy(self, preds, targets, input_):
        targets_sparse = tf.argmax(targets, axis=-1)
        preds_sparse = tf.argmax(preds, axis=-1)
        return NextItemNetwork.weighted_accuracy_static(preds_sparse, targets_sparse, self.network_config["class_weights"])


    @staticmethod
    def weighted_accuracy_static(preds_sparse, targets_sparse, class_weights):
        max_achievable_score = tf.reduce_sum(tf.gather(class_weights, targets_sparse))
        matching_preds_sparse = tf.boolean_mask(targets_sparse, tf.equal(targets_sparse, preds_sparse))
        actually_achieved_score = tf.reduce_sum(tf.gather(class_weights, matching_preds_sparse))
        return actually_achieved_score / max_achievable_score


class NextItemEarlyGameNetwork(NextItemNetwork):

    def __init__(self, my_champ_emb_scales=None, opp_champ_emb_scales=None):
        super().__init__(my_champ_emb_scales, opp_champ_emb_scales)



    def build(self):

        in_vec = input_data(shape=[None, 221], name='input')
        #  1 elements long
        pos = in_vec[:, 0]
        pos = tf.cast(pos, tf.int32)

        n = tf.shape(in_vec)[0]
        batch_index = tf.range(n)
        pos_index = tf.transpose([batch_index, pos], (1, 0))
        opp_index_5_offset = tf.transpose([batch_index, pos + self.game_config["champs_per_team"]], (1, 0))
        opp_index_no_offset = tf.transpose([batch_index, pos], (1, 0))

        # Make tensor of indices for the first dimension

        #  10 elements long
        champ_ints = in_vec[:, self.champs_start:self.champs_end]
        my_team_champ_ints = champ_ints[:, :5]
        opp_team_champ_ints = champ_ints[:, 5:]
        # champ_ints = dropout(champ_ints, 0.8)
        # this does not work since dropout scales inputs, hence embedding lookup fails after that.
        # 60 elements long
        item_ints = in_vec[:, self.items_start:self.items_end]
        cs = in_vec[:, self.cs_start:self.cs_end]
        neutral_cs = in_vec[:, self.neutral_cs_start:self.neutral_cs_end]
        lvl = in_vec[:, self.lvl_start:self.lvl_end]
        kda = in_vec[:, self.kda_start:self.kda_end]
        current_gold = in_vec[:, self.current_gold_start:self.current_gold_end]
        total_cs = cs + neutral_cs

        target_summ_current_gold = tf.expand_dims(tf.gather_nd(current_gold, pos_index), 1)
        target_summ_cs = tf.expand_dims(tf.gather_nd(total_cs, pos_index), 1)
        target_summ_kda = tf.gather_nd(tf.reshape(kda, (-1, self.game_config["champs_per_game"], 3)), pos_index)
        target_summ_lvl = tf.expand_dims(tf.gather_nd(lvl, pos_index), 1)

        # my_team_champ_embs = self.noise_embeddings(my_team_champ_ints, "my_champ_emb_scales",
        #                                                      "my_champ_embs", 1/4)
        # opp_team_champ_embs = self.noise_embeddings(opp_team_champ_ints, "opp_champ_emb_scales",
        #                                                   "opp_champ_embs", 1/4)
        # target_summ_champ_emb_dropout_flat = tf.gather_nd(my_team_champ_embs, pos_index)
        # opp_summ_champ_emb_dropout_flat = tf.gather_nd(opp_team_champ_embs, opp_index_no_offset)
        # opp_team_champ_embs_dropout_flat = tf.reshape(opp_team_champ_embs, (-1, 5*3))


        target_summ_champ_emb_dropout_flat, _ = self.get_champ_embeddings_v2(my_team_champ_ints,"my_champ_embs", 10, 11, pos_index, n)
        opp_summ_champ_emb_dropout_flat, opp_team_champ_embs_dropout_flat = self.get_champ_embeddings_v2(opp_team_champ_ints, "opp_champ_embs", 10, 11,
                                                                                                  opp_index_no_offset, n)

        items_by_champ = tf.reshape(item_ints, [-1, self.game_config["champs_per_game"], self.game_config[
            "items_per_champ"], 2])
        items_by_champ_flat = tf.reshape(items_by_champ, [-1])

        batch_indices = tf.reshape(tf.tile(tf.expand_dims(tf.range(n), 1), [1, self.game_config["champs_per_game"] *
                                                                            self.game_config["items_per_champ"]]),
                                   (-1,))
        champ_indices = tf.reshape(tf.tile(tf.tile(tf.expand_dims(tf.range(self.game_config["champs_per_game"]), 1), [1,
                                                                                                                      self.game_config[
                                                                                                                          "items_per_champ"]]),
                                           [n, 1]),
                                   (-1,))

        index_shift = tf.cast(tf.reshape(items_by_champ[:, :, :, 0] + 1, (-1,)), tf.int32)

        item_one_hot_indices = tf.cast(tf.transpose([batch_indices, champ_indices, index_shift], [1, 0]),
                                       tf.int64)

        items = tf.SparseTensor(indices=item_one_hot_indices, values=tf.reshape(items_by_champ[:, :, :, 1], (-1,)),
                                dense_shape=(n, self.game_config["champs_per_game"], self.game_config[
                                    "total_num_items"] + 1))
        items = tf.sparse.to_dense(items, validate_indices=False)
        items_by_champ_k_hot = items[:, :, 1:]

        target_summ_items_sparse = tf.gather_nd(items_by_champ, pos_index)
        target_summ_items = tf.gather_nd(items_by_champ_k_hot, pos_index)

        pos_one_hot = tf.one_hot(pos, depth=self.game_config["champs_per_team"])
        opp_kda = kda[:, 5 * 3:10 * 3]
        opp_kda = tf.reshape(opp_kda, (-1, 5, 3))
        opp_lvl = tf.expand_dims(lvl[:, 5:10], -1)
        opp_cs = tf.expand_dims(cs[:, 5:10], -1)
        opp_champ_items = items_by_champ_k_hot[:, 5:10]

        target_summ_kda_exp = tf.reshape(tf.tile(target_summ_kda, multiples=[1, 5]), (-1, 5, 3))
        target_summ_lvl_exp = tf.expand_dims(tf.tile(target_summ_lvl, multiples=[1, 5]), -1)
        target_summ_cs_exp = tf.expand_dims(tf.tile(target_summ_cs, multiples=[1, 5]), -1)
        kda_diff = opp_kda - target_summ_kda_exp
        lvl_diff = opp_lvl - target_summ_lvl_exp
        cs_diff = opp_cs - target_summ_cs_exp

        enemy_summ_strength_input = merge(
            [
                kda_diff,
                lvl_diff,
                cs_diff
            ], mode='concat', axis=2)
        enemy_summ_strength_input = tf.reshape(enemy_summ_strength_input, (-1, 5))
        # if bias=false this layer generates 0 values if kda diff, etc is 0. this causes null divison later because
        # the vector has no magnitude
        # enemy_summs_strength_output = fully_connected(enemy_summ_strength_input, 1, bias=True, activation='linear')

        # EDIT: above does not work with the valid_mag_idx = tf.reshape(tf.greater_equal(ets_magnitude, 1e-7), (-1,))
        # since it may dip below zero with negative weights.
        enemy_summs_strength_output = batch_normalization(fully_connected(enemy_summ_strength_input, 1, bias=False,
                                                                          activation='linear', regularizer="L2"))
        enemy_summs_strength_output = tf.reshape(enemy_summs_strength_output, (-1, 5))
        # enemy_summs_strength_output = tf.reshape(enemy_summs_strength_output, (-1, 5, 1))
        #
        # # ets_magnitude = tf.norm(enemy_team_strength, axis=1, keep_dims=True)
        # # this tends to cause nan errors because of div by 0
        # enemy_team_strengths = None
        # for dim in range(2, 12, 2):
        #     for norm_result in self.calc_enemy_team_strength(enemy_summs_strength_output, dim, champ_ints,
        #                                                      self.game_config["total_num_champs"], n):
        #         for res in norm_result:
        #             if enemy_team_strengths is not None:
        #                 enemy_team_strengths = tf.concat([enemy_team_strengths, res], axis=1)
        #             else:
        #                 enemy_team_strengths = res

        # enemy_team_strengths = [res for dim in range(2, 12, 2) for res
        #                         in self.calc_enemy_team_strength(enemy_summs_strength_output, dim, champ_ints,
        #                                                          total_num_champs, n)]

        # net = batch_normalization(fully_connected(enemy_team_strengths, 64, bias=False,
        #                                           activation='relu',  regularizer="L2"))
        # net = dropout(net, 0.85)
        # net = batch_normalization(fully_connected(net, 32, bias=False,
        #                                                                   activation='relu',  regularizer="L2"))
        # net = dropout(net, 0.9)
        # enemy_team_strengths_output = batch_normalization(fully_connected(net, 32, bias=False,
        #                                                                   activation='relu',  regularizer="L2"))

        # enemy_team_lane_input = merge(
        #     [
        #         self.build_team_convs(opp_champ_emb),
        #         # self.build_team_convs(opp_champ_emb_long),
        #         # self.build_team_convs(opp_champ_emb_short1),
        #         # self.build_team_convs(opp_champ_emb_short2),
        #         # opp_champs_k_hot,
        #         pos_embedded,
        #         pos_one_hot,
        #         opp_summ_champ_emb_short2,
        #         opp_summ_champ_emb_short1,
        #         opp_summ_champ_emb,
        #     ], mode='concat', axis=1)

        # net = batch_normalization(fully_connected(enemy_team_lane_input, 128, bias=False,
        #                                          activation='relu',  regularizer="L2"))
        # net = dropout(net, 0.85)
        # net = batch_normalization(fully_connected(net, 64, bias=False, activation='relu',  regularizer="L2"))
        # net = dropout(net, 0.9)
        # enemy_team_lane_output = batch_normalization(fully_connected(net, 32, bias=False,
        #                                                                   activation='relu',  regularizer="L2"))

        final_input_layer = merge(
            [
                target_summ_champ_emb_dropout_flat,
                opp_summ_champ_emb_dropout_flat,
                opp_team_champ_embs_dropout_flat,
                pos_one_hot,
                enemy_summs_strength_output,
                target_summ_current_gold,
                target_summ_items,
            ], mode='concat', axis=1)

        net = batch_normalization(fully_connected(final_input_layer, 256, bias=False, activation='relu',
                                                  regularizer="L2"))
        # net = dropout(net, 0.85)
        net = batch_normalization(fully_connected(net, 256, bias=False, activation='relu', regularizer="L2"))
        # net = dropout(net, 0.9)
        net = batch_normalization(fully_connected(net, 256, bias=False, activation='relu', regularizer="L2"))

        logits = fully_connected(net, self.game_config["total_num_items"], activation='linear')

        is_training = tflearn.get_training_mode()
        inference_output = tf.nn.softmax(logits)

        net = tf.cond(is_training, lambda: logits, lambda: inference_output)

        return regression_custom(net, target_summ_items=target_summ_items_sparse, optimizer='adam', to_one_hot=True,
                                 n_classes=self.game_config["total_num_items"],
                                 shuffle_batches=True,
                                 learning_rate=self.network_config["learning_rate"],
                                 loss=self.class_weighted_sm_ce_loss,
                                 name='target', metric=self.weighted_accuracy)


class NextItemLateGameNetwork(NextItemNetwork):

    def __init__(self, my_champ_emb_scales=None, opp_champ_emb_scales=None):
        super().__init__(my_champ_emb_scales, opp_champ_emb_scales)


    def build(self):
        in_vec = input_data(shape=[None, 221], name='input')
        #  1 elements long
        pos = in_vec[:, 0]
        pos = tf.cast(pos, tf.int32)

        n = tf.shape(in_vec)[0]
        batch_index = tf.range(n)
        pos_index = tf.transpose([batch_index, pos], (1, 0))
        opp_index_5_offset = tf.transpose([batch_index, pos + self.game_config["champs_per_team"]], (1, 0))
        opp_index_no_offset = tf.transpose([batch_index, pos], (1, 0))

        # Make tensor of indices for the first dimension

        #  10 elements long
        champ_ints = in_vec[:, self.champs_start:self.champs_end]
        my_team_champ_ints = champ_ints[:, :5]
        opp_team_champ_ints = champ_ints[:, 5:]

        # champ_ints = dropout(champ_ints, 0.8)
        # this does not work since dropout scales inputs, hence embedding lookup fails after that.
        # 60 elements long
        item_ints = in_vec[:, self.items_start:self.items_end]
        cs = in_vec[:, self.cs_start:self.cs_end]
        neutral_cs = in_vec[:, self.neutral_cs_start:self.neutral_cs_end]
        lvl = in_vec[:, self.lvl_start:self.lvl_end]
        kda = in_vec[:, self.kda_start:self.kda_end]
        current_gold = in_vec[:, self.current_gold_start:self.current_gold_end]
        total_cs = cs + neutral_cs

        target_summ_current_gold = tf.expand_dims(tf.gather_nd(current_gold, pos_index), 1)
        target_summ_cs = tf.expand_dims(tf.gather_nd(total_cs, pos_index), 1)
        target_summ_kda = tf.gather_nd(tf.reshape(kda, (-1, self.game_config["champs_per_game"], 3)), pos_index)
        target_summ_lvl = tf.expand_dims(tf.gather_nd(lvl, pos_index), 1)

        target_summ_champ_emb_dropout_flat, _ = self.get_champ_embeddings(my_team_champ_ints, "my_champ_embs",
                                                                          "my_champ_emb_scales", 2, 4, pos_index, n)
        opp_summ_champ_emb_dropout_flat, opp_team_champ_embs_dropout_flat = self.get_champ_embeddings(
            opp_team_champ_ints, "opp_champ_embs",
            "opp_champ_emb_scales", 3, 6,
            opp_index_no_offset, n)

        items_by_champ = tf.reshape(item_ints, [-1, self.game_config["champs_per_game"], self.game_config[
            "items_per_champ"], 2])
        items_by_champ_flat = tf.reshape(items_by_champ, [-1])

        batch_indices = tf.reshape(tf.tile(tf.expand_dims(tf.range(n), 1), [1, self.game_config["champs_per_game"] *
                                                                            self.game_config["items_per_champ"]]),
                                   (-1,))
        champ_indices = tf.reshape(tf.tile(tf.tile(tf.expand_dims(tf.range(self.game_config["champs_per_game"]), 1), [1,
                                                                                                                      self.game_config[
                                                                                                                          "items_per_champ"]]),
                                           [n, 1]),
                                   (-1,))

        index_shift = tf.cast(tf.reshape(items_by_champ[:, :, :, 0] + 1, (-1,)), tf.int32)

        item_one_hot_indices = tf.cast(tf.transpose([batch_indices, champ_indices, index_shift], [1, 0]),
                                       tf.int64)

        items = tf.SparseTensor(indices=item_one_hot_indices, values=tf.reshape(items_by_champ[:, :, :, 1], (-1,)),
                                dense_shape=(n, self.game_config["champs_per_game"], self.game_config[
                                    "total_num_items"] + 1))
        items = tf.sparse.to_dense(items, validate_indices=False)
        items_by_champ_k_hot = items[:, :, 1:]

        target_summ_items_sparse = tf.gather_nd(items_by_champ, pos_index)
        target_summ_items = tf.gather_nd(items_by_champ_k_hot, pos_index)

        opp_kda = kda[:, 5 * 3:10 * 3]
        opp_kda = tf.reshape(opp_kda, (-1, 5, 3))
        opp_lvl = tf.expand_dims(lvl[:, 5:10], -1)
        opp_cs = tf.expand_dims(cs[:, 5:10], -1)

        target_summ_kda_exp = tf.reshape(tf.tile(target_summ_kda, multiples=[1, 5]), (-1, 5, 3))
        target_summ_lvl_exp = tf.expand_dims(tf.tile(target_summ_lvl, multiples=[1, 5]), -1)
        target_summ_cs_exp = tf.expand_dims(tf.tile(target_summ_cs, multiples=[1, 5]), -1)
        kda_diff = opp_kda - target_summ_kda_exp
        lvl_diff = opp_lvl - target_summ_lvl_exp
        cs_diff = opp_cs - target_summ_cs_exp

        enemy_summ_strength_input = merge(
            [
                kda_diff,
                lvl_diff,
                cs_diff
            ], mode='concat', axis=2)
        enemy_summ_strength_input = tf.reshape(enemy_summ_strength_input, (-1, 5))
        # if bias=false this layer generates 0 values if kda diff, etc is 0. this causes null divison later because
        # the vector has no magnitude
        # enemy_summs_strength_output = fully_connected(enemy_summ_strength_input, 1, bias=True, activation='linear')

        # EDIT: above does not work with the valid_mag_idx = tf.reshape(tf.greater_equal(ets_magnitude, 1e-7), (-1,))
        # since it may dip below zero with negative weights.
        #
        # enemy_summs_strength_output = batch_normalization(fully_connected(enemy_summ_strength_input, 1, bias=False,
        #                                                                   activation='relu'))
        # enemy_summs_strength_output = tf.reshape(enemy_summs_strength_output, (-1, 5, 1))
        #
        # enemy_team_strengths = None
        # for dim in range(2, 12, 2):
        #     for norm_result in self.calc_enemy_team_strength(enemy_summs_strength_output, dim, champ_ints,
        #                                                      total_num_champs, n):
        #         for res in norm_result:
        #             if enemy_team_strengths is not None:
        #                 enemy_team_strengths = tf.concat([enemy_team_strengths, res], axis=1)
        #             else:
        #                 enemy_team_strengths = res

        enemy_summs_strength_output = batch_normalization(fully_connected(enemy_summ_strength_input, 1, bias=False,
                                                                          activation='linear', regularizer="L2"))
        enemy_summs_strength_output = tf.reshape(enemy_summs_strength_output, (-1, 5))
        nonstarter_input_layer = merge(
            [
                enemy_summs_strength_output,
                opp_team_champ_embs_dropout_flat,
                target_summ_champ_emb_dropout_flat,
                target_summ_items,
                target_summ_current_gold
            ], mode='concat', axis=1)

        net = batch_normalization(fully_connected(nonstarter_input_layer, 256, bias=False,
                                                  activation='relu',
                                                  regularizer="L2"))
        net = dropout(net, 0.8)
        net = batch_normalization(fully_connected(net, 128, bias=False,
                                                  activation='relu',
                                                  regularizer="L2"))
        net = dropout(net, 0.9)
        net = batch_normalization(fully_connected(net, 128, bias=False,
                                                  activation='relu',
                                                  regularizer="L2"))

        logits = fully_connected(net, self.game_config["total_num_items"], activation='linear')

        is_training = tflearn.get_training_mode()
        inference_output = tf.nn.softmax(logits)

        net = tf.cond(is_training, lambda: logits, lambda: inference_output)

        return regression_custom(net, target_summ_items=target_summ_items_sparse, optimizer='adam', to_one_hot=True,
                                 n_classes=self.game_config["total_num_items"],
                                 shuffle_batches=True,
                                 learning_rate=self.network_config["learning_rate"],
                                 loss=self.class_weighted_sm_ce_loss,
                                 name='target', metric=self.weighted_accuracy)


class NextItemStarterNetwork(NextItemNetwork):

    def __init__(self, my_champ_emb_scales=None, opp_champ_emb_scales=None):
        super().__init__(my_champ_emb_scales, opp_champ_emb_scales)


    def build(self):
        in_vec = input_data(shape=[None, 221], name='input')
        #  1 elements long
        pos = in_vec[:, 0]
        pos = tf.cast(pos, tf.int32)

        n = tf.shape(in_vec)[0]
        batch_index = tf.range(n)
        pos_index = tf.transpose([batch_index, pos], (1, 0))
        opp_index_5_offset = tf.transpose([batch_index, pos + self.game_config["champs_per_team"]], (1, 0))
        opp_index_no_offset = tf.transpose([batch_index, pos], (1, 0))

        champ_ints = in_vec[:, self.champs_start:self.champs_end]
        my_team_champ_ints = champ_ints[:, :5]
        opp_team_champ_ints = champ_ints[:, 5:]

        target_summ_champ_emb_dropout_flat, _ = self.get_champ_embeddings_v2(my_team_champ_ints, "my_champ_embs",
                                                                           [0.05], pos_index, n, 1.0)
        opp_summ_champ_emb_dropout_flat, _ = self.get_champ_embeddings_v2(
            opp_team_champ_ints, "opp_champ_embs", [0.1], opp_index_no_offset, n, 1.0)

        pos_one_hot = tf.one_hot(pos, depth=self.game_config["champs_per_team"])

        starter_input_layer = merge(
            [
                opp_summ_champ_emb_dropout_flat,
                pos_one_hot,
                # pos_embedded,
                # pos_embedded_short,
                # target_summ_champ,
                # target_summ_opp
                # target_summ_champ_emb,
                # target_summ_champ_emb_short1,
                target_summ_champ_emb_dropout_flat
            ], mode='concat', axis=1)

        net = batch_normalization(fully_connected(starter_input_layer, 16, bias=False,
                                                  activation='relu',
                                                  regularizer="L2"))
        # net = batch_normalization(fully_connected(net, 16, bias=False,
        #                                           activation='relu',
        #                                           regularizer="L2"))
        logits = fully_connected(net, self.game_config["total_num_items"], activation='linear')

        is_training = tflearn.get_training_mode()
        inference_output = tf.nn.softmax(logits)

        net = tf.cond(is_training, lambda: logits, lambda: inference_output)

        return regression(net, optimizer='adam', to_one_hot=True,
                          n_classes=self.game_config["total_num_items"],
                          shuffle_batches=True,
                          learning_rate=self.network_config["learning_rate"],
                          loss='softmax_categorical_crossentropy',
                          name='target')


class NextItemFirstItemNetwork(NextItemNetwork):

    def __init__(self, my_champ_emb_scales=None, opp_champ_emb_scales=None):
        super().__init__(my_champ_emb_scales, opp_champ_emb_scales)


    def build(self):
        in_vec = input_data(shape=[None, 221], name='input')
        #  1 elements long
        pos = in_vec[:, 0]
        pos = tf.cast(pos, tf.int32)

        n = tf.shape(in_vec)[0]
        batch_index = tf.range(n)
        pos_index = tf.transpose([batch_index, pos], (1, 0))
        opp_index_5_offset = tf.transpose([batch_index, pos + self.game_config["champs_per_team"]], (1, 0))
        opp_index_no_offset = tf.transpose([batch_index, pos], (1, 0))

        # Make tensor of indices for the first dimension

        #  10 elements long
        champ_ints = in_vec[:, self.champs_start:self.champs_end]
        my_team_champ_ints = champ_ints[:, :5]
        opp_team_champ_ints = champ_ints[:, 5:]

        # champ_ints = dropout(champ_ints, 0.8)
        # this does not work since dropout scales inputs, hence embedding lookup fails after that.
        # 60 elements long
        item_ints = in_vec[:, self.items_start:self.items_end]
        cs = in_vec[:, self.cs_start:self.cs_end]
        neutral_cs = in_vec[:, self.neutral_cs_start:self.neutral_cs_end]
        lvl = in_vec[:, self.lvl_start:self.lvl_end]
        kda = in_vec[:, self.kda_start:self.kda_end]
        current_gold = in_vec[:, self.current_gold_start:self.current_gold_end]
        total_cs = cs + neutral_cs

        target_summ_current_gold = tf.expand_dims(tf.gather_nd(current_gold, pos_index), 1)

        target_summ_champ_emb_dropout_flat, _ = self.get_champ_embeddings_v2(my_team_champ_ints, "my_champ_embs",
                                                                             [0.15], pos_index, n, 1.0)
        opp_summ_champ_emb_dropout_flat, opp_team_champ_embs_dropout_flat = self.get_champ_embeddings_v2(
            opp_team_champ_ints, "opp_champ_embs", [0.01], opp_index_no_offset, n, 1.0)


        items_by_champ = tf.reshape(item_ints, [-1, self.game_config["champs_per_game"], self.game_config[
            "items_per_champ"], 2])
        items_by_champ_flat = tf.reshape(items_by_champ, [-1])

        batch_indices = tf.reshape(tf.tile(tf.expand_dims(tf.range(n), 1), [1, self.game_config["champs_per_game"] *
                                                                            self.game_config["items_per_champ"]]),
                                   (-1,))
        champ_indices = tf.reshape(tf.tile(tf.tile(tf.expand_dims(tf.range(self.game_config["champs_per_game"]), 1), [1,
                                                                                                                      self.game_config[
                                                                                                                          "items_per_champ"]]),
                                           [n, 1]),
                                   (-1,))

        index_shift = tf.cast(tf.reshape(items_by_champ[:, :, :, 0] + 1, (-1,)), tf.int32)

        item_one_hot_indices = tf.cast(tf.transpose([batch_indices, champ_indices, index_shift], [1, 0]),
                                       tf.int64)

        items = tf.SparseTensor(indices=item_one_hot_indices, values=tf.reshape(items_by_champ[:, :, :, 1], (-1,)),
                                dense_shape=(n, self.game_config["champs_per_game"], self.game_config[
                                    "total_num_items"] + 1))
        items = tf.sparse.to_dense(items, validate_indices=False)
        items_by_champ_k_hot = items[:, :, 1:]

        target_summ_items_sparse = tf.gather_nd(items_by_champ, pos_index)
        target_summ_items = tf.gather_nd(items_by_champ_k_hot, pos_index)

        # starter_ints = list(ItemManager().get_starter_ints())
        # starter_one_hots = tf.reduce_sum(tf.one_hot(starter_ints, depth=self.game_config["total_num_items"]), axis=0)
        # target_summ_items = target_summ_items * tf.cast(
        #     tf.reshape(tf.tile(tf.logical_not(tf.cast(starter_one_hots, tf.bool)),
        #                        multiples=[n]), (n, -1)), tf.float32)
        opp_summ_items = tf.gather_nd(items_by_champ_k_hot, opp_index_5_offset)

        pos_one_hot = tf.one_hot(pos, depth=self.game_config["champs_per_team"])

        high_prio_inputs = merge(
            [
                pos_one_hot,
                # target_summ_items,
                # target_summ_current_gold,
                opp_summ_champ_emb_dropout_flat,
                opp_team_champ_embs_dropout_flat,
                target_summ_champ_emb_dropout_flat
            ], mode='concat', axis=1)

        net = batch_normalization(fully_connected(high_prio_inputs, 128, bias=False,
                                                  activation='relu',
                                                  regularizer="L2"))
        # net = dropout(net, 0.9)
        net = batch_normalization(fully_connected(net, 128, bias=False,
                                                  activation='relu',
                                                  regularizer="L2"))

        logits = fully_connected(net, self.game_config["total_num_items"], activation='linear')

        is_training = tflearn.get_training_mode()
        inference_output = tf.nn.softmax(logits)

        net = tf.cond(is_training, lambda: logits, lambda: inference_output)

        return regression_custom(net, target_summ_items=target_summ_items_sparse, optimizer='adam', to_one_hot=True,
                                 n_classes=self.game_config["total_num_items"],
                                 shuffle_batches=True,
                                 learning_rate=self.network_config["learning_rate"],
                                 loss=self.class_weighted_sm_ce_loss,
                                 name='target', metric=self.weighted_accuracy)


# class ChampEmbeddings:
#
#     def __init__(self):
#         super().__init__()
#
#         self.network_config = \
#             {
#                 "learning_rate": 0.00025,
#                 "champ_emb_dim": 3,
#                 "all_items_emb_dim": 4,
#                 "champ_all_items_emb_dim": 6,
#                 "class_weights": [1]
#             }
#         self.game_config = \
#             {
#                 "champs_per_game": game_constants.CHAMPS_PER_GAME,
#                 "champs_per_team": game_constants.CHAMPS_PER_TEAM,
#                 "total_num_champs": ChampManager().get_num("int"),
#
#                 "total_num_items": ItemManager().get_num("int"),
#                 "items_per_champ": game_constants.MAX_ITEMS_PER_CHAMP
#             }
#
#
#     def build(self):
#         total_num_champs = self.game_config["total_num_champs"]
#         total_num_items = ItemManager().get_num("int")
#
#         learning_rate = self.network_config["learning_rate"]
#
#         in_vec = input_data(shape=[None, 1+total_num_items], name='input')
#         champ_ints = in_vec[:, 0]
#         items = in_vec[:, 1:]
#
#         champs_embedded_short1 = embedding(tf.reshape(champ_ints, (-1, 1)), input_dim=total_num_champs, output_dim=3,
#                                            reuse=tf.AUTO_REUSE,
#                                            scope="champs_embedded_short1")
#         champs_embedded_short1 = tf.reshape(champs_embedded_short1, (-1, 3))
#         final_input_layer = merge(
#             [
#                 champs_embedded_short1,
#                 items
#             ], mode='concat', axis=1)
#
#         net = final_input_layer
#         net = fully_connected(net, 128, activation='relu', regularizer="L2")
#         net = fully_connected(net, 1, activation='sigmoid')
#
#         return regression(net, optimizer='adam',
#                                  shuffle_batches=True,
#                                  learning_rate=learning_rate,
#                                  loss='binary_crossentropy',
#                                  name='target', metric=self.bin_acc)
#
#     @staticmethod
#     def bin_acc(preds, targets, input_):
#         preds = tf.round(preds)
#         correct_prediction = tf.equal(preds, targets)
#         all_correct = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), 1)
#         acc = tf.reduce_mean(all_correct)
#
#         return acc
#
#
# class ChampEmbeddings2:
#
#     def __init__(self):
#         super().__init__()
#
#         self.network_config = \
#             {
#                 "learning_rate": 0.00001,
#                 "champ_emb_dim": 3,
#                 "all_items_emb_dim": 4,
#                 "champ_all_items_emb_dim": 6,
#                 "class_weights": [1]
#             }
#         self.game_config = \
#             {
#                 "champs_per_game": game_constants.CHAMPS_PER_GAME,
#                 "champs_per_team": game_constants.CHAMPS_PER_TEAM,
#                 "total_num_champs": ChampManager().get_num("int"),
#
#                 "total_num_items": ItemManager().get_num("int"),
#                 "items_per_champ": game_constants.MAX_ITEMS_PER_CHAMP
#             }
#
#
#     def build(self):
#         total_num_champs = self.game_config["total_num_champs"]
#         total_num_items = ItemManager().get_num("int")
#
#         in_vec = input_data(shape=[None, total_num_items], name='input')
#         encoder = tflearn.fully_connected(in_vec, 64)
#         encoder = tflearn.fully_connected(encoder, 3, name="my_embedding")
#         decoder = tflearn.fully_connected(encoder, 64)
#         decoder = tflearn.fully_connected(decoder, total_num_items)
#
#         # Regression, with mean square error
#         return tflearn.regression(decoder, optimizer='adam', learning_rate=0.001,
#                                  loss='mean_square', metric=None)
#


class ChampEmbeddings:

    def __init__(self):
        super().__init__()

        self.network_config = \
            {
                "learning_rate": 0.001,
                "champ_emb_dim": 3,
                "all_items_emb_dim": 4,
                "champ_all_items_emb_dim": 6,
                "class_weights": [1]
            }
        self.game_config = \
            {
                "champs_per_game": game_constants.CHAMPS_PER_GAME,
                "champs_per_team": game_constants.CHAMPS_PER_TEAM,
                "total_num_champs": ChampManager().get_num("int"),

                "total_num_items": ItemManager().get_num("int"),
                "items_per_champ": game_constants.MAX_ITEMS_PER_CHAMP
            }


    def build(self):
        total_num_champs = self.game_config["total_num_champs"]
        total_num_items = ItemManager().get_num("int")

        encoder = input_data(shape=[None, total_num_items], name='input')
        encoder = dropout(encoder, 0.8)
        encoder = tflearn.fully_connected(encoder, 64)
        encoder = dropout(encoder, 0.9)
        encoder = tflearn.fully_connected(encoder, 3, name="my_embedding", regularizer="L2")
        decoder = tflearn.fully_connected(encoder, 64)
        decoder = tflearn.fully_connected(decoder, total_num_champs)

        is_training = tflearn.get_training_mode()
        inference_output = tf.nn.softmax(decoder)

        net = tf.cond(is_training, lambda: decoder, lambda: inference_output)

        # Regression, with mean square error
        return regression_custom(net, optimizer='adam', to_one_hot=True,
                                 n_classes=total_num_champs,
                                 shuffle_batches=True,
                                 learning_rate=self.network_config["learning_rate"],
                                 loss=self.class_weighted_sm_ce_loss,
                                 name='target')

    def class_weighted_sm_ce_loss(self, y_pred, y_true, target_summ_items_sparse):

        class_weights = tf.reduce_sum(tf.multiply(y_true, tf.constant(self.network_config[
                                                                          "class_weights"], dtype=tf.float32)), 1)

        return tf.losses.softmax_cross_entropy(y_true, y_pred, weights=class_weights)
