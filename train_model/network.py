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
from tflearn.initializations import variance_scaling

from constants import game_constants
from constants.ui_constants import ResConverter
from utils.artifact_manager import ChampManager, ItemManager, SimpleManager
from train_model.input_vector import Input


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


class LolNetwork(Network):

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


    def calc_team_diff(self, vec):
        result = tf.expand_dims(vec[:, self.game_config["champs_per_team"]:self.game_config["champs_per_game"]], -1) - \
                 tf.expand_dims(vec[:, :self.game_config["champs_per_team"]], -1)
        result = tf.reshape(result, (-1, 5))
        return result




class WinPredNetwork(LolNetwork):

    def  __init__(self, network_config=None):
        super().__init__()
        if network_config:
            self.network_config = network_config


    def calc_noise(self, flat_in_vec, noise_name):
        batch_len = tf.shape(flat_in_vec)[0]
        noise_distrib = tf.distributions.Normal(loc=tf.cast(tf.tile([0.], [batch_len]), tf.float32),
                                                scale=tf.cast(self.network_config["noise"][noise_name], tf.float32))
        noise = noise_distrib.sample([1])
        return noise


    def apply_noise(self, in_vec, noise_name):
        if self.network_config["noise"][noise_name] == 0:
            return in_vec
        orig_shape = tf.shape(in_vec)
        flattened_vec = tf.reshape(in_vec, (-1,))
        noise = self.calc_noise(flattened_vec, noise_name)
        noise = tf.identity(noise, "noise")
        noised_vec = flattened_vec + noise
        noised_vec = tf.clip_by_value(noised_vec, game_constants.min_clip_scaled[noise_name],
                                      game_constants.max_clip_scaled[noise_name])
        in_vec = tf.reshape(noised_vec, orig_shape)
        return in_vec


    def apply_noise_ints(self, ints, noise_name):
        if self.network_config["noise"][noise_name] == 0:
            return ints
        ints_flat = tf.reshape(ints, (-1,))
        draws = tf.random.uniform(shape=[tf.shape(ints_flat)[0]], maxval=1.0, minval=0.0)
        apply_indices = self.network_config["noise"][noise_name] > draws
        not_apply_indices = self.network_config["noise"][noise_name] <= draws
        num_indices = tf.reduce_sum(tf.cast(apply_indices, tf.int32))
        rand_champ_ints = tf.random.uniform(shape=[num_indices], maxval=self.game_config["total_num_champs"], minval=0,
                                            dtype=tf.int32)
        noised_ints = tf.scatter_nd(tf.cast(tf.where(apply_indices), tf.int32), rand_champ_ints, tf.shape(ints_flat))
        regular_ints = tf.scatter_nd(tf.cast(tf.where(not_apply_indices), tf.int32), tf.cast(ints_flat[not_apply_indices],
                                                                                       tf.int32),
                               tf.shape(ints_flat))
        result = tf.cast(regular_ints + noised_ints, tf.int32)
        result = tf.reshape(result, (-1, self.game_config["champs_per_game"]))

        return result


    def apply_noise_flip_one_hot(self, in_vec, noise_name):
        if self.network_config["noise"][noise_name] == 0:
            return in_vec
        draws = tf.random.uniform(shape=[tf.shape(in_vec)[0]], maxval=1.0, minval=0.0)
        apply_indices = self.network_config["noise"][noise_name] > draws
        not_apply_indices = self.network_config["noise"][noise_name] <= draws
        num_indices = tf.reduce_sum(tf.cast(apply_indices, tf.int32))
        outcomes = tf.tile([[1,0], [0,1], [0,0]], [(num_indices+2)//3, 1])
        outcomes = outcomes[:num_indices]
        outcomes = tf.random.shuffle(outcomes)
        outcomes = tf.scatter_nd(tf.cast(tf.where(apply_indices), tf.int32), outcomes, tf.shape(in_vec))
        in_vec = tf.scatter_nd(tf.cast(tf.where(not_apply_indices), tf.int32), tf.cast(in_vec[not_apply_indices],
                                                                                       tf.int32),
                               tf.shape(in_vec))

        return tf.cast(in_vec + outcomes, tf.float32)


    def build(self):
        in_vec = input_data(shape=[None, Input.len], name='input')
        #  1 elements long
        pos = in_vec[:, 0]
        pos = tf.cast(pos, tf.int32)

        n = tf.shape(in_vec)[0]
        batch_index = tf.range(n)

        champ_ints = tf.cast(in_vec[:, Input.indices["start"]["champs"]:Input.indices["end"]["champs"]], tf.int32)
        cs = in_vec[:, Input.indices["start"]["cs"]:Input.indices["end"]["cs"]]
        lvl = in_vec[:, Input.indices["start"]["lvl"]:Input.indices["end"]["lvl"]]
        kills = in_vec[:, Input.indices["start"]["kills"]:Input.indices["end"]["kills"]]
        deaths = in_vec[:, Input.indices["start"]["deaths"]:Input.indices["end"]["deaths"]]
        assists = in_vec[:, Input.indices["start"]["assists"]:Input.indices["end"]["assists"]]

        total_gold = in_vec[:, Input.indices["start"]["total_gold"]:Input.indices["end"]["total_gold"]]
        baron = in_vec[:, Input.indices["start"]["baron"]:Input.indices["end"]["baron"]]
        elder = in_vec[:, Input.indices["start"]["elder"]:Input.indices["end"]["elder"]]
        dragons_killed = in_vec[:, Input.indices["start"]["dragons_killed"]:Input.indices["end"]["dragons_killed"]]
        dragon_soul_type = in_vec[:,Input.indices["start"]["dragon_soul_type"]:Input.indices["end"]["dragon_soul_type"]]
        turrets_destroyed = in_vec[:, Input.indices["start"]["turrets_destroyed"]:Input.indices["end"]["turrets_destroyed"]]
        blue_side = in_vec[:, Input.indices["start"]["blue_side"]:Input.indices["end"]["blue_side"]]

        dragons_killed = tf.identity(dragons_killed, name='dragons_killed')
        dragons_killed = self.apply_noise(dragons_killed, "dragons_killed")
        dragons_killed = tf.identity(dragons_killed, name='dragons_killed_noised')
        dragons_killed_sum = tf.reduce_sum(tf.reshape(dragons_killed, (-1, 2, 4)), axis=2)


        dragon_soul_obtained = tf.reduce_sum(tf.reshape(dragon_soul_type, (-1,2,4)), axis=2)



        kills = self.apply_noise(kills, "kills")
        deaths = self.apply_noise(deaths, "deaths")
        assists = self.apply_noise(assists, "assists")


        cs = tf.identity(cs, name='cs')
        cs = self.apply_noise(cs, "cs")
        cs = tf.identity(cs, name='cs_noised')
        lvl = tf.identity(lvl, name='lvl')
        lvl = self.apply_noise(lvl, "lvl")
        lvl = tf.identity(lvl, name='lvl_noised')
        total_gold = tf.identity(total_gold, name='total_gold')
        total_gold = self.apply_noise(total_gold, "total_gold")
        total_gold = tf.identity(total_gold, name='total_gold_noised')

        turrets_destroyed = tf.identity(turrets_destroyed, name='turrets_destroyed')
        turrets_destroyed = self.apply_noise(turrets_destroyed, "turrets_destroyed")
        turrets_destroyed = tf.identity(turrets_destroyed, name='turrets_destroyed_noised')

        champ_ints = tf.identity(champ_ints, name='champs')
        champ_ints = self.apply_noise_ints(champ_ints, "champs")
        champ_ints = tf.identity(champ_ints, name='champs_noised')

        all_champs_one_hot = tf.one_hot(tf.cast(champ_ints, tf.int32), depth=self.game_config["total_num_champs"])
        all_champs_one_hot = tf.reshape(all_champs_one_hot, (-1, self.game_config["champs_per_game"], self.game_config["total_num_champs"]))

        baron = tf.identity(baron, name='baron')
        baron = self.apply_noise_flip_one_hot(baron, "baron")
        baron = tf.identity(baron, name='baron_noised')
        elder = tf.identity(elder, name='elder')
        elder = self.apply_noise_flip_one_hot(elder, "elder")
        elder = tf.identity(elder, name='elder_noised')

        blue_side = tf.identity(blue_side, name='blue_side')
        blue_side = self.apply_noise_flip_one_hot(blue_side, "blue_side")
        blue_side = tf.identity(blue_side, name='blue_side')

        kd = kills-deaths


        kills_diff = self.calc_team_diff(kills)
        deaths_diff = self.calc_team_diff(deaths)
        assists_diff = self.calc_team_diff(assists)
        lvl_diff = self.calc_team_diff(lvl)
        cs_diff = self.calc_team_diff(cs)
        total_gold_diff = self.calc_team_diff(total_gold)

        team_kills_diff = tf.reduce_sum(kills_diff, keep_dims=True, axis=1)
        team_gold_diff = tf.reduce_sum(total_gold_diff, keep_dims=True, axis=1)

        team1_champs_one_hot = all_champs_one_hot[:, :self.game_config["champs_per_team"]]
        team2_champs_one_hot = all_champs_one_hot[:, self.game_config["champs_per_team"]:]
        if self.network_config["champ_dropout"] > 0:
            team1_champs_one_hot = dropout(team1_champs_one_hot, self.network_config["champ_dropout"],
                                       noise_shape=[n, self.game_config["champs_per_team"], 1])
            team2_champs_one_hot = dropout(team2_champs_one_hot, self.network_config["champ_dropout"],
                                       noise_shape=[n, self.game_config["champs_per_team"], 1])

        team1_champs_one_hot = tf.reshape(team1_champs_one_hot, (-1, self.game_config["champs_per_team"] *
                                                                 self.game_config["total_num_champs"]))
        team2_champs_one_hot = tf.reshape(team2_champs_one_hot, (-1, self.game_config["champs_per_team"] *
                                                                 self.game_config["total_num_champs"]))

        stats_layer = merge(
            [
                # total_gold,
                total_gold_diff,
                team_gold_diff,
                # team1_total_kills,
                # team2_total_kills,
                team_kills_diff,
                # kd,
                # kills_diff,
                # deaths_diff,
                # assists_diff,
                # lvl_diff,
                # cs_diff,
                # kda,
                lvl,
                # total_cs,
                baron,
                elder,
                dragons_killed,
                dragons_killed_sum,
                dragon_soul_obtained,
                dragon_soul_type,
                turrets_destroyed,
                blue_side
            ], mode='concat', axis=1)
        stats_layer = tf.identity(stats_layer, "stats")


        if self.network_config["stats_dropout"] > 0:
            stats_layer = dropout(stats_layer, self.network_config["stats_dropout"])


        stats_layer = tf.identity(stats_layer, "stats_dropout")

        final_input_layer = merge(
            [
                team1_champs_one_hot,
                team2_champs_one_hot,
                stats_layer
            ], mode='concat', axis=1)

        net = final_input_layer

        net = batch_normalization(fully_connected(net, 512, bias=False, activation='relu',
                                                  weights_init=variance_scaling(uniform=True)))
        # net = dropout(net, self.stats_dropout)
        net = batch_normalization(fully_connected(net, 128, bias=False, activation='relu',
                                                  weights_init=variance_scaling(uniform=True)))
        # net = dropout(net, self.stats_dropout)
        net = batch_normalization(fully_connected(net, 32, bias=False, activation='relu',
                                                  weights_init=variance_scaling(uniform=True)))
        # net = dropout(net, self.stats_dropout)
        net = batch_normalization(fully_connected(net, 8, bias=False, activation='relu',
                                                  weights_init=variance_scaling(uniform=True)))

        net = fully_connected(net, 1, weights_init="xavier", activation='tanh', name="final_output")
        net = (net+1)/2

        return regression(net, optimizer='adam',
                                 shuffle_batches=True,
                                 learning_rate=self.network_config["learning_rate"],
                                 loss='binary_crossentropy',
                                 name='target', metric=self.bin_acc)


    @staticmethod
    def bin_acc(preds, targets, input_):
        preds = tf.reshape(preds, (-1,))
        preds = tf.identity(preds, name="preds")
        targets = tf.reshape(targets, (-1,))
        targets = tf.identity(targets, name="targets")
        preds = tf.round(preds)
        correct_prediction = tf.equal(preds, targets)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return acc

    #sigmoid is not a probabiliti density function and shouldn't be interpreted as probabilities
    # @staticmethod
    # def bin_acc(preds, targets, input_):
    #     preds = tf.reshape(preds, (-1,))
    #     targets = tf.reshape(targets, (-1,))
    #     error = tf.abs(targets - preds)
    #     score = 1-error
    #     acc = tf.reduce_mean(score)
    #
    #     return acc


class WinPredNetworkInit(LolNetwork):

    def build(self):

        in_vec = input_data(shape=[None, Input.len], name='input')
        n = tf.shape(in_vec)[0]
        champ_ints = in_vec[:, Input.indices["start"]["champs"]:Input.indices["end"]["champs"]]
        first_team_has_blue_side = in_vec[:, Input.indices["start"]["blue_side"]:Input.indices["end"]["blue_side"]]


        all_champs_one_hot = tf.one_hot(tf.cast(champ_ints, tf.int32), depth=self.game_config["total_num_champs"])
        # all_champs_one_hot = dropout(all_champs_one_hot, 0.9, noise_shape=[n, self.game_config["champs_per_game"], 1])
        all_champs_one_hot = tf.reshape(all_champs_one_hot, (-1, self.game_config["total_num_champs"] *
                                                             self.game_config["champs_per_game"]))

        final_input_layer = merge(
            [
                all_champs_one_hot,
                first_team_has_blue_side
            ], mode='concat', axis=1)

        net = batch_normalization(fully_connected(final_input_layer, 64, bias=False, activation='relu',
                                                  regularizer="L2"))
        # net = dropout(net, 0.85)
        # net = batch_normalization(fully_connected(net, 16, bias=False, activation='relu', regularizer="L2"))
        # # net = dropout(net, 0.9)
        # net = batch_normalization(fully_connected(net, 32, bias=False, activation='relu', regularizer="L2"))
        net = fully_connected(final_input_layer, 1, activation='sigmoid')

        return regression(net, optimizer='adam',
                                 shuffle_batches=True,
                                 learning_rate=self.network_config["learning_rate"],
                                 loss='binary_crossentropy',
                                 name='target', metric=self.bin_acc)


    @staticmethod
    def bin_acc(preds, targets, input_):
        preds = tf.reshape(preds, (-1,))
        targets = tf.reshape(targets, (-1,))
        preds = tf.round(preds)
        correct_prediction = tf.equal(preds, targets)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return acc
    #
    # @staticmethod
    # def bin_acc(preds, targets, input_):
    #     preds = tf.reshape(preds, (-1,))
    #     targets = tf.reshape(targets, (-1,))
    #     error = tf.abs(targets - preds)
    #     score = 1-error
    #     acc = tf.reduce_mean(score)
    #
    #     return acc



class NextItemNetwork(LolNetwork):

    @abstractmethod
    def build(self):
        pass

    def calc_diff_from_target_summ(self, vec, pos_index):
        target_summ_vec = tf.expand_dims(tf.gather_nd(vec, pos_index), 1)
        opp_vec = tf.expand_dims(vec[:, 5:10], -1)
        target_summ_vec_exp = tf.expand_dims(tf.tile(target_summ_vec, multiples=[1, 5]), -1)
        return target_summ_vec_exp - opp_vec

    def get_items(self, n, items_by_champ):
        batch_indices = tf.reshape(tf.tile(tf.expand_dims(tf.range(n), 1), [1, self.game_config["champs_per_game"] *
                                                                            self.game_config["items_per_champ"]]),(-1,))
        champ_indices = tf.reshape(tf.tile(tf.tile(tf.expand_dims(tf.range(self.game_config["champs_per_game"]), 1),
                            [1,self.game_config["items_per_champ"]]),[n, 1]),(-1,))
        index_shift = tf.cast(tf.reshape(items_by_champ[:, :, :, 0] + 1, (-1,)), tf.int32)
        item_one_hot_indices = tf.cast(tf.transpose([batch_indices, champ_indices, index_shift], [1, 0]),
                                       tf.int64)
        items = tf.SparseTensor(indices=item_one_hot_indices, values=tf.reshape(items_by_champ[:, :, :, 1], (-1,)),
                                dense_shape=(n, self.game_config["champs_per_game"], self.game_config[
                                    "total_num_items"] + 1))
        items = tf.sparse.to_dense(items, validate_indices=False)
        return tf.cast(items, tf.float32)

    def final_layer(self, net):
        logits = fully_connected(net, self.game_config["total_num_items"], activation='linear')
        is_training = tflearn.get_training_mode()
        inference_output = tf.nn.softmax(logits)
        net = tf.cond(is_training, lambda: logits, lambda: inference_output)
        return net


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
    def class_weighted_sm_ce_loss(self, y_pred, y_true, target_summ_items_sparse=None, original_input=None):

        class_weights = tf.reduce_sum(tf.multiply(y_true, tf.constant(self.network_config[
                                                                          "class_weights"], dtype=tf.float32)), 1)
        if original_input is not None:
            scaled_weights = class_weights * original_input[:,Input.len]
        else:
            scaled_weights = class_weights

        loss = tf.losses.softmax_cross_entropy(y_true, y_pred, weights=scaled_weights)


        return loss


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


class StandardNextItemNetwork(NextItemNetwork):

    def __init__(self, my_champ_emb_scales=None, opp_champ_emb_scales=None):
        super().__init__(my_champ_emb_scales, opp_champ_emb_scales)



    def build(self):
        #the +1 is for the last element which is the example importance
        in_vec = input_data(shape=[None, Input.len+ 1], name='input')
        n = tf.shape(in_vec)[0]
        batch_index = tf.range(n)
        pos = tf.cast(tf.reshape(in_vec[:, Input.indices["start"]["pos"]:Input.indices["end"]["pos"]], (-1,)),
                      tf.int32)
        champ_ints = tf.cast(in_vec[:, Input.indices["start"]["champs"]:Input.indices["end"]["champs"]], tf.int32)
        item_ints = tf.cast(in_vec[:, Input.indices["start"]["items"]:Input.indices["end"]["items"]], tf.int32)
        cs = in_vec[:, Input.indices["start"]["cs"]:Input.indices["end"]["cs"]]
        lvl = in_vec[:, Input.indices["start"]["lvl"]:Input.indices["end"]["lvl"]]
        kills = in_vec[:, Input.indices["start"]["kills"]:Input.indices["end"]["kills"]]
        deaths = in_vec[:, Input.indices["start"]["deaths"]:Input.indices["end"]["deaths"]]
        assists = in_vec[:, Input.indices["start"]["assists"]:Input.indices["end"]["assists"]]
        current_gold = in_vec[:, Input.indices["start"]["current_gold"]:Input.indices["end"]["current_gold"]]

        pos_index = tf.transpose([batch_index, pos], (1, 0))
        opp_index_no_offset = tf.transpose([batch_index, pos], (1, 0))
        target_summ_current_gold = tf.expand_dims(tf.gather_nd(current_gold, pos_index), 1)

        items_by_champ = tf.reshape(item_ints, [-1, self.game_config["champs_per_game"], self.game_config[
            "items_per_champ"], 2])
        items = self.get_items(n, items_by_champ)
        items_by_champ_k_hot = items[:, :, 1:]
        target_summ_items_sparse = tf.gather_nd(items_by_champ, pos_index)

        target_summ_items = tf.gather_nd(items_by_champ_k_hot, pos_index)
        # target_summ_items = dropout(target_summ_items, 0.9)


        cs_diff = self.calc_diff_from_target_summ(cs, pos_index)
        lvl_diff = self.calc_diff_from_target_summ(lvl, pos_index)
        kills_diff = self.calc_diff_from_target_summ(kills, pos_index)
        deaths_diff = self.calc_diff_from_target_summ(deaths, pos_index)
        assists_diff = self.calc_diff_from_target_summ(assists, pos_index)

        pos_one_hot = tf.one_hot(pos, depth=self.game_config["champs_per_team"])
        pos_one_hot_tiled = tf.tile(pos_one_hot, multiples=[1, 5])
        pos_one_hot_tiled = tf.reshape(pos_one_hot_tiled, (-1, 5, 5))
        opp_champ_pos = tf.one_hot([0, 1, 2, 3, 4], depth=5)
        opp_champ_pos = tf.reshape(opp_champ_pos, (1, 5, 5))
        opp_champ_pos = tf.tile(opp_champ_pos, multiples=[n, 1, 1])

        kills = tf.identity(kills, "kills")
        kills_diff = tf.identity(kills_diff, "kills_diff")
        deaths = tf.identity(deaths, "deaths")
        assists = tf.identity(assists, "assists")
        cs = tf.identity(cs, "cs")
        lvl = tf.identity(lvl, "lvl")

        enemy_summ_strength_input = merge(
            [
                kills_diff,
                deaths_diff,
                assists_diff,
                lvl_diff,
                cs_diff,
                pos_one_hot_tiled,
                opp_champ_pos
            ], mode='concat', axis=2)
        enemy_summ_strength_input = tf.reshape(enemy_summ_strength_input, (-1, 15))
        enemy_summ_strength_input = dropout(enemy_summ_strength_input, 0.80)

        # if bias=false this layer generates 0 values if kda diff, etc is 0. this causes null divison later because
        # the vector has no magnitude
        # enemy_summs_strength_output = fully_connected(enemy_summ_strength_input, 1, bias=True, activation='linear')

        # EDIT: above does not work with the valid_mag_idx = tf.reshape(tf.greater_equal(ets_magnitude, 1e-7), (-1,))
        # since it may dip below zero with negative weights.

        opp_strength_emb_dim = 3
        enemy_summs_strength_output = batch_normalization(fully_connected(enemy_summ_strength_input, opp_strength_emb_dim, bias=False,
                                                                          activation='relu', regularizer="L2"))
        enemy_summs_strength_output = tf.reshape(enemy_summs_strength_output, (-1, 5*opp_strength_emb_dim))
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


        champs_one_hot = tf.one_hot(tf.cast(champ_ints, tf.int32), depth=self.game_config["total_num_champs"])
        opp_champs_one_hot = champs_one_hot[:, self.game_config["champs_per_team"]:]
        opp_champs_one_hot = dropout(opp_champs_one_hot, 0.6, noise_shape=[n, self.game_config["champs_per_team"], 1])
        opp_champs_k_hot = tf.reduce_sum(opp_champs_one_hot, axis=1)
        target_summ_one_hot = tf.gather_nd(champs_one_hot, pos_index)
        opp_summ_one_hot = tf.gather_nd(opp_champs_one_hot, pos_index)

        kills_diff = tf.reshape(kills_diff, (-1, self.game_config["champs_per_team"]))
        deaths_diff = tf.reshape(deaths_diff, (-1, self.game_config["champs_per_team"]))
        assists_diff = tf.reshape(assists_diff, (-1, self.game_config["champs_per_team"]))
        lvl_diff = tf.reshape(lvl_diff, (-1, self.game_config["champs_per_team"]))
        cs_diff = tf.reshape(cs_diff, (-1, self.game_config["champs_per_team"]))


        final_input_layer = merge(
            [
                # target_summ_champ_emb_dropout_flat,
                # opp_summ_champ_emb_dropout_flat,
                # opp_team_champ_embs_dropout_flat,
                opp_champs_k_hot,
                target_summ_one_hot,
                opp_summ_one_hot,
                pos_one_hot,
                enemy_summs_strength_output,
                target_summ_current_gold,
                target_summ_items,
            ], mode='concat', axis=1)
        net = batch_normalization(fully_connected(final_input_layer, 512, bias=False, activation='relu',
                                                  regularizer="L2"))
        # net = dropout(net, 0.85)
        net = batch_normalization(fully_connected(net, 256, bias=False, activation='relu', regularizer="L2"))
        net = self.final_layer(net)
        return regression_custom(net,original_input=in_vec, target_summ_items=target_summ_items_sparse,
                                 optimizer='adam', to_one_hot=True,
                                 n_classes=self.game_config["total_num_items"],
                                 shuffle_batches=True,
                                 learning_rate=self.network_config["learning_rate"],
                                 loss=self.class_weighted_sm_ce_loss,
                                 name='target', metric=self.weighted_accuracy)


class NextItemLateGameNetwork(NextItemNetwork):

    def __init__(self, my_champ_emb_scales=None, opp_champ_emb_scales=None):
        super().__init__(my_champ_emb_scales, opp_champ_emb_scales)


    def build(self):
        in_vec = input_data(shape=[None, Input.len], name='input')
        n = tf.shape(in_vec)[0]
        batch_index = tf.range(n)
        pos = tf.cast(tf.reshape(in_vec[:, Input.indices["start"]["pos"]:Input.indices["end"]["pos"]], (-1,)),
                      tf.int32)
        champ_ints = tf.cast(in_vec[:, Input.indices["start"]["champs"]:Input.indices["end"]["champs"]], tf.int32)
        item_ints = tf.cast(in_vec[:, Input.indices["start"]["items"]:Input.indices["end"]["items"]], tf.int32)
        cs = in_vec[:, Input.indices["start"]["cs"]:Input.indices["end"]["cs"]]
        lvl = in_vec[:, Input.indices["start"]["lvl"]:Input.indices["end"]["lvl"]]
        kills = in_vec[:, Input.indices["start"]["kills"]:Input.indices["end"]["kills"]]
        deaths = in_vec[:, Input.indices["start"]["deaths"]:Input.indices["end"]["deaths"]]
        assists = in_vec[:, Input.indices["start"]["assists"]:Input.indices["end"]["assists"]]

        pos_index = tf.transpose([batch_index, pos], (1, 0))
        opp_index_no_offset = tf.transpose([batch_index, pos], (1, 0))
        my_team_champ_ints = champ_ints[:, :5]
        opp_team_champ_ints = champ_ints[:, 5:]

        target_summ_champ_emb_dropout_flat, _ = self.get_champ_embeddings_v2(my_team_champ_ints, "my_champ_embs",
                                                                             [0.1], pos_index, n, 1.0)
        _, opp_team_champ_embs_dropout_flat = self.get_champ_embeddings_v2(
            opp_team_champ_ints, "opp_champ_embs", [0.1], opp_index_no_offset, n, 1.0)
        items_by_champ = tf.reshape(item_ints, [-1, self.game_config["champs_per_game"], self.game_config[
            "items_per_champ"], 2])
        items = self.get_items(n, items_by_champ)
        items_by_champ_k_hot = items[:, :, 1:]
        target_summ_items_sparse = tf.gather_nd(items_by_champ, pos_index)
        target_summ_items = tf.gather_nd(items_by_champ_k_hot, pos_index)
        target_summ_items = dropout(target_summ_items, 0.8)

        cs_diff = self.calc_diff_from_target_summ(cs, pos_index)
        lvl_diff = self.calc_diff_from_target_summ(lvl, pos_index)
        kills_diff = self.calc_diff_from_target_summ(kills, pos_index)
        deaths_diff = self.calc_diff_from_target_summ(deaths, pos_index)
        assists_diff = self.calc_diff_from_target_summ(assists, pos_index)

        pos_one_hot = tf.one_hot(pos, depth=self.game_config["champs_per_team"])
        pos_one_hot = tf.tile(pos_one_hot, multiples=[1, 5])
        pos_one_hot = tf.reshape(pos_one_hot, (-1, 5, 5))
        opp_champ_pos = tf.one_hot([0,1,2,3,4], depth=5)
        opp_champ_pos = tf.reshape(opp_champ_pos, (1,5,5))
        opp_champ_pos =  tf.tile(opp_champ_pos, multiples=[n, 1, 1])
        enemy_summ_strength_input = merge(
            [
                kills_diff,
                deaths_diff,
                assists_diff,
                lvl_diff,
                cs_diff,
                pos_one_hot,
                opp_champ_pos
            ], mode='concat', axis=2)
        enemy_summ_strength_input = tf.reshape(enemy_summ_strength_input, (-1, 15))
        enemy_summ_strength_input = dropout(enemy_summ_strength_input, 0.80)
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
                target_summ_items
            ], mode='concat', axis=1)
        net = batch_normalization(fully_connected(nonstarter_input_layer, 256, bias=False,activation='relu',regularizer="L2"))
        net = dropout(net, 0.8)
        net = batch_normalization(fully_connected(net, 128, bias=False,activation='relu',regularizer="L2"))
        net = dropout(net, 0.9)
        net = batch_normalization(fully_connected(net, 128, bias=False,activation='relu',regularizer="L2"))
        net = self.final_layer(net)
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
        in_vec = input_data(shape=[None, Input.len], name='input')
        n = tf.shape(in_vec)[0]
        batch_index = tf.range(n)
        pos = tf.cast(tf.reshape(in_vec[:, Input.indices["start"]["pos"]:Input.indices["end"]["pos"]], (-1,)),
                      tf.int32)
        champ_ints = tf.cast(in_vec[:, Input.indices["start"]["champs"]:Input.indices["end"]["champs"]], tf.int32)
        pos_index = tf.transpose([batch_index, pos], (1, 0))
        opp_index_no_offset = tf.transpose([batch_index, pos], (1, 0))
        my_team_champ_ints = champ_ints[:, :5]
        opp_team_champ_ints = champ_ints[:, 5:]
        #0.01 seems to work best
        target_summ_champ_emb_dropout_flat, _ = self.get_champ_embeddings_v2(my_team_champ_ints, "my_champ_embs",
                                                                           [0.01], pos_index, n, 1.0)
        opp_summ_champ_emb_dropout_flat, _ = self.get_champ_embeddings_v2(
            opp_team_champ_ints, "opp_champ_embs", [0.1], opp_index_no_offset, n, 1.0)
        pos_one_hot = tf.one_hot(pos, depth=self.game_config["champs_per_team"])
        starter_input_layer = merge(
            [
                opp_summ_champ_emb_dropout_flat,
                pos_one_hot,
                target_summ_champ_emb_dropout_flat
            ], mode='concat', axis=1)
        net = batch_normalization(fully_connected(starter_input_layer, 16, bias=False,
                                                  activation='relu',
                                                  regularizer="L2"))
        net = self.final_layer(net)
        return regression(net, optimizer='adam', to_one_hot=True,
                          n_classes=self.game_config["total_num_items"],
                          shuffle_batches=True,
                          learning_rate=self.network_config["learning_rate"],
                          loss='softmax_categorical_crossentropy',
                          name='target')


class NextItemBootsNetwork(NextItemNetwork):

    def build(self):
        in_vec = input_data(shape=[None, Input.len], name='input')
        pos = tf.cast(tf.reshape(in_vec[:, Input.indices["start"]["pos"]:Input.indices["end"]["pos"]], (-1,)),
                      tf.int32)
        champ_ints = tf.cast(in_vec[:, Input.indices["start"]["champs"]:Input.indices["end"]["champs"]], tf.int32)
        n = tf.shape(in_vec)[0]
        batch_index = tf.range(n)
        pos_index = tf.transpose([batch_index, pos], (1, 0))
        opp_index_no_offset = tf.transpose([batch_index, pos], (1, 0))
        my_team_champ_ints = champ_ints[:, :5]
        opp_team_champ_ints = champ_ints[:, 5:]
        target_summ_champ_emb_dropout_flat, _ = self.get_champ_embeddings_v2(my_team_champ_ints, "my_champ_embs",
                                                                             [0.1], pos_index, n, 1.0)
        opp_summ_champ_emb_dropout_flat, opp_team_champ_embs_dropout_flat = self.get_champ_embeddings_v2(
            opp_team_champ_ints, "opp_champ_embs", [0.05], opp_index_no_offset, n, 1.0)
        pos_one_hot = tf.one_hot(pos, depth=self.game_config["champs_per_team"])
        champs_one_hot = tf.one_hot(tf.cast(champ_ints, tf.int32), depth=self.game_config["total_num_champs"])
        target_summ_one_hot = tf.gather_nd(champs_one_hot, pos_index)
        input_layer = merge(
            [
                target_summ_one_hot,
                opp_summ_champ_emb_dropout_flat,
                pos_one_hot,
                opp_team_champ_embs_dropout_flat
            ], mode='concat', axis=1)
        net = batch_normalization(fully_connected(input_layer, 16, bias=False,
                                                  activation='relu',
                                                  regularizer="L2"))
        net = self.final_layer(net)
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
        in_vec = input_data(shape=[None, Input.len], name='input')
        n = tf.shape(in_vec)[0]
        batch_index = tf.range(n)
        pos = tf.cast(tf.reshape(in_vec[:, Input.indices["start"]["pos"]:Input.indices["end"]["pos"]], (-1,)),
                      tf.int32)
        champ_ints = tf.cast(in_vec[:, Input.indices["start"]["champs"]:Input.indices["end"]["champs"]], tf.int32)
        pos_index = tf.transpose([batch_index, pos], (1, 0))
        opp_index_no_offset = tf.transpose([batch_index, pos], (1, 0))
        my_team_champ_ints = champ_ints[:, :5]
        opp_team_champ_ints = champ_ints[:, 5:]
        target_summ_champ_emb_dropout_flat, _ = self.get_champ_embeddings_v2(my_team_champ_ints, "my_champ_embs",
                                                                             [0.1], pos_index, n, 1.0)
        opp_summ_champ_emb_dropout_flat, opp_team_champ_embs_dropout_flat = self.get_champ_embeddings_v2(
            opp_team_champ_ints, "opp_champ_embs", [0.1], opp_index_no_offset, n, 1.0)
        pos_one_hot = tf.one_hot(pos, depth=self.game_config["champs_per_team"])
        high_prio_inputs = merge(
            [
                pos_one_hot,
                opp_summ_champ_emb_dropout_flat,
                opp_team_champ_embs_dropout_flat,
                target_summ_champ_emb_dropout_flat
            ], mode='concat', axis=1)
        net = batch_normalization(fully_connected(high_prio_inputs, 32, bias=False,
                                                  activation='relu',
                                                  regularizer="L2"))
        net = batch_normalization(fully_connected(net, 64, bias=False,
                                                  activation='relu',
                                                  regularizer="L2"))
        net = self.final_layer(net)
        return regression_custom(net, target_summ_items=None, optimizer='adam', to_one_hot=True,
                                 n_classes=self.game_config["total_num_items"],
                                 shuffle_batches=True,
                                 learning_rate=self.network_config["learning_rate"],
                                 loss=self.class_weighted_sm_ce_loss,
                                 name='target', metric=self.weighted_accuracy)

class ChampEmbeddings(NextItemNetwork):

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


    #given the common items for a champ we're predicting the champ int
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
