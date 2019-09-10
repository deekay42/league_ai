import tensorflow as tf
import tflearn
from tflearn.activations import relu
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import fully_connected, input_data, dropout
from tflearn.layers.embedding_ops import embedding
from tflearn.layers.estimator import regression
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import batch_normalization
from abc import ABC, abstractmethod
import numpy as np

from utils.artifact_manager import ChampManager, ItemManager, SpellManager
from constants import ui_constants, game_constants



class Network(ABC):

    @abstractmethod
    def build(self):
        pass


class ImgNetwork(Network):

    def __init__(self):
        self.num_elements = self.get_num_elements()
        self.learning_rate = 0.001


    @abstractmethod
    def get_num_elements(self):
        pass


class ChampImgNetwork(ImgNetwork):

    def __init__(self):
        super().__init__()
        self.img_size = ui_constants.NETWORK_CHAMP_IMG_CROP


    def get_num_elements(self):
        return ChampManager().get_num("img_int")


    def build(self):
        is_training = tflearn.get_training_mode()
        # input
        network = input_data(shape=[None, self.img_size[1], self.img_size[0], 3], name='input')

        conv1 = relu(batch_normalization(
            conv_2d(network, 16, 5, bias=False, activation=None, regularizer="L2"), trainable=is_training))
        maxpool1 = max_pool_2d(conv1, 3)

        conv2 = relu(batch_normalization(
            conv_2d(maxpool1, 32, 5, bias=False, activation=None, regularizer="L2"), trainable=is_training))
        maxpool2 = max_pool_2d(conv2, 3)

        net = fully_connected(maxpool2, 64, activation='relu', regularizer="L2")
        # net = fully_connected(net, 128, activation='relu')
        net = fully_connected(net, self.num_elements, activation='softmax')

        return regression(net, optimizer='adam', learning_rate=self.learning_rate, to_one_hot=True,
                          n_classes=self.num_elements, shuffle_batches=True,
                          loss='categorical_crossentropy', name='target')


class ItemImgNetwork(ImgNetwork):

    def __init__(self):
        super().__init__()
        self.img_size = ui_constants.NETWORK_ITEM_IMG_CROP


    def get_num_elements(self):
        return ItemManager().get_num("img_int")


    def build(self):
        is_training = tflearn.get_training_mode()
        # input
        network = input_data(shape=[None, self.img_size[1], self.img_size[0], 3], name='input')

        conv1 = relu(batch_normalization(
            conv_2d(network, 32, 5, bias=False, activation=None, regularizer="L2"), trainable=is_training))
        maxpool1 = max_pool_2d(conv1, 3)

        conv2 = relu(batch_normalization(
            conv_2d(maxpool1, 64, 5, bias=False, activation=None, regularizer="L2"), trainable=is_training))
        maxpool2 = max_pool_2d(conv2, 3)

        net = fully_connected(maxpool2, 128, activation='relu', regularizer="L2")
        # net = fully_connected(net, 128, activation='relu')
        net = fully_connected(net, self.num_elements, activation='softmax')

        return regression(net, optimizer='adam', learning_rate=self.learning_rate, to_one_hot=True,
                          n_classes=self.num_elements, shuffle_batches=True,
                          loss='categorical_crossentropy', name='target')


class SelfImgNetwork(ImgNetwork):

    def __init__(self):
        super().__init__()
        self.img_size = ui_constants.NETWORK_SELF_IMG_CROP


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

        return regression(net, optimizer='adam', learning_rate=self.learning_rate,
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
                "total_num_spells": SpellManager().get_num("int"),
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
    def best_permutations_one_hot(pred):
        pred_5x5 = tf.reshape(pred, [-1, 5, 5])
        pred_5x5_T = tf.transpose(pred_5x5, (1, 2, 0))
        all_perms = tf.constant(list(PositionsNetwork.permute([0, 1, 2, 3, 4], 0, 4)))
        selected_elemens_per_example = tf.gather_nd(pred_5x5_T, all_perms)
        sums_per_example = tf.reduce_sum(selected_elemens_per_example, axis=1)
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

        learning_rate = self.network_config["learning_rate"]
        champ_emb_dim = self.network_config["champ_emb_dim"]

        in_vec = input_data(shape=[None, champs_per_team + champs_per_team * (spells_per_summ + rest_dim)],
                            name='input')

        champ_ints = in_vec[:, 0:champs_per_team]
        champs = embedding(champ_ints, input_dim=total_num_champs, output_dim=champ_emb_dim, reuse=tf.AUTO_REUSE,
                           scope="champ_scope")
        champs = tf.reshape(champs, [-1, champs_per_team * champ_emb_dim])

        spell_ints = in_vec[:, champs_per_team:champs_per_team + spells_per_summ * champs_per_team]
        spell_ints = tf.reshape(spell_ints, [-1, champs_per_team, spells_per_summ])

        spells_one_hot_i = tf.one_hot(tf.cast(spell_ints, tf.int32), depth=total_num_spells)
        spells_one_hot = tf.reduce_sum(spells_one_hot_i, axis=2)
        spells_one_hot = tf.reshape(spells_one_hot, [-1, champs_per_team * total_num_spells])
        rest = in_vec[:, champs_per_team + spells_per_summ * champs_per_team:]

        final_input_layer = merge([champs, spells_one_hot, rest], mode='concat', axis=1)

        net = dropout(final_input_layer, 0.8)

        net = relu(
            batch_normalization(fully_connected(net, 256, bias=False, activation=None, regularizer="L2")))
        net = dropout(net, 0.6)

        net = fully_connected(net, champs_per_team * champs_per_team, activation=None)

        return regression(net, optimizer='adam', to_one_hot=False, shuffle_batches=True,
                          learning_rate=learning_rate,
                          loss='binary_crossentropy', name='target', metric=PositionsNetwork.multi_class_acc_positions)


class NextItemNetwork(Network):

    def __init__(self):
        self.game_config = \
            {
                "champs_per_game": game_constants.CHAMPS_PER_GAME,
                "champs_per_team": game_constants.CHAMPS_PER_TEAM,
                "total_num_champs": ChampManager().get_num("int"),

                "total_num_items": ItemManager().get_num("int"),
                "items_per_champ": game_constants.MAX_ITEMS_PER_CHAMP
            }

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

    # def multi_class_acc_positions(pred, target, input):
    #     pred = tf.reshape(pred, [-1, 5, 5])
    #     target = tf.reshape(target, [-1, 5, 5])
    #     correct_prediction = tf.equal(tf.argmax(pred, axis=2), tf.argmax(target, axis=2))
    #     all_correct = tf.reduce_min(tf.cast(correct_prediction, tf.float32), 1)
    #     acc = tf.reduce_mean(all_correct)
    #     return acc


class NextItemEarlyGameNetwork(NextItemNetwork):

    def __init__(self):
        super().__init__()

        self.network_config = \
            {
                "learning_rate": 0.00025,
                "champ_emb_dim": 3,
                "all_items_emb_dim": 6,
                "champ_all_items_emb_dim": 8,
                "class_weights": 1
            }


    def build(self):
        champs_per_game = self.game_config["champs_per_game"]
        total_num_champs = self.game_config["total_num_champs"]
        total_num_items = self.game_config["total_num_items"]
        items_per_champ = self.game_config["items_per_champ"]
        champs_per_team = self.game_config["champs_per_team"]

        learning_rate = self.network_config["learning_rate"]
        champ_emb_dim = self.network_config["champ_emb_dim"]

        all_items_emb_dim = self.network_config["all_items_emb_dim"]
        champ_all_items_emb_dim = self.network_config["champ_all_items_emb_dim"]

        total_champ_dim = champs_per_game
        total_item_dim = champs_per_game * items_per_champ

        in_vec = input_data(shape=[None, 1 + total_champ_dim + total_item_dim], name='input')
        #  1 elements long
        pos = in_vec[:, 0]
        pos = tf.cast(pos, tf.int32)

        n = tf.shape(in_vec)[0]
        batch_index = tf.range(n)
        pos_index = tf.transpose([batch_index, pos], (1, 0))


        # Make tensor of indices for the first dimension

        #  10 elements long
        champ_ints = in_vec[:, 1:champs_per_game + 1]
        # 60 elements long
        item_ints = in_vec[:, champs_per_game + 1:]
        champs_embedded = embedding(champ_ints, input_dim=total_num_champs, output_dim=champ_emb_dim,
                                   reuse=tf.AUTO_REUSE,
                           scope="champ_scope")
        target_summ_champ = tf.gather_nd(champs_embedded, pos_index)
        champs = tf.reshape(champs_embedded, [-1, champs_per_game * champ_emb_dim])
        # items = embedding(item_ids, input_dim=total_num_items, output_dim=item_emb_dim, reuse=tf.AUTO_REUSE,
        #                   scope="item_scope")

        items_by_champ = tf.reshape(item_ints, [-1, champs_per_game, items_per_champ])
        items_by_champ_one_hot = tf.one_hot(tf.cast(items_by_champ, tf.int32), depth=total_num_items)
        items_by_champ_k_hot = tf.reduce_sum(items_by_champ_one_hot, axis=2)

        #we are deleting the 0 items, i.e. the Empty items. they are required to ensure that each summoner always has
        # 6 items, but they shouldn't influence the next item calculation
        # edit: doesn't make a difference in training. might actually be detrimental to determining how many item
        # slots are left open
        #items_by_champ_k_hot = items_by_champ_k_hot[:,:,1:]
        #total_num_items = total_num_items - 1
        #items_by_champ_k_hot_flat = tf.reshape(items_by_champ_k_hot, [-1, total_num_items * champs_per_game])
        target_summ_items = tf.gather_nd(items_by_champ_k_hot, pos_index)

        items_by_champ_k_hot_rep = tf.reshape(items_by_champ_k_hot, [-1, total_num_items])
        summed_items_by_champ_emb = fully_connected(items_by_champ_k_hot_rep, all_items_emb_dim, bias=False, activation=None,
                                                    reuse=tf.AUTO_REUSE,
                                                    scope="item_sum_scope")
        summed_items_by_champ = tf.reshape(summed_items_by_champ_emb, (-1, champs_per_game, all_items_emb_dim))

        summed_items_by_champ_exp = tf.expand_dims(summed_items_by_champ, -1)
        champs_exp = tf.expand_dims(champs_embedded, -2)
        champs_with_items = tf.multiply(champs_exp, summed_items_by_champ_exp)
        champs_with_items = tf.reshape(champs_with_items, (-1, champ_emb_dim * all_items_emb_dim))
        champs_with_items_emb = fully_connected(champs_with_items, champ_all_items_emb_dim, bias=False, activation=None,
                                                reuse=tf.AUTO_REUSE, scope="champ_item_scope")
        champs_with_items_emb = tf.reshape(champs_with_items_emb, (-1, champs_per_game * champ_all_items_emb_dim))

        pos = tf.one_hot(pos, depth=champs_per_team)
        final_input_layer = merge(
            [pos, target_summ_champ, target_summ_items, champs_with_items_emb, champs],
            mode='concat', axis=1)
        # final_input_layer = dropout(final_input_layer, 0.9)
        net = batch_normalization(fully_connected(final_input_layer, 256, bias=False, activation='relu',
                                                  regularizer="L2"))
        # net = dropout(net, 0.9)
        # net = batch_normalization(fully_connected(net, 256, bias=False, activation='relu', regularizer="L2"))


        net = fully_connected(net, total_num_items, activation='linear')

        is_training = tflearn.get_training_mode()
        inference_output = tf.nn.softmax(net)

        net = tf.cond(is_training, lambda: net, lambda: inference_output)

        # TODO: consider using item embedding layer as output layer...
        return regression(net, optimizer='adam', to_one_hot=True, n_classes=total_num_items, shuffle_batches=True,
                          learning_rate=learning_rate,
                          loss=self.class_weighted_sm_ce_loss,
                          name='target')


    def class_weighted_sm_ce_loss(self, y_pred, y_true):
        class_weights = tf.reduce_sum(tf.multiply(y_true, tf.constant(self.network_config[
                                            "class_weights"], dtype=tf.float32)), 1)
        return tf.losses.softmax_cross_entropy(y_true, y_pred, weights=class_weights)

#
# class NextItemEarlyGameNetwork(NextItemNetwork):
#
#     def __init__(self):
#         super().__init__()
#
#         self.network_config = \
#             {
#                 "learning_rate": 0.00025,
#                 "champ_emb_dim": 3,
#                 "all_items_emb_dim": 6,
#                 "champ_all_items_emb_dim": 8
#             }
#
#
#     def build(self):
#         champs_per_game = self.game_config["champs_per_game"]
#         total_num_champs = self.game_config["total_num_champs"]
#         total_num_items = self.game_config["total_num_items"]
#         items_per_champ = self.game_config["items_per_champ"]
#         champs_per_team = self.game_config["champs_per_team"]
#
#         learning_rate = self.network_config["learning_rate"]
#         champ_emb_dim = self.network_config["champ_emb_dim"]
#
#         all_items_emb_dim = self.network_config["all_items_emb_dim"]
#         champ_all_items_emb_dim = self.network_config["champ_all_items_emb_dim"]
#
#         total_champ_dim = champs_per_game
#         total_item_dim = champs_per_game * items_per_champ
#
#         in_vec = input_data(shape=[None, 1 + total_champ_dim + total_item_dim], name='input')
#         #  1 elements long
#         pos = in_vec[:, 0]
#         pos = tf.cast(pos, tf.int32)
#
#         n = tf.shape(in_vec)[0]
#         batch_index = tf.range(n)
#         pos_index = tf.transpose([batch_index, pos], (1, 0))
#
#
#         # Make tensor of indices for the first dimension
#
#         #  10 elements long
#         champ_ints = in_vec[:, 1:champs_per_game + 1]
#         # 60 elements long
#         item_ints = in_vec[:, champs_per_game + 1:]
#
#         champs_one_hot = tf.one_hot(tf.cast(champ_ints, tf.int32), depth=total_num_champs)
#         champs_one_hot_flat = tf.reshape(champs_one_hot, [-1, champs_per_game * total_num_champs])
#
#         items_by_champ = tf.reshape(item_ints, [-1, champs_per_game, items_per_champ])
#         items_by_champ_one_hot = tf.one_hot(tf.cast(items_by_champ, tf.int32), depth=total_num_items)
#         items_by_champ_k_hot = tf.reduce_sum(items_by_champ_one_hot, axis=2)
#         items_by_champ_k_hot_flat =  tf.reshape(items_by_champ_k_hot, [-1, champs_per_game * total_num_items])
#
#         pos = tf.one_hot(pos, depth=champs_per_team)
#         final_input_layer = merge(
#             [pos, champs_one_hot_flat, items_by_champ_k_hot_flat],
#             mode='concat', axis=1)
#
#         net = fully_connected(final_input_layer, 1500, activation='relu')
#
#         net = fully_connected(net, total_num_items, activation='softmax')
#
#         # TODO: consider using item embedding layer as output layer...
#         return regression(net, optimizer='adam', to_one_hot=True, n_classes=total_num_items, shuffle_batches=True,
#                           learning_rate=learning_rate,
#                           loss='categorical_crossentropy', name='target')



class NextItemLateGameNetwork(NextItemNetwork):

    def __init__(self):
        super().__init__()

        self.network_config = \
            {
                "learning_rate": 0.00025,
                "champ_emb_dim": 3,
                "all_items_emb_dim": 6,
                "champ_all_items_emb_dim": 8
            }

    def build(self):
        champs_per_game = self.game_config["champs_per_game"]
        total_num_champs = self.game_config["total_num_champs"]
        total_num_items = self.game_config["total_num_items"]
        items_per_champ = self.game_config["items_per_champ"]
        champs_per_team = self.game_config["champs_per_team"]

        learning_rate = self.network_config["learning_rate"]
        champ_emb_dim = self.network_config["champ_emb_dim"]

        all_items_emb_dim = self.network_config["all_items_emb_dim"]
        champ_all_items_emb_dim = self.network_config["champ_all_items_emb_dim"]

        total_champ_dim = champs_per_game
        total_item_dim = champs_per_game * items_per_champ

        in_vec = input_data(shape=[None, 1 + total_champ_dim + total_item_dim], name='input')
        #  1 elements long
        pos = in_vec[:, 0]
        pos = tf.cast(pos, tf.int32)

        n = tf.shape(in_vec)[0]
        batch_index = tf.range(n)
        pos_index = tf.transpose([batch_index, pos], (1, 0))


        # Make tensor of indices for the first dimension

        #  10 elements long
        champ_ints = in_vec[:, 1:champs_per_game + 1]
        # 60 elements long
        item_ints = in_vec[:, champs_per_game + 1:]
        champs_embedded = embedding(champ_ints, input_dim=total_num_champs, output_dim=champ_emb_dim,
                                   reuse=tf.AUTO_REUSE,
                           scope="champ_scope")
        target_summ_champ = tf.gather_nd(champs_embedded, pos_index)
        champs = tf.reshape(champs_embedded, [-1, champs_per_game * champ_emb_dim])
        # items = embedding(item_ids, input_dim=total_num_items, output_dim=item_emb_dim, reuse=tf.AUTO_REUSE,
        #                   scope="item_scope")

        items_by_champ = tf.reshape(item_ints, [-1, champs_per_game, items_per_champ])
        items_by_champ_one_hot = tf.one_hot(tf.cast(items_by_champ, tf.int32), depth=total_num_items)
        items_by_champ_k_hot = tf.reduce_sum(items_by_champ_one_hot, axis=2)

        #we are deleting the 0 items, i.e. the Empty items. they are required to ensure that each summoner always has
        # 6 items, but they shouldn't influence the next item calculation
        # edit: doesn't make a difference in training. might actually be detrimental to determining how many item
        # slots are left open
        #items_by_champ_k_hot = items_by_champ_k_hot[:,:,1:]
        #total_num_items = total_num_items - 1
        #items_by_champ_k_hot_flat = tf.reshape(items_by_champ_k_hot, [-1, total_num_items * champs_per_game])
        target_summ_items = tf.gather_nd(items_by_champ_k_hot, pos_index)

        items_by_champ_k_hot_rep = tf.reshape(items_by_champ_k_hot, [-1, total_num_items])
        summed_items_by_champ_emb = fully_connected(items_by_champ_k_hot_rep, all_items_emb_dim, bias=False, activation=None,
                                                    reuse=tf.AUTO_REUSE,
                                                    scope="item_sum_scope")
        summed_items_by_champ = tf.reshape(summed_items_by_champ_emb, (-1, champs_per_game, all_items_emb_dim))

        summed_items_by_champ_exp = tf.expand_dims(summed_items_by_champ, -1)
        champs_exp = tf.expand_dims(champs_embedded, -2)
        champs_with_items = tf.multiply(champs_exp, summed_items_by_champ_exp)
        champs_with_items = tf.reshape(champs_with_items, (-1, champ_emb_dim * all_items_emb_dim))
        champs_with_items_emb = fully_connected(champs_with_items, champ_all_items_emb_dim, bias=False, activation=None,
                                                reuse=tf.AUTO_REUSE, scope="champ_item_scope")
        champs_with_items_emb = tf.reshape(champs_with_items_emb, (-1, champs_per_game * champ_all_items_emb_dim))

        final_input_layer = merge(
            [target_summ_champ, target_summ_items, champs_with_items_emb, champs], mode='concat', axis=1)
        net = batch_normalization(fully_connected(final_input_layer, 256, bias=False, activation='relu',
                                                  regularizer="L2"))
        net = dropout(net, 0.9)
        net = batch_normalization(fully_connected(net, 256, bias=False, activation='relu', regularizer="L2"))
        net = dropout(net, 0.9)
        net = fully_connected(net, total_num_items, activation='softmax')

        # TODO: consider using item embedding layer as output layer...
        return regression(net, optimizer='adam', to_one_hot=True, n_classes=total_num_items, shuffle_batches=True,
                          learning_rate=learning_rate,
                          loss='categorical_crossentropy', name='target')
