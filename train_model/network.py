import tensorflow as tf
import tflearn
from tflearn.activations import relu
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import fully_connected, input_data, dropout
from tflearn.layers.embedding_ops import embedding
from tflearn.layers.estimator import regression, regression_custom
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import batch_normalization
from abc import ABC, abstractmethod
import numpy as np
import itertools
from utils.artifact_manager import ChampManager, ItemManager, SimpleManager
from constants import ui_constants, game_constants
from constants.ui_constants import ResConverter



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
            np.array(list(PositionsNetwork.permute([0, 1, 2, 3, 4], 0, 4)))[:, :, 1][:, :,tf.newaxis])
        x = np.transpose(x, (1, 2, 0))
        x = tf.gather_nd(np.array(x), perm_indices)
        x = tf.transpose(x, (3, 0, 1, 2))
        x = tf.reshape(x, (-1, 55))
        return x


    #expects already permutated pred input vector
    @staticmethod
    def select_best_input_perm(pred):

        perm_indices = tf.constant(
            np.array(list(PositionsNetwork.permute([0, 1, 2, 3, 4], 0, 4)))[:, :, 1][:, :, tf.newaxis])
        num_perms = perm_indices.shape[0]
        batch_size = pred.shape[0]//num_perms
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
                           scope="champ_scope")
        champs = tf.reshape(champs, [-1, champs_per_team * champ_emb_dim])

        spell_ints = [in_vec[:,i*total_sum_dim+1:i*total_sum_dim+1+spells_per_summ] for i in range(champs_per_team)]
        spell_ints = tf.transpose(spell_ints, (1, 0, 2))
        spell_ints = tf.reshape(spell_ints, [-1, champs_per_team, spells_per_summ])

        spells_one_hot_i = tf.one_hot(tf.cast(spell_ints, tf.int32), depth=total_num_spells)
        spells_one_hot = tf.reduce_sum(spells_one_hot_i, axis=2)
        spells_one_hot = tf.reshape(spells_one_hot, [-1, champs_per_team * total_num_spells])

        rest = [in_vec[:,i*total_sum_dim+1+spells_per_summ:i*total_sum_dim+1+spells_per_summ+rest_dim] for i in
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
#                 "champ_all_items_emb_dim": 8,
#                 "class_weights": 1
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
#         champs_embedded = embedding(champ_ints, input_dim=total_num_champs, output_dim=champ_emb_dim,
#                                    reuse=tf.AUTO_REUSE,
#                            scope="champ_scope")
#         target_summ_champ = tf.gather_nd(champs_embedded, pos_index)
#         champs = tf.reshape(champs_embedded, [-1, champs_per_game * champ_emb_dim])
#         # items = embedding(item_ids, input_dim=total_num_items, output_dim=item_emb_dim, reuse=tf.AUTO_REUSE,
#         #                   scope="item_scope")
#
#         items_by_champ = tf.reshape(item_ints, [-1, champs_per_game, items_per_champ])
#         items_by_champ_one_hot = tf.one_hot(tf.cast(items_by_champ, tf.int32), depth=total_num_items)
#         items_by_champ_k_hot = tf.reduce_sum(items_by_champ_one_hot, axis=2)
#
#         #we are deleting the 0 items, i.e. the Empty items. they are required to ensure that each summoner always has
#         # 6 items, but they shouldn't influence the next item calculation
#         # edit: doesn't make a difference in training. might actually be detrimental to determining how many item
#         # slots are left open
#         #items_by_champ_k_hot = items_by_champ_k_hot[:,:,1:]
#         #total_num_items = total_num_items - 1
#         #items_by_champ_k_hot_flat = tf.reshape(items_by_champ_k_hot, [-1, total_num_items * champs_per_game])
#         target_summ_items = tf.gather_nd(items_by_champ_k_hot, pos_index)
#
#         items_by_champ_k_hot_rep = tf.reshape(items_by_champ_k_hot, [-1, total_num_items])
#         summed_items_by_champ_emb = fully_connected(items_by_champ_k_hot_rep, all_items_emb_dim, bias=False, activation=None,
#                                                     reuse=tf.AUTO_REUSE,
#                                                     scope="item_sum_scope")
#         summed_items_by_champ = tf.reshape(summed_items_by_champ_emb, (-1, champs_per_game, all_items_emb_dim))
#
#         summed_items_by_champ_exp = tf.expand_dims(summed_items_by_champ, -1)
#         champs_exp = tf.expand_dims(champs_embedded, -2)
#         champs_with_items = tf.multiply(champs_exp, summed_items_by_champ_exp)
#         champs_with_items = tf.reshape(champs_with_items, (-1, champ_emb_dim * all_items_emb_dim))
#         champs_with_items_emb = fully_connected(champs_with_items, champ_all_items_emb_dim, bias=False, activation=None,
#                                                 reuse=tf.AUTO_REUSE, scope="champ_item_scope")
#         champs_with_items_emb = tf.reshape(champs_with_items_emb, (-1, champs_per_game * champ_all_items_emb_dim))
#
#         pos = tf.one_hot(pos, depth=champs_per_team)
#         final_input_layer = merge(
#             [pos, target_summ_champ, target_summ_items, champs_with_items_emb, champs],
#             mode='concat', axis=1)
#         # final_input_layer = dropout(final_input_layer, 0.9)
#         net = batch_normalization(fully_connected(final_input_layer, 256, bias=False, activation='relu', regularizer="L2"))
#         # net = dropout(net, 0.9)
#         # net = batch_normalization(fully_connected(net, 256, bias=False, activation='relu', regularizer="L2"))
#
#
#         net = fully_connected(net, total_num_items, activation='linear')
#
#         is_training = tflearn.get_training_mode()
#         inference_output = tf.nn.softmax(net)
#
#         net = tf.cond(is_training, lambda: net, lambda: inference_output)
#
#         # TODO: consider using item embedding layer as output layer...
#         return regression(net, optimizer='adam', to_one_hot=True, n_classes=total_num_items, shuffle_batches=True,
#                           learning_rate=learning_rate,
#                           loss=self.class_weighted_sm_ce_loss,
#                           name='target', metric=self.weighted_accuracy)
#
#
#     def class_weighted_sm_ce_loss(self, y_pred, y_true):
#         class_weights = tf.reduce_sum(tf.multiply(y_true, tf.constant(self.network_config[
#                                             "class_weights"], dtype=tf.float32)), 1)
#         return tf.losses.softmax_cross_entropy(y_true, y_pred, weights=class_weights)
#
#
#     def weighted_accuracy(self, preds, targets, input_):
#         targets_sparse = tf.argmax(targets, axis=-1)
#         max_achievable_score = tf.reduce_sum(tf.gather(self.network_config[
#                                             "class_weights"], targets_sparse))
#         preds_sparse = tf.argmax(preds, axis=-1)
#         matching_preds_sparse = tf.boolean_mask(targets_sparse, tf.equal(targets_sparse, preds_sparse))
#         actually_achieved_score = tf.reduce_sum(tf.gather(self.network_config[
#                                             "class_weights"], matching_preds_sparse))
#         return actually_achieved_score / max_achievable_score


class NextItemEarlyGameNetwork(NextItemNetwork):

    def __init__(self):
        super().__init__()

        self.network_config = \
            {
                "learning_rate": 0.00025,
                "champ_emb_dim": 3,
                "all_items_emb_dim": 6,
                "champ_all_items_emb_dim": 8,
                "class_weights": [1]
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

        pos_start = 0
        pos_end = pos_start + 1
        champs_start = pos_end
        champs_end = champs_start + champs_per_game
        items_start = champs_end
        items_end = items_start + items_per_champ*2*champs_per_game
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
        kda_end = kda_start + champs_per_game*3
        current_gold_start = kda_end
        current_gold_end = current_gold_start + champs_per_game

        in_vec = input_data(shape=[None, 221], name='input')
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
        # champ_ints = dropout(champ_ints, 0.8)
        #this does not work since dropout scales inputs, hence embedding lookup fails after that.
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
        target_summ_kda = tf.gather_nd(tf.reshape(kda,(-1,champs_per_game, 3)), pos_index)
        target_summ_lvl = tf.expand_dims(tf.gather_nd(lvl, pos_index), 1)

        champs_embedded = embedding(champ_ints, input_dim=total_num_champs, output_dim=champ_emb_dim,
                                                                       reuse=tf.AUTO_REUSE,
                                                               scope="champ_scope")

        champs_embedded_flat = tf.reshape(champs_embedded, (-1, champ_emb_dim*champs_per_game))
        champs_one_hot = tf.one_hot(tf.cast(champ_ints, tf.int32), depth=total_num_champs)
        opp_champs_one_hot = champs_one_hot[:,champs_per_team:]
        opp_champs_k_hot = tf.reduce_sum(opp_champs_one_hot, axis=1)
        opp_champs_k_hot = tf.cast(tf.cast(tf.greater(opp_champs_k_hot, 0), tf.int32), tf.float32)
        # champs_one_hot_flat = tf.reshape(champs_one_hot, [-1, champs_per_game * total_num_champs])
        target_summ_champ = tf.gather_nd(champs_one_hot, pos_index)
        opp_summ_champ = tf.gather_nd(champs_one_hot, opp_index)

        target_summ_champ_emb = tf.gather_nd(champs_embedded, pos_index)
        opp_summ_champ_emb = tf.gather_nd(champs_embedded, opp_index)

        items_by_champ = tf.reshape(item_ints, [-1, champs_per_game, items_per_champ, 2])
        items_by_champ_flat = tf.reshape(items_by_champ, [-1])

        batch_indices = tf.reshape(tf.tile(tf.expand_dims(tf.range(n), 1), [1, champs_per_game * items_per_champ]),
                                   (-1,))
        champ_indices = tf.reshape(tf.tile(tf.tile(tf.expand_dims(tf.range(champs_per_game),1), [1,
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

        items_by_champ_k_hot_flat =  tf.reshape(items_by_champ_k_hot, [-1, champs_per_game * total_num_items])

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

        target_summ_items_sparse = tf.gather_nd(items_by_champ, pos_index)
        target_summ_items = tf.gather_nd(items_by_champ_k_hot, pos_index)
        opp_summ_items = tf.gather_nd(items_by_champ_k_hot, opp_index)

        pos = tf.one_hot(pos, depth=champs_per_team)
        final_input_layer1 = merge(
            [
                pos,
                target_summ_champ_emb,
                target_summ_items,
                target_summ_current_gold
        ], mode='concat', axis=1)

        final_input_layer2 = merge(
            [
                target_summ_cs,
                target_summ_kda,
                target_summ_lvl,
                lvl,
                kda,
                total_cs,
                # opp_summ_champ,
                opp_summ_champ_emb,
                opp_summ_items,
                opp_champs_k_hot,
                champs_with_items_emb,
                # target_summ_champ,
            ], mode='concat', axis=1)
        final_input_layer2 = dropout(final_input_layer2, 0.5)

        final_input_layer = merge(
            [
                final_input_layer1,
                final_input_layer2
            ], mode='concat', axis=1)

        net = batch_normalization(fully_connected(final_input_layer, 1024, bias=False, activation='relu',
                                                regularizer="L2"))
        # net = dropout(net, 0.8)
        net = batch_normalization(fully_connected(net, 512, bias=False, activation='relu',
                                                  regularizer="L2"))
        # net = dropout(net, 0.9)
        net = batch_normalization(fully_connected(net, 256, bias=False, activation='relu',
                                                  regularizer="L2"))
        net = merge([target_summ_current_gold, net], mode='concat', axis=1)
        net = fully_connected(net, total_num_items, activation='linear')
        is_training = tflearn.get_training_mode()
        inference_output = tf.nn.softmax(net)

        net = tf.cond(is_training, lambda: net, lambda: inference_output)

        return regression_custom(net, target_summ_items=target_summ_items_sparse, optimizer='adam', to_one_hot=True, n_classes=total_num_items,
                                 shuffle_batches=True,
                                  learning_rate=learning_rate,
                                  loss=self.class_weighted_sm_ce_loss,
                                  name='target', metric=self.weighted_accuracy)


    #calculates penalty if prediction is an item we already have, or an item with an item effect we already have
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
        return weighted_accuracy(preds_sparse, targets_sparse, self.network_config["class_weights"])


class NextItemLateGameNetwork(NextItemNetwork):

    def __init__(self):
        super().__init__()

        self.network_config = \
            {
                "learning_rate": 0.00025,
                "champ_emb_dim": 3,
                "all_items_emb_dim": 4,
                "champ_all_items_emb_dim": 6,
                "class_weights": [1]
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
        pos_dim = 2

        pos_start = 0
        pos_end = pos_start + 1
        champs_start = pos_end
        champs_end = champs_start + champs_per_game
        items_start = champs_end
        items_end = items_start + items_per_champ*2*champs_per_game
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
        kda_end = kda_start + champs_per_game*3
        current_gold_start = kda_end
        current_gold_end = current_gold_start + champs_per_game

        in_vec = input_data(shape=[None, 221], name='input')
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
        # champ_ints = dropout(champ_ints, 0.8)
        #this does not work since dropout scales inputs, hence embedding lookup fails after that.
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
        target_summ_kda = tf.gather_nd(tf.reshape(kda,(-1,champs_per_game, 3)), pos_index)
        target_summ_lvl = tf.expand_dims(tf.gather_nd(lvl, pos_index), 1)
        opp_summ_cs = tf.expand_dims(tf.gather_nd(total_cs, opp_index), 1)
        opp_summ_kda = tf.gather_nd(tf.reshape(kda, (-1, champs_per_game, 3)), opp_index)
        opp_summ_lvl = tf.expand_dims(tf.gather_nd(lvl, opp_index), 1)

        champs_embedded = embedding(champ_ints, input_dim=total_num_champs, output_dim=champ_emb_dim,
                                    reuse=tf.AUTO_REUSE,
                                    scope="champ_scope")
        champs_embedded_short1 = embedding(champ_ints, input_dim=total_num_champs, output_dim=champ_emb_dim-1,
                                    reuse=tf.AUTO_REUSE,
                                    scope="champs_embedded_short1")
        champs_embedded_short2 = embedding(champ_ints, input_dim=total_num_champs, output_dim=champ_emb_dim-2,
                                    reuse=tf.AUTO_REUSE,
                                    scope="champs_embedded_short2")

        champs_embedded_flat = tf.reshape(champs_embedded, (-1, champ_emb_dim*champs_per_game))
        champs_one_hot = tf.one_hot(tf.cast(champ_ints, tf.int32), depth=total_num_champs)
        opp_champs_one_hot = champs_one_hot[:,champs_per_team:]
        opp_champs_k_hot = tf.reduce_sum(opp_champs_one_hot, axis=1)
        opp_champs_k_hot = tf.cast(tf.cast(tf.greater(opp_champs_k_hot, 0), tf.int32), tf.float32)
        # champs_one_hot_flat = tf.reshape(champs_one_hot, [-1, champs_per_game * total_num_champs])
        target_summ_champ = tf.gather_nd(champs_one_hot, pos_index)
        opp_summ_champ = tf.gather_nd(champs_one_hot, opp_index)

        target_summ_champ_emb = tf.gather_nd(champs_embedded, pos_index)
        opp_summ_champ_emb = tf.gather_nd(champs_embedded, opp_index)
        target_summ_champ_emb_short1 = tf.gather_nd(champs_embedded_short1, pos_index)
        opp_summ_champ_emb_short1 = tf.gather_nd(champs_embedded_short1, opp_index)
        target_summ_champ_emb_short2 = tf.gather_nd(champs_embedded_short2, pos_index)
        opp_summ_champ_emb_short2 = tf.gather_nd(champs_embedded_short2, opp_index)

        items_by_champ = tf.reshape(item_ints, [-1, champs_per_game, items_per_champ, 2])
        items_by_champ_flat = tf.reshape(items_by_champ, [-1])

        batch_indices = tf.reshape(tf.tile(tf.expand_dims(tf.range(n), 1), [1, champs_per_game * items_per_champ]),
                                   (-1,))
        champ_indices = tf.reshape(tf.tile(tf.tile(tf.expand_dims(tf.range(champs_per_game),1), [1,
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

        items_by_champ_k_hot_flat =  tf.reshape(items_by_champ_k_hot, [-1, champs_per_game * total_num_items])

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

        target_summ_items_sparse = tf.gather_nd(items_by_champ, pos_index)
        target_summ_items = tf.gather_nd(items_by_champ_k_hot, pos_index)
        opp_summ_items = tf.gather_nd(items_by_champ_k_hot, opp_index)

        # pos = tf.one_hot(pos, depth=champs_per_team)

        pos = tf.expand_dims(pos, -1)
        pos_embedded = embedding(pos, input_dim=5, output_dim=pos_dim,
                                    reuse=tf.AUTO_REUSE,
                                    scope="pos_scope")
        pos_embedded = tf.reshape(pos_embedded, (-1, pos_dim))


        opp_strength_input = merge(
            [
                opp_summ_champ_emb,
                opp_summ_champ_emb_short1,
                opp_summ_items,
                opp_champs_k_hot,
                champs_with_items_emb
            ], mode='concat', axis=1)
        opp_strength_output = batch_normalization(fully_connected(opp_strength_input, 20, bias=False,
                                                                 activation='relu', regularizer="L2"))

        stats_input = merge(
            [
                target_summ_cs,
                target_summ_kda,
                target_summ_lvl,
                opp_summ_cs,
                opp_summ_kda,
                opp_summ_lvl,
                lvl,
                kda,
                total_cs
            ], mode='concat', axis=1)
        stats_output = batch_normalization(fully_connected(stats_input, 10, bias=False,
                                                                 activation='relu', regularizer="L2"))

        champ_context_input = merge(
            [
                opp_strength_output,
                stats_output,
                opp_summ_champ_emb_short2,
                target_summ_champ_emb,
                pos_embedded
            ], mode='concat', axis=1)
        champ_context_output = batch_normalization(fully_connected(champ_context_input, 32, bias=False,
                                                                 activation='relu', regularizer="L2"))

        final_input_layer = merge(
            [
                champ_context_output,
                target_summ_champ_emb_short1,
                target_summ_champ_emb_short2,
                target_summ_items
        ], mode='concat', axis=1)
        net = batch_normalization(fully_connected(final_input_layer, 256, bias=False, activation='relu',
                                                  regularizer="L2"))

        net = merge([target_summ_current_gold, net], mode='concat', axis=1)
        net = fully_connected(net, total_num_items, activation='linear')

        is_training = tflearn.get_training_mode()
        inference_output = tf.nn.softmax(net)

        net = tf.cond(is_training, lambda: net, lambda: inference_output)

        return regression_custom(net, target_summ_items=target_summ_items_sparse, optimizer='adam', to_one_hot=True, n_classes=total_num_items,
                                 shuffle_batches=True,
                                  learning_rate=learning_rate,
                                  loss=self.class_weighted_sm_ce_loss,
                                  name='target', metric=self.weighted_accuracy)


    #calculates penalty if prediction is an item we already have, or an item with an item effect we already have
    def class_weighted_sm_ce_loss(self, y_pred, y_true, target_summ_items_sparse):
        class_weights = tf.reduce_sum(tf.multiply(y_true, tf.constant(self.network_config[
                                            "class_weights"], dtype=tf.float32)), 1)
        return tf.losses.softmax_cross_entropy(y_true, y_pred, weights=class_weights)

    def weighted_accuracy(self, preds, targets, input_):
        targets_sparse = tf.argmax(targets, axis=-1)
        preds_sparse = tf.argmax(preds, axis=-1)
        return weighted_accuracy(preds_sparse, targets_sparse, self.network_config["class_weights"])


def weighted_accuracy(preds_sparse, targets_sparse, class_weights):
    max_achievable_score = tf.reduce_sum(tf.gather(class_weights, targets_sparse))
    matching_preds_sparse = tf.boolean_mask(targets_sparse, tf.equal(targets_sparse, preds_sparse))
    actually_achieved_score = tf.reduce_sum(tf.gather(class_weights, matching_preds_sparse))
    return actually_achieved_score / max_achievable_score

