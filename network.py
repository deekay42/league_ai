from tflearn.layers.estimator import regression

import tflearn
# Data loading and preprocessing
from tflearn.activations import relu

from tflearn.layers.conv import avg_pool_2d, conv_2d, max_pool_2d, conv_1d
from tflearn.layers.core import fully_connected, input_data, dropout, highway
from tflearn.layers.embedding_ops import embedding
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import batch_normalization
import tensorflow as tf

SPELL_IMG_SIZE = (20,20)
ITEM_IMG_SIZE = (24,24)
CHAMP_IMG_SIZE = (20,20)
SELF_IMG_SIZE = (20,20)
CORNER_IMG_SIZE = (400,300)
COORDS_IMG_SIZE = (300, 122)


def detect_corners(img_size):

    is_training = tflearn.get_training_mode()
    # input
    network = input_data(shape=[None, *img_size, 3], name='input')

    conv1a_3_3 = relu(batch_normalization(
        conv_2d(network, 32, 3, strides=2, bias=False, padding='VALID', activation=None, name='Conv2d_1a_3x3'), trainable=is_training))
    conv2a_3_3 = relu(batch_normalization(
        conv_2d(conv1a_3_3, 32, 3, bias=False, padding='VALID', activation=None, name='Conv2d_2a_3x3'), trainable=is_training))
    conv2b_3_3 = relu(
        batch_normalization(conv_2d(conv2a_3_3, 64, 3, bias=False, activation=None, name='Conv2d_2b_3x3'), trainable=is_training))
    maxpool3a_3_3 = max_pool_2d(conv2b_3_3, 3, strides=2, padding='VALID', name='MaxPool_3a_3x3')
    conv3b_1_1 = relu(batch_normalization(
        conv_2d(maxpool3a_3_3, 80, 1, bias=False, padding='VALID', activation=None, name='Conv2d_3b_1x1'), trainable=is_training))
    conv4a_3_3 = relu(batch_normalization(
        conv_2d(conv3b_1_1, 192, 3, bias=False, padding='VALID', activation=None, name='Conv2d_4a_3x3'), trainable=is_training))
    maxpool5a_3_3 = max_pool_2d(conv4a_3_3, 3, strides=2, padding='VALID', name='MaxPool_5a_3x3')

    tower_conv = relu(
        batch_normalization(conv_2d(maxpool5a_3_3, 96, 1, bias=False, activation=None, name='Conv2d_5b_b0_1x1'), trainable=is_training))

    tower_conv1_0 = relu(
        batch_normalization(conv_2d(maxpool5a_3_3, 48, 1, bias=False, activation=None, name='Conv2d_5b_b1_0a_1x1'), trainable=is_training))
    tower_conv1_1 = relu(
        batch_normalization(conv_2d(tower_conv1_0, 64, 5, bias=False, activation=None, name='Conv2d_5b_b1_0b_5x5'), trainable=is_training))

    tower_conv2_0 = relu(
        batch_normalization(conv_2d(maxpool5a_3_3, 64, 1, bias=False, activation=None, name='Conv2d_5b_b2_0a_1x1'), trainable=is_training))
    tower_conv2_1 = relu(
        batch_normalization(conv_2d(tower_conv2_0, 96, 3, bias=False, activation=None, name='Conv2d_5b_b2_0b_3x3'), trainable=is_training))
    tower_conv2_2 = relu(
        batch_normalization(conv_2d(tower_conv2_1, 96, 3, bias=False, activation=None, name='Conv2d_5b_b2_0c_3x3'), trainable=is_training))

    tower_pool3_0 = avg_pool_2d(maxpool5a_3_3, 3, strides=1, padding='same', name='AvgPool_5b_b3_0a_3x3')
    tower_conv3_1 = relu(
        batch_normalization(conv_2d(tower_pool3_0, 64, 1, bias=False, activation=None, name='Conv2d_5b_b3_0b_1x1'), trainable=is_training))

    tower_5b_out = merge([tower_conv, tower_conv1_1, tower_conv2_2, tower_conv3_1], mode='concat', axis=3)

    net = fully_connected(tower_5b_out, 128, activation='relu')
    # net = dropout(net, 0.5)
    # output
    net = fully_connected(net, 8, activation='linear')
    # net = dropout(net, 0.5)
    # output

    return regression(net, optimizer='adam', learning_rate=learning_rate,
                      loss='mean_square', name='target', metric='R2')

def detect_cornersV2(img_size):

    is_training = tflearn.get_training_mode()
    # input
    network = input_data(shape=[None, img_size[1], img_size[0], 1], name='input')

    conv1 = relu(batch_normalization(
        conv_2d(network, 16, 5, bias=False, activation=None), trainable=is_training))
    maxpool1 = max_pool_2d(conv1, 2)

    conv2 = relu(batch_normalization(
        conv_2d(maxpool1, 32, 5, bias=False, activation=None), trainable=is_training))
    maxpool2 = max_pool_2d(conv2, 2)

    conv3 = relu(batch_normalization(
        conv_2d(maxpool2, 32, 5, bias=False, activation=None), trainable=is_training))
    maxpool3 = max_pool_2d(conv3, 2)

    conv4 = relu(batch_normalization(
        conv_2d(maxpool3, 32, 5, bias=False, activation=None), trainable=is_training))
    maxpool4 = max_pool_2d(conv4, 2)

    conv5 = relu(batch_normalization(
        conv_2d(maxpool4, 32, 5, bias=False, activation=None), trainable=is_training))
    maxpool5 = max_pool_2d(conv5, 2)

    net = fully_connected(maxpool5, 512, activation='relu')
    net = fully_connected(net, 8, activation='linear')

    return regression(net, optimizer='nesterov', learning_rate=learning_rate,
                      loss='mean_square', name='target', metric='R2')

def classify_spells(img_size, num_elements, learning_rate):

    is_training = tflearn.get_training_mode()
    # input
    network = input_data(shape=[None, img_size[1], img_size[0], 3], name='input')

    conv1 = relu(batch_normalization(
        conv_2d(network, 16, 3, bias=False, activation=None), trainable=is_training))
    maxpool1 = max_pool_2d(conv1, 3)

    conv2 = relu(batch_normalization(
        conv_2d(maxpool1, 32, 3, bias=False, activation=None), trainable=is_training))
    maxpool2 = max_pool_2d(conv2, 3)

    conv3 = relu(batch_normalization(
        conv_2d(maxpool2, 32, 3, bias=False, activation=None), trainable=is_training))
    maxpool3 = max_pool_2d(conv3, 3)

    net = fully_connected(maxpool3, 128, activation='relu')
    net = fully_connected(net, num_elements, activation='softmax')

    return regression(net, optimizer='adam', learning_rate=learning_rate,
                      loss='categorical_crossentropy', name='target')

def classify_champs(img_size, num_elements, learning_rate):

    is_training = tflearn.get_training_mode()
    # input
    network = input_data(shape=[None, img_size[1], img_size[0], 1], name='input')

    conv1 = relu(batch_normalization(
        conv_2d(network, 16, 5, bias=False, activation=None, regularizer="L2"), trainable=is_training))
    maxpool1 = max_pool_2d(conv1, 3)

    conv2 = relu(batch_normalization(
        conv_2d(maxpool1, 32, 5, bias=False, activation=None, regularizer="L2"), trainable=is_training))
    maxpool2 = max_pool_2d(conv2, 3)

    net = fully_connected(maxpool2, 64, activation='relu', regularizer="L2")
    # net = fully_connected(net, 128, activation='relu')
    net = fully_connected(net, num_elements, activation='softmax')

    return regression(net, optimizer='adam', learning_rate=learning_rate,
                      loss='categorical_crossentropy', name='target')


def classify_items(img_size, num_elements, learning_rate):

    is_training = tflearn.get_training_mode()
    # input
    network = input_data(shape=[None, img_size[0], img_size[1], 3], name='input')

    conv1 = relu(batch_normalization(
        conv_2d(network, 16, 5, bias=False, activation=None, regularizer="L2"), trainable=is_training))
    # maxpool1 = max_pool_2d(conv1, 2)

    conv2 = relu(batch_normalization(
        conv_2d(conv1, 32, 5, bias=False, activation=None, regularizer="L2"), trainable=is_training))
    # maxpool2 = max_pool_2d(conv2, 2)

    # conv3 = relu(batch_normalization(
    #     conv_2d(conv2, 64, 3, bias=False, activation=None, regularizer="L2"), trainable=is_training))

    net = fully_connected(conv2, 128, activation='relu', regularizer="L2")
    net = fully_connected(net, num_elements, activation='softmax')

    return regression(net, optimizer='adam', learning_rate=learning_rate,
                      loss='categorical_crossentropy', name='target')


def classify_self(img_size, num_elements, learning_rate):

    is_training = tflearn.get_training_mode()
    # input
    network = input_data(shape=[None, img_size[0], img_size[1], 1], name='input')

    conv1 = relu(batch_normalization(
        conv_2d(network, 8, 5, bias=False, activation=None, regularizer="L2"), trainable=is_training))
    maxpool1 = max_pool_2d(conv1, 2)

    # conv2 = relu(batch_normalization(
    #     conv_2d(maxpool1, 16, 5, bias=False, activation=None, regularizer="L2"), trainable=is_training))
    # maxpool2 = max_pool_2d(conv2, 2)

    # net = fully_connected(maxpool2, 16, activation='relu', regularizer="L2")
    net = fully_connected(maxpool1, num_elements, activation='sigmoid')

    return regression(net, optimizer='adam', learning_rate=learning_rate,
                      loss='binary_crossentropy', name='target')


def multi_class_acc(pred, target, input):
    pred = tf.reshape(pred, [-1, 5, 204])
    target = tf.reshape(target, [-1, 5, 204])
    correct_prediction = tf.equal(tf.argmax(pred, axis=2), tf.argmax(target, axis=2))
    # all_labels_true = tf.reduce_min(tf.cast(correct_prediction, tf.float32), 2)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return acc

def multi_class_top_k_acc(preds, targets, input):
    preds = tf.reshape(preds, [-1, 204])
    targets = tf.reshape(targets, [-1, 204])
    targets = tf.cast(targets, tf.int32)
    correct_pred = tf.nn.in_top_k(preds, tf.argmax(targets, 1), 4)
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return acc

# def multi_class_acc_positions(pred, target, input):
#     pred_5x5 = tf.reshape(pred, [-1, 5, 5])
#
#     correct = 0
#     tf.map_fn(lambda x:, elems_flat)
#
#     for example in pred_5x5:
#         final_pred = [-1, -1, -1, -1, -1]
#         for _ in range(5):
#             summ, role = tf.unravel_index(tf.argmax(example), (5,5))
#             example[summoner, :] = float('-inf')
#             example[:, role] = float('-inf')
#             final_pred[summ] = tf.one_hot(role, depth=5)
#         correct += tf.equal(final_pred, target)
#     return correct/batch_len

def permute(a, l, r):
    if l==r:
        yield list(zip([0,1,2,3,4],a))
    else:
        for i in range(l,r+1):
            a[l], a[i] = a[i], a[l]
            yield from permute(a, l+1, r)
            a[l], a[i] = a[i], a[l]

def best_permutations_one_hot(pred):
    pred_5x5 = tf.reshape(pred, [-1, 5, 5])
    pred_5x5_T = tf.transpose(pred_5x5, (1, 2, 0))
    all_perms = tf.constant(list(permute([0, 1, 2, 3, 4], 0, 4)))
    selected_elemens_per_example = tf.gather_nd(pred_5x5_T, all_perms)
    sums_per_example = tf.reduce_sum(selected_elemens_per_example, axis=1)
    best_perm_per_example_index = tf.argmax(sums_per_example, axis=0)
    best_perms = tf.gather_nd(all_perms, best_perm_per_example_index[:, tf.newaxis])[:, :, 1]
    pred_5x5_one_hot = tf.reshape(tf.one_hot(best_perms, depth=5), (-1, 5, 5))
    return pred_5x5_one_hot

def multi_class_acc_positions(pred, target, input):
    pred_5x5_one_hot = best_permutations_one_hot(pred)
    target_5x5 = tf.reshape(target, [-1, 5, 5])
    correct_prediction = tf.equal(tf.argmax(pred_5x5_one_hot, axis=2), tf.argmax(target_5x5, axis=2))
    all_correct = tf.reduce_min(tf.cast(correct_prediction, tf.float32), 1)
    acc = tf.reduce_mean(all_correct)
    return acc

#
# def multi_class_acc_positions(pred, target, input):
#     pred = tf.reshape(pred, [-1, 5, 5])
#     target = tf.reshape(target, [-1, 5, 5])
#     correct_prediction = tf.equal(tf.argmax(pred, axis=2), tf.argmax(target, axis=2))
#     all_correct = tf.reduce_min(tf.cast(correct_prediction, tf.float32), 1)
#     acc = tf.reduce_mean(all_correct)
#     return acc

positions_game_config = \
    {
        "champs_per_team": 5,
        "total_num_champs": 141,
        "spells_per_summ": 2,
        "total_num_spells": 10,
        "rest_dim": 8
    }

positions_network_config = \
    {
        "learning_rate": 0.00025,
        "champ_emb_dim": 6,
        "item_emb_dim": 7,
        "all_items_emb_dim": 10,
        "champ_all_items_emb_dim": 12,
        "target_summ": 1
    }


def classify_positions(game_config, network_config):
    champs_per_team = game_config["champs_per_team"]
    total_num_champs = game_config["total_num_champs"]
    spells_per_summ = game_config["spells_per_summ"]
    total_num_spells = game_config["total_num_spells"]
    rest_dim = game_config["rest_dim"]

    learning_rate = network_config["learning_rate"]
    champ_emb_dim = network_config["champ_emb_dim"]



    in_vec = input_data(shape=[None, champs_per_team + champs_per_team*(spells_per_summ + rest_dim)], name='input')

    champ_ints = in_vec[:, 0:champs_per_team]
    champs = embedding(champ_ints, input_dim=total_num_champs, output_dim=champ_emb_dim, reuse=tf.AUTO_REUSE,
                       scope="champ_scope")
    champs = tf.reshape(champs, [-1, champs_per_team * champ_emb_dim])

    spell_ints = in_vec[:, champs_per_team:champs_per_team+spells_per_summ*champs_per_team]
    spell_ints = tf.reshape(spell_ints, [-1, champs_per_team, spells_per_summ])

    spells_one_hot_i = tf.one_hot(tf.cast(spell_ints, tf.int32), depth=total_num_spells)
    spells_one_hot = tf.reduce_sum(spells_one_hot_i, axis=2)
    spells_one_hot = tf.reshape(spells_one_hot, [-1, champs_per_team*total_num_spells])
    rest = in_vec[:,champs_per_team+spells_per_summ*champs_per_team:]

    final_input_layer = merge([champs, spells_one_hot, rest], mode='concat', axis=1)

    net = dropout(final_input_layer, 0.8)

    net = relu(
        batch_normalization(fully_connected(net, 256, bias=False, activation=None, regularizer="L2")))
    net = dropout(net, 0.6)

    net = fully_connected(net, champs_per_team*champs_per_team, activation=None)

    return regression(net, optimizer='adam', to_one_hot=False, shuffle_batches=True,
                      learning_rate=learning_rate,
                      loss='binary_crossentropy', name='target', metric=multi_class_acc_positions)



game_config = \
    {
        "champs_per_game": 10,
        "champs_per_team": 5,
        "total_num_champs": 141,
        "total_num_items": 204,
        "items_per_champ": 6
    }

next_network_config = \
    {
        "learning_rate": 0.00025,
        "champ_emb_dim": 6,
        "item_emb_dim": 7,
        "all_items_emb_dim": 10,
        "champ_all_items_emb_dim": 12,
        "target_summ": 1
    }


def classify_next_item(game_config, network_config):
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

    in_vec = input_data(shape=[None, total_champ_dim + total_item_dim], name='input')
    #  5 elements long
    # pos = in_vec[:, :champs_per_team]
    #  10 elements long
    champ_ints = in_vec[:, :champs_per_game]
    # 60 elements long
    item_ints = in_vec[:, champs_per_game:]
    champs = embedding(champ_ints, input_dim=total_num_champs, output_dim=champ_emb_dim, reuse=tf.AUTO_REUSE,
                       scope="champ_scope")
    # items = embedding(item_ids, input_dim=total_num_items, output_dim=item_emb_dim, reuse=tf.AUTO_REUSE,
    #                   scope="item_scope")

    items_by_champ = tf.reshape(item_ints, [-1, champs_per_game, items_per_champ])
    items_by_champ_one_hot = tf.one_hot(tf.cast(items_by_champ, tf.int32), depth=total_num_items)
    items_by_champ_k_hot = tf.reduce_sum(items_by_champ_one_hot, axis=2)
    # target_summ_items = items_by_champ_k_hot[:, target_summ]
    # target_summ_opponent_items = items_by_champ_k_hot[:, target_summ+champs_per_team]
    items_by_champ_k_hot = tf.reshape(items_by_champ_k_hot, [-1, total_num_items])
    summed_items_by_champ_emb = fully_connected(items_by_champ_k_hot, item_emb_dim, bias=False, activation=None,
                                                reuse=tf.AUTO_REUSE,
                                                scope="item_sum_scope")
    summed_items_by_champ = tf.reshape(summed_items_by_champ_emb, (-1, item_emb_dim * champs_per_game))

    items_by_champ_k_hot = tf.reshape(items_by_champ_k_hot, [-1, total_num_items*champs_per_game])

    champs = tf.reshape(champs, [-1, champs_per_game * champ_emb_dim])

    summed_items_by_team1 = summed_items_by_champ[:, :champs_per_team * item_emb_dim]
    summed_items_by_team2 = summed_items_by_champ[:, champs_per_team * item_emb_dim:]

    champs_team1 = champs[:, :champs_per_team * champ_emb_dim]
    champs_team2 = champs[:, champs_per_team * champ_emb_dim:]

    team1 = merge([champs_team1, summed_items_by_team1], mode='concat', axis=1)
    team2 = merge([champs_team2, summed_items_by_team2], mode='concat', axis=1)

    team1_score = fully_connected(team1, 10, bias=False, activation=None,
                                  reuse=tf.AUTO_REUSE,
                                  scope="team_sum_scope")
    team2_score = fully_connected(team2, 10, bias=False, activation=None,
                                  reuse=tf.AUTO_REUSE,
                                  scope="team_sum_scope")

    final_input_layer = merge([items_by_champ_k_hot, summed_items_by_champ, champs, team1_score, team2_score], mode='concat', axis=1)
    net = dropout(final_input_layer, 0.9)
    net = relu(
        batch_normalization(fully_connected(net, 512, bias=False, activation=None, regularizer="L2")))
    net = dropout(net, 0.7)
    net = relu(
        batch_normalization(fully_connected(net, 256, bias=False, activation=None, regularizer="L2")))
    net = dropout(net, 0.6)

    # net = merge([net, pos], mode='concat', axis=1)

    net = fully_connected(net, total_num_items, activation='softmax')

    # TODO: consider using item embedding layer as output layer...
    return regression(net, optimizer='adam', to_one_hot=True, n_classes=total_num_items, shuffle_batches=True,
                      learning_rate=learning_rate,
                      loss='categorical_crossentropy', name='target')

