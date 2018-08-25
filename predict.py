import cv2 as cv

import constants
import utils
import tflearn
import network
import numpy as np
import tensorflow as tf
import train

class Predictor:

    def __init__(self, img_width, img_height):
        self.const = constants.ResConverter(img_width, img_height)
        summ_names_displayed = utils.summ_names_displayed()
        if summ_names_displayed:
            item_x_offset = -9
            item_y_offset = 11
        else:
            item_x_offset = 0
            item_y_offset = 0

        print("Initializing neural networks...")
        champ_model_path = './best_models/champs/my_model'
        item_model_path = './best_models/items/my_model'
        spell_model_path = './best_models/spells/my_model'
        self_model_path = './best_models/self/my_model'
        next_item_model_path = './best_models/next_items/my_model'

        self.champ_mapper = utils.Converter().champ_int2string_old
        self.item_mapper = utils.Converter().item_int2string_old
        self.spell_mapper = utils.int2spell()
        self.self_mapper = [0,1]
        
        self.champ_coords = utils.generateChampCoordinates(self.const.CHAMP_LEFT_X_OFFSET, self.const.CHAMP_RIGHT_X_OFFSET, self.const.CHAMP_Y_DIFF, self.const.CHAMP_Y_OFFSET)
        self.champ_coords = np.reshape(self.champ_coords, (-1, 2))

        self.item_coords = utils.generateItemCoordinates(self.const.ITEM_X_DIFF, self.const.ITEM_LEFT_X_OFFSET, self.const.ITEM_RIGHT_X_OFFSET, self.const.ITEM_Y_DIFF, self.const.ITEM_Y_OFFSET)
        self.item_coords = np.reshape(self.item_coords, (-1, 2))
        self.item_coords = [(coord[0] + self.const.ITEM_INNER_OFFSET+item_x_offset, coord[1] + self.const.ITEM_INNER_OFFSET+item_y_offset) for coord in self.item_coords]
        
        self.self_coords = utils.generateChampCoordinates(self.const.SELF_INDICATOR_LEFT_X_OFFSET, self.const.SELF_INDICATOR_RIGHT_X_OFFSET, self.const.SELF_INDICATOR_Y_DIFF, self.const.SELF_INDICATOR_Y_OFFSET)
        self.self_coords = np.reshape(self.self_coords, (-1, 2))

        self.champ_graph = tf.Graph()
        with self.champ_graph.as_default():
            champ_network = network.classify_champs(network.CHAMP_IMG_SIZE, train.NUM_CHAMPS, 0.001)
            self.champ_model = tflearn.DNN(champ_network)
            self.champ_model.load(champ_model_path)
        self.item_graph = tf.Graph()
        with self.item_graph.as_default():
            item_network = network.classify_items(network.ITEM_IMG_SIZE, 202, 0.001)
            self.item_model = tflearn.DNN(item_network)
            self.item_model.load(item_model_path)
        self.spell_graph = tf.Graph()
        with self.spell_graph.as_default():
            spell_network = network.classify_spells(network.SPELL_IMG_SIZE, train.NUM_SPELLS, 0.001)
            self.spell_model = tflearn.DNN(spell_network)
            self.spell_model.load(spell_model_path)
        self.self_graph = tf.Graph()
        with self.self_graph.as_default():
            self_network = network.classify_self(network.SELF_IMG_SIZE, train.NUM_SELF, 0.001)
            self.self_model = tflearn.DNN(self_network)
            self.self_model.load(self_model_path)
        self.next_graph = tf.Graph()
        # with self.next_graph.as_default():
        next_network = network.classify_next_item(network.game_config, network.next_network_config)
        self.next_model = tflearn.DNN(next_network, tensorboard_verbose=0)
        self.next_model.load(next_item_model_path)
        self.cvt = utils.Converter()
        print("Complete")

    
            
    def predictElements(self, img, coords, size, model, graph, model_input_size, bw=False, binary=False):
        X = [img[coord[1]:coord[1] + size, coord[0]:coord[0] + size] for coord in coords]
        X = [cv.resize(img, model_input_size, cv.INTER_AREA) for img in X]

        if bw:
            X = [cv.cvtColor(x, cv.COLOR_BGR2GRAY) for x in X]
            X = [np.reshape(x, (*x.shape[:2], 1)) for x in X]

        # counter = 0
        # for i in X:
        #     cv.imshow(str(counter), i)
        #     counter +=1
        # cv.waitKey(0)
         
        with graph.as_default():
            return model.predict(X)

    def predict_sb_elems(self, img):

        champs_int = self.predictElements(img, self.champ_coords, self.const.CHAMP_SIZE, self.champ_model, self.champ_graph,
                                      network.CHAMP_IMG_SIZE, True)

        items_int = self.predictElements(img, self.item_coords, self.const.ITEM_SIZE, self.item_model, self.item_graph,
                                     network.ITEM_IMG_SIZE)

        # spells = self.predictElements(img, self.spell_coords, self.const.SPELL_SIZE, self.spell_mapper, self.spell_model, self.spell_graph,
        #                               network.SPELL_IMG_SIZE)

        self_ = self.predictElements(img, self.self_coords, self.const.SELF_INDICATOR_SIZE, self.self_mapper, self.self_model,
                                     self.self_graph,
                                     network.SELF_IMG_SIZE, True, True)

        champs_str = [self.cvt.champ_int2string_old[champ] for champ in champs_int]
        items_str = [self.cvt.item_int2string_old[champ] for champ in items_int]

        print(champs_str)
        print(items_str)
        print(self_)
        # from string to id
        champs_id = [self.cvt.champ_string2id_dict[champ] for champ in champs_str]
        items_id = [self.cvt.item_string2id_dict[item] for item in items_str]

        return np.array(champs_int), np.array(champs_id), np.array(items_int), np.array(items_id), np.array(self_)

    def predict_next_items(self, X):
        # with next_graph.as_default():
        Y_pred_int = self.next_model.predict(X)
        Y_pred_int = np.reshape(Y_pred_int, [network.game_config["champs_per_team"], network.game_config["total_num_items"]])
        Y_pred_str = []
        counter = 0
        for y in Y_pred_int:
            pred_int = np.argmax(y)
            if pred_int == 0:
                pred_int = np.argmax(y[1:])
                print("High uncertainty!"+counter)
                counter += 1
            pred_str = self.cvt.item_int2string(pred_int)
            pred_id = self.cvt.item_string2id(pred_str)
            Y_pred_str.append(pred_mapped)
            Y_pred_id.append(pred_id)
        return Y_pred_int, Y_pred_id, Y_pred_str

