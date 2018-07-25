import cv2 as cv

import constants
import utils
import tflearn
import network
import numpy as np
import tensorflow as tf
import train

class Predictor:
    def __init__(self):
    
        x,y = utils.getResolution()
        if (x,y) == (1440,900):
            const = constants.Res_1440_900()
        elif (x,y) == (1600,900):
            const = constants.Res_1600_900()
        elif (x,y) == (1920,1080):
            print("Using resolution 1920,1080")
            const = constants.Res_1920_1080()

        print("Initializing neural networks...")
        champ_model_path = './best_models/champs/my_model'
        item_model_path = './best_models/items/my_model'
        spell_model_path = './best_models/spells/my_model'
        self_model_path = './best_models/self/my_model'

        self.champ_mapper = utils.champ_int2string()
        self.item_mapper = utils.item_int2string()
        self.spell_mapper = utils.int2spell()
        self.self_mapper = [0,1]
        
        my_champ_leftx_offset = utils.cvtHrzt(const.CHAMP_LEFT_X_OFFSET, x)
        my_champ_rightx_offset = utils.cvtHrzt(const.CHAMP_RIGHT_X_OFFSET, x)
        my_champ_ydiff = utils.cvtVert(const.CHAMP_Y_DIFF, y)
        my_champ_yoffset = utils.cvtVert(const.CHAMP_Y_OFFSET, y)
        
        self.champ_coords = utils.generateChampCoordinates(my_champ_leftx_offset, my_champ_rightx_offset, my_champ_ydiff, my_champ_yoffset)
        self.champ_coords = np.reshape(self.champ_coords, (-1, 2))
        
        
        my_item_leftx_offset = utils.cvtHrzt(const.ITEM_LEFT_X_OFFSET, x)
        my_item_rightx_offset = utils.cvtHrzt(const.ITEM_RIGHT_X_OFFSET, x)
        my_item_ydiff = utils.cvtVert(const.ITEM_Y_DIFF, y)
        my_item_yoffset = utils.cvtVert(const.ITEM_Y_OFFSET, y)
        
        self.item_coords = utils.generateItemCoordinates(const.ITEM_X_DIFF, my_item_leftx_offset, my_item_rightx_offset, my_item_ydiff, my_item_yoffset)
        self.item_coords = np.reshape(self.item_coords, (-1, 2))
        self.item_coords = [(coord[0] + const.ITEM_INNER_OFFSET, coord[1] + const.ITEM_INNER_OFFSET) for coord in self.item_coords]
        
        my_spell_leftx_offset = utils.cvtHrzt(const.SPELL_LEFT_X_OFFSET, x)
        my_spell_rightx_offset = utils.cvtHrzt(const.SPELL_RIGHT_X_OFFSET, x)
        my_spell_ydiff = utils.cvtVert(const.SPELL_Y_DIFF_LARGE, y)
        my_spell_yoffset = utils.cvtVert(const.SPELL_Y_OFFSET, y)
        
        self.spell_coords = utils.generateSpellCoordinatesLarge(const.SPELL_SIZE, my_spell_leftx_offset, my_spell_rightx_offset, my_spell_ydiff, my_spell_yoffset)
        self.spell_coords = np.reshape(self.spell_coords, (-1, 2))

        
        my_self_leftx_offset = utils.cvtHrzt(const.SELF_INDICATOR_LEFT_X_OFFSET, x)
        my_self_rightx_offset = utils.cvtHrzt(const.SELF_INDICATOR_RIGHT_X_OFFSET, x)
        my_self_ydiff = utils.cvtVert(const.SELF_INDICATOR_Y_DIFF, y)
        my_self_yoffset = utils.cvtVert(const.SELF_INDICATOR_Y_OFFSET, y)
        
        self.self_coords = utils.generateChampCoordinates(my_self_leftx_offset, my_self_rightx_offset, my_self_ydiff, my_self_yoffset)
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

        self.const = const
        print("Complete")

    
            
    def predictElements(self, img, coords, size, mapper, model, graph, model_input_size, bw=False, binary=False):
        X = [img[coord[1]:coord[1] + size, coord[0]:coord[0] + size] for coord in coords]
        X = [cv.resize(img, model_input_size, cv.INTER_AREA) for img in X]

        if bw:
            X = [cv.cvtColor(x, cv.COLOR_BGR2GRAY) for x in X]
            X = [np.reshape(x, (*x.shape[:2], 1)) for x in X]

        counter = 0
        # for i in X:
            # cv.imshow(str(counter), i)
            # counter +=1
        # cv.waitKey(0)
         
        with graph.as_default():
            
            Y_pred = model.predict([X[0]])
            
            
            # Y_pred = model.predict(X)
            if not binary:
                Y_pred_mapped  = [mapper[np.argmax(y)] for y in Y_pred]
            else:
                Y_pred_mapped = np.argmax(Y_pred)
            print(Y_pred_mapped)    
            cv.imshow('lol', X[0])
            cv.waitKey(0)
            
            return Y_pred_mapped

    def __call__(self, img):
        return self.predictSBElems(img)

    def predictSBElems(self, img):

        champs = self.predictElements(img, self.champ_coords, self.const.CHAMP_SIZE, self.champ_mapper, self.champ_model, self.champ_graph,
                                      network.CHAMP_IMG_SIZE, True)

        items = self.predictElements(img, self.item_coords, self.const.ITEM_SIZE, self.item_mapper, self.item_model, self.item_graph,
                                     network.ITEM_IMG_SIZE)

        spells = self.predictElements(img, self.spell_coords, self.const.SPELL_SIZE, self.spell_mapper, self.spell_model, self.spell_graph,
                                      network.SPELL_IMG_SIZE)

        self_ = self.predictElements(img, self.self_coords, self.const.SELF_INDICATOR_SIZE, self.self_mapper, self.self_model,
                                     self.self_graph,
                                     network.SELF_IMG_SIZE, True, True)

        return champs, spells, items, self_