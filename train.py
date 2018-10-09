import constants
import network
import cv2 as cv
import utils
import numpy as np
import json
import glob
from threading import Thread
from queue import Queue
import tflearn
from tflearn.data_utils import shuffle, to_categorical
import data_loader
import tensorflow as tf

num_epochs = 100000
batch_size = 128
num_examples = 1024

NUM_SPELLS = 9
NUM_CHAMPS = 144
NUM_ITEMS = 203
NUM_SELF = 1


def _load_spell_test():
    _, spells, _ = opencv_detect.getScoreboardStatsImgs()
    # spells = [cv.resize(spell, (IMG_SIZE, IMG_SIZE), interpolation=cv.INTER_AREA) for spell in spells]
    keys = ["Flash", "Teleport", "Smite", "Flash", "Flash", "Ignite", "Heal", "Flash", "Flash", "Ignite", "Flash", "Teleport", "Flash", "Smite", "Exhaust", "Flash", "Flash", "Heal", "Flash", "Ignite"]
    images = []
    classes = []
    nextkey = iter(keys)

    for team_index in range(2):
        for champ_index in range(5):
            images += [spells[team_index][champ_index][0]]
            images += [spells[team_index][champ_index][1]]
            classes += [utils.spell2int(next(nextkey)), utils.spell2int(next(nextkey))]

    return np.array(images), np.array(classes)


def load_train_data(img_size, num_elements, loader_func):
    (X, Y) = loader_func(img_size)
    X, Y = shuffle(X, Y)
    Y = np.reshape(Y, (-1,1))
    # Y = to_categorical(Y, nb_classes=num_elements)
    return (X, Y)


def load_corner_train_data(num_examples):
    (X, Y) = generate.getBatch(num_examples, CORNER_IMG_SIZE)
    Y = [coords for coords in Y[:,0]]
    X = [np.reshape(x, (CORNER_IMG_SIZE[1], CORNER_IMG_SIZE[0], 1)) for x in X]
    # X, Y = shuffle(X, Y)
    Y = [coords.flatten() for coords in Y]
    return (X, Y)


def load_test_data():
    (X_test, Y_test) = _load_spell_test()
    Y_test = to_categorical(Y_test, nb_classes=NUM_SPELLS)
    return (X_test, Y_test)


def load_corner_test_data(img_size):
    image_paths = glob.glob('test_data/*.jpg')
    image_paths.sort()
    imgs = [cv.imread(path) for path in image_paths]
    with open('test_data/test_labels.json', "r") as f:
        corners = json.load(f)
    corners = [list(v['corners'].values()) for k, v in corners.items()]
    img_orig_size = imgs[0].shape
    imgs = [cv.resize(img, img_size, interpolation=cv.INTER_AREA) for img in imgs]
    x_scale = img_size[0] / img_orig_size[1]
    y_scale = img_size[1] / img_orig_size[0]
    corners = [(corner[0] * x_scale, corner[1] * y_scale, corner[2] * x_scale, corner[3] * y_scale, corner[4] * x_scale,
                corner[5] * y_scale, corner[6] * x_scale, corner[7] * y_scale) for corner in corners]
    corners = list(map(lambda x: list(map(round, x)), corners))
    return imgs, corners

def load_elems_test_data():
    image_paths = glob.glob('test_data/easy/*.png')
    image_paths.sort()
    imgs = [cv.imread(path) for path in image_paths]
    with open('test_data/easy/test_labels.json', "r") as f:
        elems = json.load(f)
    spells_y = [list(v['spells']) for k, v in elems.items()]
    champs_y = [list(v['champs']) for k, v in elems.items()]
    items_y = [list(v['items']) for k, v in elems.items()]
    self_y = [v['self'] for k, v in elems.items()]
    spells_mapper = dict(zip(utils.getSpellTemplateDict().keys(), utils.init_spell_data_for_training(network.SPELL_IMG_SIZE).keys()))
    champs_mapper = dict(zip(utils.getChampTemplateDict().keys(), utils.init_champ_data_for_training().keys()))
    items_mapper = dict(zip(utils.getItemTemplateDict().keys(), utils.init_item_data_for_training().keys()))
    # spells_y = np.ravel([[spells_mapper[y] for y in spells_y_set] for spells_y_set in spells_y])
    # champs_y = np.ravel([[champs_mapper[y.replace(" ", "")] for y in champs_y_set] for champs_y_set in champs_y])
    # items_y = np.ravel([[items_mapper[y] for y in items_y_set] for items_y_set in items_y])


    champ_coords = utils.generateChampCoordinates(constants.CHAMP_LEFT_X_OFFSET, constants.CHAMP_RIGHT_X_OFFSET,
                                                  constants.CHAMP_Y_DIFF,
                                                  constants.CHAMP_Y_OFFSET)
    item_coords = utils.generateItemCoordinates(constants.ITEM_X_DIFF, constants.ITEM_LEFT_X_OFFSET,
                                                constants.ITEM_RIGHT_X_OFFSET,
                                                constants.ITEM_Y_DIFF,
                                                constants.ITEM_Y_OFFSET)
    spell_coords = utils.generateSpellCoordinates(constants.SPELL_SIZE, constants.SPELL_LEFT_X_OFFSET,
                                                  constants.SPELL_RIGHT_X_OFFSET, constants.SPELL_Y_DIFF,
                                                  constants.SPELL_Y_OFFSET)
    self_coords = utils.generateChampCoordinates(constants.SELF_INDICATOR_LEFT_X_OFFSET,
                                                 constants.SELF_INDICATOR_RIGHT_X_OFFSET,
                                                 constants.SELF_INDICATOR_Y_DIFF,
                                                 constants.SELF_INDICATOR_Y_OFFSET)

    champ_coords = np.reshape(champ_coords, (-1, 2))
    item_coords = np.reshape(item_coords, (-1, 2))
    spell_coords = np.reshape(spell_coords, (-1, 2))
    self_coords = np.reshape(self_coords, (-1, 2))
    item_coords = [(coord[0] + 2, coord[1] + 2) for coord in item_coords]

    spells_x = []
    champs_x = []
    items_x = []
    self_x = []
    for img in imgs:
        spells_x_raw = [img[coord[1]:coord[1] + constants.SPELL_SIZE, coord[0]:coord[0] + constants.SPELL_SIZE] for coord in spell_coords]
        spells_x += [cv.resize(img, network.SPELL_IMG_SIZE, cv.INTER_AREA) for img in spells_x_raw]
        items_x_raw = [img[coord[1]:coord[1] + constants.ITEM_SIZE, coord[0]:coord[0] + constants.ITEM_SIZE] for coord in item_coords]
        items_x += [cv.resize(img, network.ITEM_IMG_SIZE, cv.INTER_AREA) for img in items_x_raw]
        champs_x_raw = [img[coord[1]:coord[1] + constants.CHAMP_SIZE, coord[0]:coord[0] + constants.CHAMP_SIZE] for coord in champ_coords]
        champs_x += [cv.resize(img, network.CHAMP_IMG_SIZE, cv.INTER_AREA) for img in champs_x_raw]
        self_x_raw = [img[coord[1]:coord[1] + constants.SELF_INDICATOR_SIZE, coord[0]:coord[0] + constants.SELF_INDICATOR_SIZE] for coord in self_coords]
        self_x += [cv.resize(img, network.SELF_IMG_SIZE, cv.INTER_AREA) for img in self_x_raw]

    champs_x = [cv.cvtColor(champ, cv.COLOR_BGR2GRAY) for champ in champs_x]
    champs_x = [np.reshape(champ, (*champ.shape[:2], 1)) for champ in champs_x]

    # items_x = [cv.cvtColor(item, cv.COLOR_BGR2GRAY) for item in items_x]
    # items_x = [np.reshape(item, (*item.shape[:2], 1)) for item in items_x]

    self_x = [cv.cvtColor(x, cv.COLOR_BGR2GRAY) for x in self_x]
    self_x = [np.reshape(x, (*x.shape[:2], 1)) for x in self_x]

    # spells_y = to_categorical(spells_y, nb_classes=NUM_SPELLS)
    # champs_y = to_categorical(champs_y, nb_classes=NUM_CHAMPS)
    # items_y = to_categorical(items_y, nb_classes=NUM_ITEMS)
    self_y = to_categorical(self_y, nb_classes=10)
    self_y = np.reshape(self_y, (-1,1))
    return spells_x, items_x, champs_x, self_x, spells_y, items_y, champs_y, self_y

def preprocess(img, med_it, kernel):
    for _ in range(med_it):
        img = cv.medianBlur(img, kernel)
    return img

# X,Y = load_corner_train_data(3)
def train_corner_network():


    network = detect_cornersV2(CORNER_IMG_SIZE)
    # Train using classifier
    X_test, Y_test = load_corner_test_data(CORNER_IMG_SIZE)
    X_pre_test = [preprocess(x) for x in X_test]
    #
    # sample = 0
    # for x,y in zip(X,Y):
    #     cv.imwrite('training_data/'+str(sample)+'.png', x)
    #     with open("training_data/sample"+str(sample), "w") as f:
    #         for coord in y:
    #             f.write(str(coord)+' ')
    #         f.write('\n')
    #         f.flush()
    #     sample += 1

    model = tflearn.DNN(network, tensorboard_verbose=0)
    #
    # x1 = cv.imread('training_data/0.png')
    #
    model.load('./models/my_model1656')
    #
    # y = model.predict([x1])
    # print(y)


    queue = Queue(10)

    class GenerateTrainingData(Thread):
        def run(self):
            global queue
            while True:
                x,y = load_corner_train_data(32)
                queue.put((x,y))

    for _ in range(4):
        GenerateTrainingData().start()

    with open("models/accuracies", "w") as f:
        for epoch in range(num_epochs):
            X, Y = queue.get()
            queue.task_done()
            model.fit(X, Y, n_epoch=1, shuffle=True, validation_set=None,
                      show_metric=True, batch_size=batch_size, run_id='whaddup_glib_globs'+str(epoch))
            # pred1 = model.evaluate(np.array(X_test), np.array(Y_test), batch_size=batch_size)
            # pred2 = model.evaluate(np.array(X_pre_test), np.array(Y_test), batch_size=batch_size)
            # print("Raw test set accuracy: {0:.2f}".format(pred1[0]))
            # print("Preprocessed test set accuracy: {0:.2f}".format(pred2[0]))
            #
            # y = model.predict(X_test)
            # print("Raw test data predictions: {0}".format(y))
            # print("Actual: {0}".format(Y_test))
            model.save('models/my_model'+str(epoch))
            # f.write("Epoch {0} raw images accuracy {1:.2f} | preprocessed images accuracy {2:.2f}\n".format(epoch, pred1[0], pred2[0]))
            # f.flush()

queue = Queue(10)

def train_elements_network(training_data_generator, img_size, num_elements, lr):
    net = network.classify_self(img_size, num_elements, lr)
    model = tflearn.DNN(net, tensorboard_verbose=3)
    # model.load('best_models/items/my_model406')
    # #
    # X,Y = load_train_data(img_size, num_elements, training_data_generator)
    # sample = 0
    # for x,y in zip(X,Y):
    #     cv.imwrite('training_data/'+str(sample)+'.png', x)
    #     with open("training_data/sample"+str(sample), "w") as f:
    #         for coord in y:
    #             f.write(str(coord)+' ')
    #         f.write('\n')
    #         f.flush()
    #     sample += 1
    #
    # Y_pred = model.predict(X)
    # print(Y)
    # print(np.round_(Y_pred))
    #
    # Y_pred_mapped = [champ_imgs2[utils.one_hot2int(y)] for y in Y_pred]
    # print(Y_pred_mapped)

    # print(model.evaluate(X, Y, batch_size=batch_size))

    _,_,_,X_test,_,_,_,Y_test = load_elems_test_data()
    X_pre_test = [preprocess(x, 1, 3) for x in X_test]
    X_pre_test2 = [preprocess(x, 1, 5) for x in X_test]
    X_pre_test3 = [preprocess(x, 2, 3) for x in X_test]
    X_pre_test4 = [preprocess(x, 2, 5) for x in X_test]
    X_pre_test = [np.reshape(x, (*x.shape[:2], 1)) for x in X_pre_test]
    X_pre_test2 = [np.reshape(x, (*x.shape[:2], 1)) for x in X_pre_test2]
    X_pre_test3 = [np.reshape(x, (*x.shape[:2], 1)) for x in X_pre_test3]
    X_pre_test4 = [np.reshape(x, (*x.shape[:2], 1)) for x in X_pre_test4]
    #
    # counter = 0
    # for i in X_test:
    #     cv.imshow(str(counter), i)
    #     counter +=1
    # cv.waitKey(0)

    # for i in range(315,0,-1):
    #     print('model: '+str(i))
    #     model.load('models/my_model'+str(i))
    #     pred1 = model.evaluate(np.array(X_test), np.array(Y_test), batch_size=batch_size)
    #     pred2 = model.evaluate(np.array(X_pre_test), np.array(Y_test), batch_size=batch_size)
    #     print("Raw test set accuracy: {0:.2f}".format(pred1[0]))
    #     print("Preprocessed test set accuracy: {0:.2f}".format(pred2[0]))

    class GenerateTrainingData(Thread):
        def run(self):
            global queue
            while True:
                x, y = load_train_data(img_size, num_elements, training_data_generator)
                x = [cv.cvtColor(x_, cv.COLOR_BGR2GRAY) for x_ in x]
                x = [np.reshape(x, (*img_size, 1)) for x in x]
                queue.put((x, y))

    for _ in range(3):
        GenerateTrainingData().start()

    # champ_mapper = dict(zip(generate.init_champ_data().keys(), utils.getChampTemplateDict().keys()))

    with open("models/accuracies", "w") as f:

        class MonitorCallback(tflearn.callbacks.Callback):

            def on_epoch_end(self, training_state):
                f.write("Epoch {0} accuracy {1:.2f} | loss {2:.2f}\n".format(training_state.epoch,training_state.acc_value, training_state.global_loss))
                f.flush()

        monitorCallback = MonitorCallback()
        for epoch in range(num_epochs):
            X, Y = queue.get()
            queue.task_done()
            model.fit(X, Y, n_epoch=1, shuffle=True, validation_set=None,
                      show_metric=True, batch_size=batch_size, run_id='whaddup_glib_globs'+str(epoch), callbacks=monitorCallback)
            pred1 = model.evaluate(np.array(X_test), np.array(Y_test), batch_size=batch_size)
            pred2 = model.evaluate(np.array(X_pre_test), np.array(Y_test), batch_size=batch_size)
            pred3 = model.evaluate(np.array(X_pre_test2), np.array(Y_test), batch_size=batch_size)
            pred4 = model.evaluate(np.array(X_pre_test3), np.array(Y_test), batch_size=batch_size)
            pred5 = model.evaluate(np.array(X_pre_test4), np.array(Y_test), batch_size=batch_size)
            print("Raw test set accuracy: {0:.2f}".format(pred1[0]))
            print("Preprocessed test set accuracy: {0:.2f}".format(pred2[0]))
            print("Preprocessed test set accuracy: {0:.2f}".format(pred3[0]))
            print("Preprocessed test set accuracy: {0:.2f}".format(pred4[0]))
            print("Preprocessed test set accuracy: {0:.2f}".format(pred5[0]))


            y = model.predict(X_test)
            # y = [np.argmax(y_) for y_ in y]
            # y_test = [np.argmax(y_) for y_ in Y_test]
            print("Pred Actual")
            for i in range(len(y)):
                print("{2}: {0:.2f}     {1}".format(y[i][0],Y_test[i][0], i%10))
            # print("Raw test data predictions: {0}".format(y))
            # print("Actual test data  values : {0}".format(Y_test))
            model.save('models/my_model'+str(epoch+1))
            f.write("Epoch {0} raw images accuracy {1:.2f} | preprocessed images accuracy {2:.2f}\n".format(epoch + 1,
                                                                                                            pred1[0],
                                                                                                            pred2[0]))
            f.flush()

def train_spell_network():
    spell_imgs = utils.init_spell_data_for_training(SPELL_IMG_SIZE)
    training_data_generator = lambda img_size: generate.generate_training_data(spell_imgs, 100, img_size)
    train_elements_network(training_data_generator, SPELL_IMG_SIZE, NUM_SPELLS, 0.001)

def train_champ_network():
    champ_imgs = utils.init_champ_data_for_training()
    training_data_generator = lambda img_size: generate.generate_training_data(champ_imgs, 100, img_size)
    train_elements_network(training_data_generator, CHAMP_IMG_SIZE, NUM_CHAMPS, 0.001)

def train_item_network():
    item_imgs = utils.init_item_data_for_training()
    training_data_generator = lambda img_size: generate.generate_training_data(item_imgs, 100, img_size)
    train_elements_network(training_data_generator, ITEM_IMG_SIZE, NUM_ITEMS, 0.001)

def train_self_network():
    self_imgs = utils.init_self_data_for_training()
    training_data_generator = lambda img_size: generate.generate_training_data(self_imgs, 1024, img_size)
    train_elements_network(training_data_generator, network.SELF_IMG_SIZE, NUM_SELF, 0.01)

def train_positions_network():

    # model.load('./models/my_model1')
    print("Loading training data")
    dataloader = data_loader.PositionsDataLoader()
    print("Encoding training data")
    X, Y = dataloader.get_train_data()

    print("Encoding test data")
    X_test, Y_test = dataloader.get_test_data()


    # for i in range(1,44):
    #     with tf.Graph().as_default():
    #         net = network.classify_positions(network.positions_game_config, network.positions_network_config)
    #         model = tflearn.DNN(net, tensorboard_verbose=0)
    #         model.load('./position_models/models_improved/positions/my_model'+str(i))
    #         pred1 = model.evaluate(X_test, Y_test, batch_size=batch_size)
    #         print(i)
    #         print("eval is {0:.4f}".format(pred1[0]))

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        tflearn.is_training(True, session=sess)
    # with tf.device('/device:GPU:0'):
        print("Building model")
        net = network.classify_positions(network.positions_game_config, network.positions_network_config)
        model = tflearn.DNN(net, tensorboard_verbose=0, session=sess)
        sess.run(tf.global_variables_initializer())
        print("Commencing training")
        with open("models/positions/accuracies", "w") as f:
            class MonitorCallback(tflearn.callbacks.Callback):

                def on_epoch_end(self, training_state):
                    f.write("Epoch {0} train accuracy {1:.4f} | loss {2:.4f}\n".format(training_state.epoch,training_state.acc_value, training_state.global_loss))
                    f.flush()
                    pass

            monitorCallback = MonitorCallback()
            for epoch in range(num_epochs):
                model.fit(X,Y, n_epoch=1, shuffle=True, validation_set=None,
                          show_metric=True, batch_size=batch_size, run_id='whaddup_glib_globs'+str(epoch), callbacks=monitorCallback)
                pred1 = model.evaluate(X_test, Y_test, batch_size=batch_size)
                print("eval is {0:.4f}".format(pred1[0]))

                model.save('models/positions/my_model' + str(epoch + 1))
                f.write("Epoch {0} eval accuracy {1:.4f}\n".format(epoch + 1, pred1[0]))
                f.flush()


def train_elements_network():

    # model.load('./models/my_model1')
    print("Loading training data")
    dataloader = data_loader.NextItemsDataLoader()
    print("Encoding training data")
    X, Y = dataloader.get_train_data()

    print("Encoding test data")
    X_test, Y_test = dataloader.get_test_data()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        tflearn.is_training(True, session=sess)
    # with tf.device('/device:GPU:0'):
        print("Building model")
        net = network.classify_next_item(network.game_config, network.next_network_config)
        model = tflearn.DNN(net, tensorboard_verbose=0, session=sess)
        sess.run(tf.global_variables_initializer())
        print("Commencing training")
        with open("models/accuracies", "w") as f:
            class MonitorCallback(tflearn.callbacks.Callback):

                def on_epoch_end(self, training_state):
                    f.write("Epoch {0} train accuracy {1:.4f} | loss {2:.4f}\n".format(training_state.epoch,training_state.acc_value, training_state.global_loss))
                    f.flush()

            monitorCallback = MonitorCallback()
            for epoch in range(num_epochs):
                model.fit(X,Y, n_epoch=1, shuffle=True, validation_set=None,
                          show_metric=True, batch_size=batch_size, run_id='whaddup_glib_globs'+str(epoch), callbacks=monitorCallback)
                pred1 = model.evaluate(X_test, Y_test, batch_size=batch_size)
                print("eval is {0:.4f}".format(pred1[0]))
                # prediction = model.predict(X)
                # print("Prediction 1 is")
                # for i,j in zip(prediction[0], Y[0]):
                #     print("{0:.2f} {1:.2f}".format(i,j))
                # print("Prediction 2 is")
                # for i,j in zip(prediction[1], Y[1]):
                #     print("{0:.2f} {1:.2f}".format(i,j))


                model.save('models/my_model' + str(epoch + 1))
                f.write("Epoch {0} eval accuracy {1:.4f}\n".format(epoch + 1, pred1[0]))
                f.flush()

if __name__ == '__main__':
    train_elements_network()