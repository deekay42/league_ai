import numpy as np
import pyautogui
import cv2 as cv
import keyboard
import glob
import os
from predict import Predictor
import time
import network
import configparser
import cassiopeia as cass
import utils
from build_path import build_path
import copy


class Main:
    def __init__(self):
        self.config = configparser.ConfigParser()
        # self.config.read("L:\Spiele\lol\Config\game.cfg")
        # self.res = int(self.config['General']['Width']), int(self.config['General']['Height'])
        # self.show_names_in_sb = bool(int(self.config['HUD']['ShowSummonerNamesInScoreboard']))
        # self.flipped_sb = bool(int(self.config['HUD']['MirroredScoreboard']))
        # self.predict = Predictor(*self.res)
        self.predict = Predictor(1440,810)
        self.cvt = utils.Converter()


    @staticmethod
    def take_screenshot():
        image = pyautogui.screenshot()
        image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
        return image

    @staticmethod
    def take_windows_screenshot():
        folder = 'L:\Spiele\lol\Screenshots\*'
        list_of_files = glob.glob(folder)  # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        print(latest_file)
        screenshot = cv.imread(latest_file)
        return screenshot

    # top, jg, mid, bot, sup
    # to
    # top, mid, jg, bot, sup
    @staticmethod
    def swapItems(items):
        items = np.reshape(items, [10, 6])
        tmp = np.copy(items[1])
        items[1] = items[2]
        items[2] = tmp

        tmp = np.copy(items[6])
        items[6] = items[7]
        items[7] = tmp
        items = np.ravel(items)

        return items

    @staticmethod
    def swapChamps(champs):
        champs = np.reshape(champs, [10])
        tmp = np.copy(champs[1])
        champs[1] = champs[2]
        champs[2] = tmp
        tmp = np.copy(champs[6])
        champs[6] = champs[7]
        champs[7] = tmp
        champs = np.ravel(champs)

        return champs

    @staticmethod
    def swapTeams(champs, items):
        tmp = np.copy(champs[:5])
        champs[:5] = champs[5:]
        champs[5:] = tmp

        tmp = np.copy(items[:30])
        items[:30] = items[30:]
        items[30:] = tmp

        return champs, items

    def analyzeGame(self, champs_id_cpy, champs_int_cpy, items_id_cpy, items_int_cpy):
        for i in range(2):
            # if summ_index > 4:
            #     print("switching teams!")
            #     champs_int, items_int = self.swapTeams(champs_int, items_int)
            #     champs_id, items_id = self.swapTeams(champs_id, items_id)
            #     summ_index -= 5

            # items_int = self.swapItems(items_int)
            # items_id = self.swapItems(items_id)
            # champs_int = self.swapChamps(champs_int)
            print("\n")
            for role in range(5):
                print("\nRole: " + str(role))
                champs_id, champs_int, items_id, items_int = copy.deepcopy(champs_id_cpy), copy.deepcopy(
                    champs_int_cpy), copy.deepcopy(items_id_cpy), copy.deepcopy(items_int_cpy)
                if i:
                    champs_int, items_int = self.swapTeams(champs_int, items_int)
                    champs_id, items_id = self.swapTeams(champs_id, items_id)
                    # items_int[12] = 96
                    # items_id[12] = 3070

                summ_next_item_cass = None
                while summ_next_item_cass == None or summ_next_item_cass.tier < 3:
                    next_items_input = np.concatenate([[role], champs_int, items_int], axis=0)
                    next_items_int, next_items_id, next_items_str = self.predict.predict_next_items([next_items_input])
                    # print(next_items_str[summ_index])
                    summ_curr_items = items_id[role * 6:role * 6 + 6]
                    next_items, _, abs_items, _ = build_path(summ_curr_items, cass.Item(id=next_items_id, region="KR"))
                    for next_item in next_items:
                        print(self.cvt.item_id2string(next_item.id))

                    abs_items[-1] = list(filter(lambda a: a != 0, abs_items[-1]))
                    try:
                        items_id[role * 6:role * 6 + 6] = np.pad(abs_items[-1], (0, 6 - len(abs_items[-1])),
                                                                 'constant',
                                                                 constant_values=(
                                                                     0, 0))
                        new_summ_items_int = [self.cvt.item_id2int(item) for item in abs_items[-1]]
                        items_int[role * 6:role * 6 + 6] = np.pad(new_summ_items_int, (0, 6 - len(new_summ_items_int)),
                                                                  'constant',
                                                                  constant_values=(
                                                                      0, 0))
                        summ_next_item_cass = cass.Item(id=next_items_id, region="KR")
                    except:
                        break

    def run(self):

        for current in sorted(glob.glob('screenies/streamers/*')):
            # keyboard.wait('tab')
            print('you pressed tab + f12 '+current)
            time.sleep(2)
            # screenshot = self.take_windows_screenshot()

            screenshot = cv.imread(current)
            champs_int, champs_id, items_int, items_id, summ_index = self.predict.predict_sb_elems(screenshot)

            # TODO: Remove control wards from input. Network isn't trained on those.
            # replace seraphs with archangels
            # and muramana with manamune
            items_int = np.reshape(items_int, (-1, 7))
            items_int = items_int[:, :network.game_config["items_per_champ"]]
            items_int = np.ravel(items_int)
            items_int = [1 if item == 59 else 0 if item == 46 else 64 if item==82 else 63 if item==80 else item for item in items_int]

            items_id = np.reshape(items_id, (-1, 7))
            items_id = items_id[:, :network.game_config["items_per_champ"]]
            items_id = np.ravel(items_id)
            items_id = [1001 if item == 2422 else 0 if item == 2055 else 3004 if item == 3042 else 3003 if item == 3040 else item for item in
                        items_id]

            champs_id_cpy, champs_int_cpy, items_id_cpy, items_int_cpy = copy.deepcopy(champs_id), copy.deepcopy(champs_int), copy.deepcopy(items_id), copy.deepcopy(items_int)
            self.analyzeGame(champs_id_cpy, champs_int_cpy, items_id_cpy, items_int_cpy)



m = Main()
# m.run()


# champs_id = [516,  33, 134, 119,  25,  36,  76,  41,  29,  40]
#         champs_int = [ 48,  10,  55, 111 , 86 , 56  , 6  ,22 , 99 ,129]
#         items_id = [3111, 3076, 3751, 3024, 2033, 1028, 1401, 3111, 1011, 3076, 1031, 3133, 3285, 3916, 3020, 1026, 2033, 3382, 3095, 1038, 3006, 1053, 1055, 3133, 3191, 3009, 3108, 3098, 2423, 0, 3065, 3111, 3751, 1031, 1054, 0, 1401, 3020, 3191, 3108, 2031, 0, 3078, 3036, 3009, 2033, 1036, 1055, 3095, 3086, 3006, 1055, 1083, 1042, 3504, 3009, 3098, 3113, 3067, 1004]
#         items_int = [125, 101, 191, 70, 44, 8, 30, 125, 4, 101, 10, 133, 165, 202, 68, 6, 44, 178, 113, 14, 65, 21, 23, 133, 155, 66, 122, 116, 60, 0, 92, 125, 191, 10, 22, 0, 30, 68, 155, 122, 42, 0, 103, 79, 66, 44, 12, 23, 113, 107, 65, 23, 28, 17, 183, 66, 116, 126, 93, 2]

# champs_id = [516,  33, 134, 119,  25,  36,  76,  41,  29,  40]
# champs_int = [ 48,  10,  55, 111 , 86 , 56  , 6  ,22 , 99 ,129]
# items_id = [3111, 3076, 3751, 3024, 2033, 1028, 1401, 3111, 1011, 3076, 1031, 3133, 3285, 3165, 3020, 3089, 3135, 3382, 3095, 1038, 3006, 1053, 1055, 3133, 3191, 3009, 3108, 3098, 2423, 0, 3065, 3111, 3068, 0, 1054, 0, 1401, 3020, 3157, 0, 2031, 0, 3078, 3036, 3009, 2033, 3142, 0, 3095, 3085, 3006, 1055, 1083, 1042, 3504, 3009, 3098, 3113, 3067, 1004]
# items_int = [125, 101, 191, 70, 44, 8, 30, 125, 4, 101, 10, 133, 165, 152, 68, 109, 135, 178, 113, 14, 65, 21, 23, 133, 155, 66, 122, 116, 60, 0, 92, 125, 94, 0, 22, 0, 30, 68, 150, 0, 42, 0, 103, 79, 66, 44, 139, 0, 113, 106, 65, 23, 28, 17, 183, 66, 116, 126, 93, 2]
