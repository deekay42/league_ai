import numpy as np
# import pyautogui
import cv2 as cv
# import keyboard
import glob
import os
from predict import Predictor
import time
import network
import configparser


class Main:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read("/Applications/League of Legends.app/Contents/LoL/Config/game.cfg")
        self.res = int(self.config['General']['Width']), int(self.config['General']['Height'])
        self.show_names_in_sb = bool(int(self.config['HUD']['ShowSummonerNamesInScoreboard']))
        self.flipped_sb = bool(int(self.config['HUD']['MirroredScoreboard']))
        self.predict = Predictor(*self.res)


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
    #
    # def item_slots_left(items, summ_id):
    #     counter = -1
    #     for i in items[6 * summ_id:6 * summ_id + 6]:
    #         if i == 0:
    #             counter += 1
    #             break
    #         counter += 1
    #     return counter

    def run(self):

        while True:
            # keyboard.wait('tab')
            print('you pressed tab + f12')
            time.sleep(2)
            # screenshot = take_screenshot()
            screenshot = cv.imread('fds_screenshot_25.07.2018.png')
            champs, items, summ_index = self.predict.predict_sb_elems(screenshot)
            items = np.reshape(items, (-1, 7))
            items = items[:,:network.game_config["items_per_champ"]]
            items = np.ravel(items)
            if summ_index > 4:
                print("switching teams!")
                champs, items = swapTeams(champs, items)
                summ_index -= 5
            # items = swapItems(items)
            # champs = swapChamps(champs)
            next_items_input = np.concatenate([champs, items], axis=0)
            next_items_str, next_items_int = self.predict.predict_next_items([next_items_input])
            print(next_items_str)
            cv.waitKey(0)
            # while True:
            #     empty_item_index = item_slots_left(items, summ_index)
            #     if empty_item_index == -1:
            #         break
            #     items[summ_index*6 + empty_item_index] = next_items_int[summ_index]
            #     next_items_input = np.concatenate([champs, items], axis=0)
            #     next_items_str, next_items_int = predict.predict_next_items([next_items_input])
            #     print(next_items_str)

m = Main()
m.run()