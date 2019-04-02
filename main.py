import numpy as np
#import pyautogui
import cv2 as cv
#import keyboard
import glob
import os
from predict import Predictor
import time
import network
import configparser
import cassiopeia as cass
import cassiopeia_championgg
import utils
from build_path import build_path
import copy
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import traceback
from tkinter import Tk
from tkinter.filedialog import askdirectory
from tkinter import messagebox


class Main(FileSystemEventHandler):
    def configurePredictor(self, loldir):
        self.config = configparser.ConfigParser()
        self.config.read(loldir+"/Config/game.cfg")
        self.res = int(self.config['General']['Width']), int(self.config['General']['Height'])
        self.show_names_in_sb = bool(int(self.config['HUD']['ShowSummonerNamesInScoreboard']))
        self.flipped_sb = bool(int(self.config['HUD']['MirroredScoreboard']))
        if self.flipped_sb:
            Tk().withdraw()
            messagebox.showinfo("Error","League IQ does not work if the scoreboard is mirrored. Please untick the \"Mirror Scoreboard\" checkbox in the game settings (Press Esc while in-game)")
            raise Exception("League IQ does not work if the scoreboard is mirrored.")
        self.predict = Predictor(*self.res, self.show_names_in_sb)
        print(f"Res is {self.res}")
        #self.predict = Predictor(1440,810, True)
        self.cvt = utils.Converter()


    @staticmethod
    def take_screenshot():
        image = pyautogui.screenshot()
        image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
        return image

    @staticmethod
    def take_windows_screenshot():
        folder = 'L:\Spiele\lol\Screenshots\*'
        list_of_files = glob.glob(folder)
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

    def analyzeChamp(self, role, champs_id_cpy, champs_int_cpy, items_id_cpy, items_int_cpy):
        
        if role > 4:
            print("switching teams!")
            champs_int_cpy, items_int_cpy = self.swapTeams(champs_int_cpy, items_int_cpy)
            champs_id_cpy, items_id_cpy = self.swapTeams(champs_id_cpy, items_id_cpy)
            role -= 5

    
        print("\nRole: " + str(role))
        champs_id, champs_int, items_id, items_int = copy.deepcopy(champs_id_cpy), copy.deepcopy(
            champs_int_cpy), copy.deepcopy(items_id_cpy), copy.deepcopy(items_int_cpy)
        

        summ_next_item_cass = None
        result = []
        items_ahead = 0
        while summ_next_item_cass == None or items_ahead <= 4:   
            items_ahead += 1
            next_items_input = np.concatenate([[role], champs_int, items_int], axis=0)
            next_items_int, next_items_id, next_items_str = self.predict.predict_next_items([next_items_input])
            # print("Next item predicted for role is: ")
            # print(next_items_str)
            summ_curr_items = items_id[role * 6:role * 6 + 6]
            next_items, _, abs_items, _ = build_path(summ_curr_items, cass.Item(id=next_items_id, region="KR"))
            result.extend(next_items)
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
                print("Max items reached!!")
                break

        return result
       


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

    def on_created(self, event):
        file_path = event.src_path
        print("Got event for file %s" % file_path)

        oldsize = -1
        while True:
            size = os.path.getsize(file_path)
            if size == oldsize:
                break
            else:
                oldsize = size
                time.sleep(0.05)
                
        try:
            self.processImage(event.src_path)
        except Exception as e:
            print(e)
        

    def processImage(self, img_path):

        # for current in [sorted(glob.glob('screenies/streamers/*'))[0]]:
            # keyboard.wait('tab')
            print('you pressed tab + f12 '+img_path)
            
            # screenshot = self.take_windows_screenshot()

            try:              
                print("Now trying to predict image")
                screenshot = cv.imread(img_path)
                champs_int, champs_id, items_int, items_id, summ_index = self.predict.predict_sb_elems(screenshot)
                # champs_id = [92,104,136,17,163,516,64,34,432,117]
                # champs_int = [20,30,115,88,87,48,116,93,69,108]
                # items_id = [3133,2003,1055,0,0,0,3340,1041,2031,0,0,0,0,3340
                # ,2033,2403,0,0,0,0,3340,1056,2003,0,0,0,0,3340
                # ,3303,2003,2423,0,0,0,3340,2033,3024,0,0,0,0,3041
                # ,1041,2031,0,0,0,0,3041,1056,3024,2003,2423,0,0,3041
                # ,1036,2031,0,0,0,0,3041,3303,2003,0,0,0,0,3041]
                # items_int = [133,37,23,0,0,0,169,16,42,0,0,0,0,169,44,58,0,0
                # ,0,0,169,24,37,0,0,0,0,169,168,37,60,0,0,0,169,44
                # ,70,0,0,0,0,81,16,42,0,0,0,0,81,24,70,37,60,0
                # ,0,81,12,42,0,0,0,0,81,168,37,0,0,0,0,81]
                # summ_index = 0

            except FileNotFoundError as e:
                print(e)
                return
            except Exception as e:
                print(e)
                traceback.print_exc()
                return

            # print(champs_id)
            # print(champs_int)
            # print(items_id)
            # print(items_int)
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
            result = self.analyzeChamp(summ_index, champs_id_cpy, champs_int_cpy, items_id_cpy, items_int_cpy)
            result_id = [item.id for item in result]
            result_int = [self.cvt.item_id2int(item) for item in result_id]
            print("This is the result: ")
            print(result_id)
            print(result_int)
            
            with open("last", "w") as f:
                if result_int[0]:
                    f.write(str(result_int[0]))
                for item in result_int[1:]:
                    f.write(","+str(item))
                    
    def query_lol_dir(self):  
        Tk().withdraw()
        messagebox.showinfo("Information","We were unable to locate your League of Legends installation. Please select your main League of Legends folder.")
        loldir = askdirectory(initialdir = "C:", title = "Please select your main League of Legends folder")
        while not os.path.isdir(loldir + "/RADS"):
            messagebox.showinfo("Information","That wasn't it. Select the folder that has the RADS folder in it.")
            loldir = askdirectory(initialdir = "C:", title = "Please select your main League of Legends folder")
        print(loldir)
        screenshot_dir = loldir+"/Screenshots"
        try:
            os.makedirs(screenshot_dir)
        except Exception as e:
            print(e)
        with open("loldir", "w") as f:
            f.write(loldir)
    
    def get_lol_dir(self):
        while True:
            if os.path.exists("loldir"):

                with open("loldir", "r") as f:
                    loldir = f.read()
            else:
                loldir = "C:/Riot Games/League of Legends"
            if not os.path.isdir(loldir + "/RADS"):
                self.query_lol_dir()
            else: 
                return loldir

        
            


if __name__=="__main__":
    print("In the main function")
    tmp = cass.Item(id=2033, region="KR")
    tmp0 = tmp.builds_from
    m = Main()
    loldir = m.get_lol_dir()
    m.configurePredictor(loldir)
    observer = Observer()
    observer.schedule(m, path=loldir+"/Screenshots")
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()




# champs_id = [516,  33, 134, 119,  25,  36,  76,  41,  29,  40]
#         champs_int = [ 48,  10,  55, 111 , 86 , 56  , 6  ,22 , 99 ,129]
#         items_id = [3111, 3076, 3751, 3024, 2033, 1028, 1401, 3111, 1011, 3076, 1031, 3133, 3285, 3916, 3020, 1026, 2033, 3382, 3095, 1038, 3006, 1053, 1055, 3133, 3191, 3009, 3108, 3098, 2423, 0, 3065, 3111, 3751, 1031, 1054, 0, 1401, 3020, 3191, 3108, 2031, 0, 3078, 3036, 3009, 2033, 1036, 1055, 3095, 3086, 3006, 1055, 1083, 1042, 3504, 3009, 3098, 3113, 3067, 1004]
#         items_int = [125, 101, 191, 70, 44, 8, 30, 125, 4, 101, 10, 133, 165, 202, 68, 6, 44, 178, 113, 14, 65, 21, 23, 133, 155, 66, 122, 116, 60, 0, 92, 125, 191, 10, 22, 0, 30, 68, 155, 122, 42, 0, 103, 79, 66, 44, 12, 23, 113, 107, 65, 23, 28, 17, 183, 66, 116, 126, 93, 2]

# champs_id = [516,  33, 134, 119,  25,  36,  76,  41,  29,  40]
# champs_int = [ 48,  10,  55, 111 , 86 , 56  , 6  ,22 , 99 ,129]
# items_id = [3111, 3076, 3751, 3024, 2033, 1028, 1401, 3111, 1011, 3076, 1031, 3133, 3285, 3165, 3020, 3089, 3135, 3382, 3095, 1038, 3006, 1053, 1055, 3133, 3191, 3009, 3108, 3098, 2423, 0, 3065, 3111, 3068, 0, 1054, 0, 1401, 3020, 3157, 0, 2031, 0, 3078, 3036, 3009, 2033, 3142, 0, 3095, 3085, 3006, 1055, 1083, 1042, 3504, 3009, 3098, 3113, 3067, 1004]
# items_int = [125, 101, 191, 70, 44, 8, 30, 125, 4, 101, 10, 133, 165, 152, 68, 109, 135, 178, 113, 14, 65, 21, 23, 133, 155, 66, 122, 116, 60, 0, 92, 125, 94, 0, 22, 0, 30, 68, 150, 0, 42, 0, 103, 79, 66, 44, 139, 0, 113, 106, 65, 23, 28, 17, 183, 66, 116, 126, 93, 2]
