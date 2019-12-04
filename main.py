import configparser
import os
import time
import traceback
from tkinter import Tk
from tkinter import messagebox
import cv2 as cv
import numpy as np
import cProfile
import io
import pstats
import copy
import glob
import json
 
import cassiopeia as cass
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from utils import utils
from train_model.model import ChampImgModel, ItemImgModel, SelfImgModel, NextItemEarlyGameModel, CSImgModel, \
    KDAImgModel, CurrentGoldImgModel, LvlImgModel, MultiTesseractModel
from utils.artifact_manager import ChampManager, ItemManager, SimpleManager
from utils.build_path import build_path
from constants import ui_constants, game_constants, app_constants
import functools
from train_model import data_loader




class Main(FileSystemEventHandler):

    def __init__(self):
        self.onTimeout = False
        self.loldir = utils.get_lol_dir()
        self.config = configparser.ConfigParser()
        self.config.read(self.loldir + os.sep +"Config" + os.sep + "game.cfg")
        try:
            # res = 1440,810
            res = int(self.config['General']['Width']), int(self.config['General']['Height'])
        except KeyError as e:
            print(repr(e))
            res = 1366, 768
            print("Couldn't find Width or Height sections")

        try:
            show_names_in_sb = bool(int(self.config['HUD']['ShowSummonerNamesInScoreboard']))
        except KeyError as e:
            print(repr(e))
            show_names_in_sb = False
        
        try:
            flipped_sb = bool(int(self.config['HUD']['MirroredScoreboard']))
        except KeyError as e:
            print(repr(e))
            flipped_sb = False
        
        try:
            hud_scale = float(self.config['HUD']['GlobalScale'])
        except KeyError as e:
            print(repr(e))
            hud_scale = 0.5
        
        
        if flipped_sb:
            Tk().withdraw()
            messagebox.showinfo("Error",
                                "League IQ does not work if the scoreboard is mirrored. Please untick the \"Mirror Scoreboard\" checkbox in the game settings (Press Esc while in-game)")
            raise Exception("League IQ does not work if the scoreboard is mirrored.")
        self.res_converter = ui_constants.ResConverter(*res, hud_scale, show_names_in_sb)
        print(f"Res is {res}")

        self.item_manager = ItemManager()
        if Main.shouldTerminate():
            return
        self.next_item_model = NextItemEarlyGameModel()
        self.next_item_model.load_model()
        if Main.shouldTerminate():
            return
        self.champ_img_model = ChampImgModel(self.res_converter)
        self.champ_img_model.load_model()
        if Main.shouldTerminate():
            return
        self.item_img_model = ItemImgModel(self.res_converter)
        self.item_img_model.load_model()
        if Main.shouldTerminate():
            return
        self.self_img_model = SelfImgModel(self.res_converter)
        self.self_img_model.load_model()


        self.kda_img_model = KDAImgModel(self.res_converter)
        self.kda_img_model.load_model()
        self.tesseract_models = MultiTesseractModel([LvlImgModel(self.res_converter),
                                                     CSImgModel(self.res_converter),
                                                     CurrentGoldImgModel(self.res_converter)])

        Main.test_connection()

    def set_res_converter(self, res_cvt):
        self.res_converter = res_cvt
        self.champ_img_model.res_converter = res_cvt
        self.item_img_model.res_converter = res_cvt
        self.self_img_model.res_converter = res_cvt
        self.kda_img_model.res_converter = res_cvt
        for model in self.tesseract_models.tesseractmodels:
            model.res_converter = res_cvt

    @staticmethod
    def test_connection(timeout=0):
        try:
            lol = build_path([], cass.Item(id=3040, region="KR"))
        except Exception as e:
            print(f"Connection error. Retry in {timeout}")
            time.sleep(timeout)
            Main.test_connection(5)


    @staticmethod
    def swap_teams(champs, items):
        tmp = np.copy(champs[:5])
        champs[:5] = champs[5:]
        champs[5:] = tmp

        tmp = np.copy(items[:30])
        items[:30] = items[30:]
        items[30:] = tmp
        return champs, items


    def summoner_items_slice(self, role):
        return np.s_[role * game_constants.MAX_ITEMS_PER_CHAMP:role * game_constants.MAX_ITEMS_PER_CHAMP + game_constants.MAX_ITEMS_PER_CHAMP]


    def predict_next_item(self, role, champs, items, cs, lvl, kda, current_gold):
        champs_int = [int(champ["int"]) for champ in champs]
        items_id = [[int(item["id"]) for item in summ_items] for summ_items in items]

        return self.next_item_model.predict_easy(role, champs_int, items_id, cs, lvl, kda, current_gold)


    def build_path(self, items, next_item, role):
        items_id = [int(item["main_img"]) if "main_img" in item else int(item["id"]) for item in items]
        summ_curr_items = items_id[self.summoner_items_slice(role)]

        #TODO: this is bad. the item class should know when to return main_img or id
        next_items, _, abs_items, _ = build_path(summ_curr_items, cass.Item(id=(int(next_item["main_img"]) if "main_img" in next_item else int(next_item["id"])), region="KR"))
        next_items = [self.item_manager.lookup_by("id", str(item_.id)) for item_ in next_items]
        abs_items = [[self.item_manager.lookup_by("id", str(item_)) for item_ in items_] for items_ in abs_items]
        return next_items, abs_items


    def remove_low_value_items(self, items):
        return list(filter(lambda a: "Potion" not in a["name"] and "Cull" not in a["name"] and "Doran" not in a[
            "name"] and "Dark Seal" not in a[
            "name"] and "Soul" not in a["name"], items))

    def simulate_game(self, items, champs):
        count = 0
        at_same_number = 0
        last_number = 0
        while count != 10:
            count = 0
            for summ_index in range(10):
                if summ_index == 5:

                    champs, items = self.swap_teams(champs, items)
                count += int(self.next_item_for_champ(summ_index % 5, champs, items))

            if count == last_number:
                at_same_number += 1
                if count > 6 and at_same_number >= 10:
                    break
            else:
                at_same_number = 0
                last_number = count
            champs, items = self.swap_teams(champs, items)
            for i, item in enumerate(items):
                print(f"{divmod(i, 6)}: {item}")
            pass


    def analyze_champ(self, role, champs, items, cs, lvl, kda, current_gold):
        assert (len(champs) == 10)
        assert (len(items) == 60)
        print("\nRole: " + str(role))
        empty_item = self.item_manager.lookup_by("int", 0)
        if role > 4:
            print("Switching teams!")
            champs, items = self.swap_teams(champs, items)
            role -= 5

        result = []

        items_id = np.reshape(items, (10, 6))
        items_id = [list(filter(lambda a: a["id"] != '0', summ_items)) for summ_items in items_id]

        while current_gold > 0:
            try:
                next_item = self.predict_next_item(role, champs, items_id, cs, lvl, kda, current_gold)
            except ValueError as e:
                abs_items[-1] = self.remove_low_value_items(abs_items[-1])


            next_items, abs_items = self.build_path(items, next_item, role)
            result.extend(next_items)
            for next_item in next_items:
                cass_next_item = cass.Item(id=(int(next_item["main_img"]) if "main_img" in
                                                next_item else int(next_item["id"])), region="KR")
                current_gold -= cass_next_item.gold.base

        return result


    def next_item_for_champ(self, role, champs, items):
        assert (len(champs) == 10)
        assert (len(items) == 60)

        empty_item = self.item_manager.lookup_by("int", 0)

        summ_next_item_cass = None
        result = []

        containsCompletedItem = False
        req_item_tier = 0
        items_ahead = 0


        while items_ahead < 20:
            items_ahead += 1
            completes = sum([(1 if "completion" in item and item["completion"]=="complete" else 0) for item in
                             items[self.summoner_items_slice(role)]])
            if completes >= 6:
                return True

            next_item = self.predict_next_item(role, champs, items)
            next_items, abs_items = self.build_path(items, next_item, role)

            abs_items[-1] = list(filter(lambda a: a["id"] != '0', abs_items[-1]))

            try:
                if len(abs_items[-1]) >= game_constants.MAX_ITEMS_PER_CHAMP:
                    print("too many items:")
                    print(abs_items[-1])
                    abs_items[-1] = self.remove_low_value_items(abs_items[-1])
                    print("removing low value items:")
                    print(abs_items[-1])
                items[self.summoner_items_slice(role)] = np.pad(
                    abs_items[-1], (0, game_constants.MAX_ITEMS_PER_CHAMP - len(abs_items[-1])),
                    'constant',
                    constant_values=(
                        empty_item, empty_item))
                return False
            except ValueError as e:
                print("Max items reached!!")
                print(repr(e))

                if req_item_tier == 2:
                    return True
                else:
                    req_item_tier += 1
        return True


    def on_created(self, event):
        # pr.enable()
        
        # prevent keyboard mashing
        if self.onTimeout:
            return
        file_path = event.src_path
        print("Got event for file %s" % file_path)
        # stupid busy waiting until file finishes writing
        oldsize = -1
        while True:
            size = os.path.getsize(file_path)
            if size == oldsize:
                break
            else:
                oldsize = size
                time.sleep(0.05)

        self.process_image(event.src_path)
        
        # pr.disable()
        # s = io.StringIO()
        # sortby = 'cumulative'
        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps.print_stats()
        # print(s.getvalue())

        self.timeout()

    def run_test_games(self):
        with open('test_data/items_test/setups.json', "r") as f:
            games = json.load(f)
        for key in games:
            champs = games[key]["champs"]
            items = games[key]["items"]

            champs = [ChampManager().lookup_by('name', champ) for champ in champs]
            items = [ItemManager().lookup_by('name', item) for item in items]
            items = np.delete(items, np.arange(6, len(items), 7))
            print(f"----------- SIMULATING {key}--------------------------")
            print(champs)
            print(items)
            self.simulate_game(items, champs)



    def timeout(self):
        self.onTimeout = True
        time.sleep(5.0)
        self.onTimeout = False


    def process_image(self, img_path):

        print('you pressed tab + f12 ' + img_path)

        try:
            print("Now trying to predict image")
            screenshot = cv.imread(img_path)
            # utils.show_coords(screenshot, self.champ_img_model.coords, self.champ_img_model.img_size)
            print("Trying to predict champ imgs")
            
            champs = list(self.champ_img_model.predict(screenshot))
            print(f"Champs: {champs}\n")

            try:
                kda = list(self.kda_img_model.predict(screenshot))
            except Exception as e:
                kda = [[0,0,0]]*10
            print(f"KDA:\n {kda}\n")
            tesseract_result = self.tesseract_models.predict(screenshot)
            try:
                lvl = next(tesseract_result)
            except Exception as e:
                lvl = [0]*10
            try:
                cs = next(tesseract_result)
            except Exception as e:
                cs = [0]*10
            try:
                current_gold = next(tesseract_result)[0]
            except Exception as e:
                current_gold = 500

            print(f"Lvl:\n {lvl}\n")
            print(f"CS:\n {cs}\n")
            print(f"Current Gold:\n {current_gold}\n")
            print("Trying to predict item imgs. \nHere are the raw items: ")
            items = list(self.item_img_model.predict(screenshot))
            for i, item in enumerate(items):
                print(f"{divmod(i, 7)}: {item}")
            items = [self.item_manager.lookup_by("int", item["int"])  for item in items]
            print("Here are the converted items:")
            for i, item in enumerate(items):
                print(f"{divmod(i, 7)}: {item}")
            print("Trying to predict self imgs")
            self_index = self.self_img_model.predict(screenshot)
            print(self_index)

        except FileNotFoundError as e:
            print(e)
            return
        except Exception as e:
            print(e)
            traceback.print_exc()
            return

        #remove items that the network is not trained on, such as control wards
        items = [item if (item["name"] != "Warding Totem (Trinket)" and item[
            "name"] != "Farsight Alteration" and item["name"] != "Oracle Lens") else self.item_manager.lookup_by("int", 0) for item in
                 items]

        #we don't care about the trinkets
        items = np.delete(items, np.arange(6, len(items), 7))
        #
        # items = [self.item_manager.lookup_by('int', 0)] * 60
        # items[30:] = [self.item_manager.lookup_by('int', 0)]*30
        # champs[0] = ChampManager().lookup_by('name', 'Aatrox')
        # champs[2] = ChampManager().lookup_by('name', 'Vladimir')
        # champs[4] = ChampManager().lookup_by('name', 'Soraka')


        # x = np.load(sorted(glob.glob(app_constants.train_paths[
        #                                  "next_items_early_processed"] + 'train_x*.npz'))[0])['arr_0']
        #
        # y = np.load(sorted(glob.glob(app_constants.train_paths[
        #                                  "next_items_early_processed"] + 'train_y*.npz'))[0])['arr_0']
        #
        # items = [ItemManager().lookup_by("int", item) for item in x[25][10:]]
        # champs = [ChampManager().lookup_by("int", champ) for champ in x[0][:10]]
        # self.simulate_game(items, champs)

        # for summ_index in range(10):
        #     champs_copy = copy.deepcopy(champs)
        #     items_copy = copy.deepcopy(items)
        #     items_to_buy = self.analyze_champ(summ_index, champs_copy, items_copy)
        #     print(f"This is the result for summ_index {summ_index}: ")
        #     print(items_to_buy)



        items_to_buy = self.analyze_champ(self_index, champs, items, cs, lvl, kda, current_gold)
        print(f"This is the result for summ_index {self_index}: ")
        print(items_to_buy)
        out_string = ""
        if items_to_buy[0]:
            out_string += str(items_to_buy[0]["id"])
        for item in items_to_buy[1:]:
            out_string += "," + str(item["id"])
        with open(os.path.join(os.getenv('LOCALAPPDATA'), "League IQ", "last"), "w") as f:
            f.write(out_string)

    @staticmethod
    def shouldTerminate():
        return os.path.isfile(os.path.join(os.getenv('LOCALAPPDATA'), "League IQ", "terminate"))

    def run(self):
        
        observer = Observer()
        ss_path = os.path.join(self.loldir, "Screenshots")
        print(f"Now listening for screenshots at: {ss_path}")
        observer.schedule(self, path=ss_path)
        observer.start()
        try:
            with open(os.path.join(os.getenv('LOCALAPPDATA'), "League IQ", "ai_loaded"), 'w') as f:
                f.write("true")
            while not Main.shouldTerminate():
                time.sleep(1)
            observer.stop()
            os.remove(os.path.join(os.getenv('LOCALAPPDATA'), "League IQ", "terminate"))
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

m = Main()
m.run()
# m.run_test_games()

# pr = cProfile.Profile()

# dataloader_1 = data_loader.UnsortedNextItemsDataLoader()
# X_un = dataloader_1.get_train_data()
# dataloader = data_loader.SortedNextItemsDataLoader(app_constants.train_paths["next_items_processed_sorted"])
# X, Y = dataloader.get_train_data()
# m = NextItemEarlyGameModel()
# # X = X[Y==2]
# X_ = X[:, 1:]
# X_ = X_[500:700]
# m.output_logs(X_[:200])

#
# blob = cv.imread("blob.png", cv.IMREAD_GRAYSCALE )
# cv.imshow("blob", blob)
# cv.waitKey(0)
# ret, thresholded = cv.threshold(blob, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# cv.imshow("thresholded", thresholded)
# cv.waitKey(0)
#
# from train_model.model import CurrentGoldImgModel, CSImgModel, LvlImgModel, MultiTesseractModel
# with open('test_data/easy/test_labels.json', "r") as f:
#     elems = json.load(f)

# base_path = "test_data/easy/"
# m = Main()


# for key in elems:


#     if elems[key]["hud_scale"] != None:
#         test_image_y = elems[key]

#         m.set_res_converter(ui_constants.ResConverter(*(test_image_y["res"].split(",")), elems[key]["hud_scale"],
#                                                       elems[key]["summ_names_displayed"]))

#         m.process_image(base_path + test_image_y["filename"])

            # KDAImgModel(res_cvt).predict(test_image_x)

# cass.Item(id=2055, region="KR")