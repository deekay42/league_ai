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

import cassiopeia as cass
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from utils import utils
from train_model.model import ChampImgModel, ItemImgModel, SelfImgModel, NextItemsModel
from utils.artifact_manager import ChampManager, ItemManager, SelfManager, SpellManager
from utils.build_path import build_path
from constants import ui_constants, game_constants, app_constants



class Main(FileSystemEventHandler):

    def __init__(self):
        self.onTimeout = False
        self.loldir = utils.get_lol_dir()
        self.config = configparser.ConfigParser()
        self.config.read(self.loldir + os.sep +"Config" + os.sep + "game.cfg")
        try:
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


        if flipped_sb:
            Tk().withdraw()
            messagebox.showinfo("Error",
                                "League IQ does not work if the scoreboard is mirrored. Please untick the \"Mirror Scoreboard\" checkbox in the game settings (Press Esc while in-game)")
            raise Exception("League IQ does not work if the scoreboard is mirrored.")
        self.res_converter = ui_constants.ResConverter(*res)
        print(f"Res is {res}")
        self.item_manager = ItemManager()
        if Main.shouldTerminate():
            return
        self.next_item_model = NextItemsModel()
        if Main.shouldTerminate():
            return
        self.champ_img_model = ChampImgModel(self.res_converter)
        if Main.shouldTerminate():
            return
        self.item_img_model = ItemImgModel(self.res_converter, show_names_in_sb)
        if Main.shouldTerminate():
            return
        self.self_img_model = SelfImgModel(self.res_converter)
        Main.test_connection()
        

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


    def predict_next_item(self, role, champs, items):
        champs_int = [champ["int"] for champ in champs]
        items_int = [item["int"] for item in items]
        next_items_input = np.concatenate([[role], champs_int, items_int], axis=0)
        next_item = self.next_item_model.predict([next_items_input])
        return next_item


    def build_path(self, items, next_item, role):
        items_id = [int(item["id"]) for item in items]
        summ_curr_items = items_id[self.summoner_items_slice(role)]
        next_items, _, abs_items, _ = build_path(summ_curr_items, cass.Item(id=int(next_item["id"]), region="KR"))
        next_items = [self.item_manager.lookup_by("id", str(item_.id)) for item_ in next_items]
        abs_items = [[self.item_manager.lookup_by("id", str(item_)) for item_ in items_] for items_ in abs_items]
        return next_items, abs_items


    def analyze_champ(self, role, champs, items):
        assert(len(champs) == 10)
        assert(len(items) == 60)
        print("\nRole: " + str(role))
        empty_item = self.item_manager.lookup_by("int", 0)
        if role > 4:
            print("Switching teams!")
            champs, items = self.swap_teams(champs, items)
            role -= 5
        summ_next_item_cass = None
        result = []
        items_ahead = 0
        while True:
            items_ahead += 1
            next_item = self.predict_next_item(role, champs, items)
            next_items, abs_items = self.build_path(items, next_item, role)
            
            result.extend(next_items)
            if len(result) > 4:
                result = result[:4]
                break
            abs_items[-1] = list(filter(lambda a: a["id"] != '0', abs_items[-1]))
            try:
                items[self.summoner_items_slice(role)] = np.pad(
                    abs_items[-1], (0, game_constants.MAX_ITEMS_PER_CHAMP - len(abs_items[-1])),
                    'constant',
                    constant_values=(
                        empty_item, empty_item))
            except ValueError as e:
                print("Max items reached!!")
                print(repr(e))
                break

        return result


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
            for champ in champs:
                print(champ)
            print("Trying to predict item imgs")
            items = list(self.item_img_model.predict(screenshot))
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
        #we don't care about the trinkets
        items = np.delete(items, np.arange(6, len(items), 7))
        items_to_buy = self.analyze_champ(self_index, champs, items)
        print("This is the result: ")
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
        observer.schedule(self, path=self.loldir  + os.sep + "Screenshots")
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

# pr = cProfile.Profile()