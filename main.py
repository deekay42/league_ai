#DONT CHANGE THESE IMPORTS. PYINSTALLER NEEDS THESE
print("starting load")
import time
import threading
import importlib
sstarttime = time.time()


from utils import heavy_imports 

print("starting load 1")
starttime = time.time()
import os
print(f"Took {time.time() - starttime} s")
print("2")
starttime = time.time()
import configparser
print(f"Took {time.time() - starttime} s")
print("3")
starttime = time.time()
import traceback
print(f"Took {time.time() - starttime} s")
print("4")
starttime = time.time()
from tkinter import Tk
print(f"Took {time.time() - starttime} s")
print("5")
starttime = time.time()
from tkinter import messagebox
print(f"Took {time.time() - starttime} s")
print("6")
# starttime = time.time()
# import cv2 as cv
# print(f"Took {time.time() - starttime} s")
print("7")
starttime = time.time()
import numpy as np
print(f"Took {time.time() - starttime} s")
print("8")
starttime = time.time()
import cProfile
print(f"Took {time.time() - starttime} s")
print("9")
starttime = time.time()
import io
print(f"Took {time.time() - starttime} s")
print("10")
starttime = time.time()
import pstats
print(f"Took {time.time() - starttime} s")
print("11")
starttime = time.time()
import copy
print(f"Took {time.time() - starttime} s")
print("12")
starttime = time.time()
import glob
print(f"Took {time.time() - starttime} s")
print("13")
starttime = time.time()
import json
print(f"Took {time.time() - starttime} s")
print("14")
starttime = time.time()
from collections import Counter
starttime = time.time()
print(f"Took {time.time() - starttime} s")
# print("15")
# starttime = time.time()
# # from utils import cass_configured as cass
# from cassiopeia.core.staticdata import Item
# print(f"Took {time.time() - starttime} s")
print("16")
starttime = time.time()
from range_key_dict import RangeKeyDict
print(f"Took {time.time() - starttime} s")
print("17")
starttime = time.time()
from watchdog.events import FileSystemEventHandler
print(f"Took {time.time() - starttime} s")
print("18")
starttime = time.time()
from watchdog.observers import Observer
print(f"Took {time.time() - starttime} s")
print("19")
starttime = time.time()
from constants import ui_constants, game_constants, app_constants
print(f"Took {time.time() - starttime} s")
print("20")
starttime = time.time()
from train_model.model import ChampImgModel, ItemImgModel, SelfImgModel, NextItemModel, CSImgModel, \
    KDAImgModel, CurrentGoldImgModel, LvlImgModel, MultiTesseractModel, CPredict
print(f"Took {time.time() - starttime} s")
print("22")
starttime = time.time()
from utils.artifact_manager import ChampManager, ItemManager
print(f"Took {time.time() - starttime} s")
print("23")
starttime = time.time()
from utils.build_path import build_path_for_gold, InsufficientGold, NoPathFound
print(f"Took {time.time() - starttime} s")
print("24")
starttime = time.time()
from utils.misc import itemslots_left
print(f"Took {time.time() - starttime} s")
print("25")
starttime = time.time()
from utils import misc
print(f"Took {time.time() - starttime} s")
print("26")
starttime = time.time()
import configparser
print(f"Took {time.time() - starttime} s")
print("27")
starttime = time.time()
import logging
print(f"Took {time.time() - starttime} s")
print("28")
starttime = time.time()
import sys
print(f"Took {time.time() - starttime} s")

print(f"OVERALL Took {time.time() - sstarttime} s")
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)
# logger.addHandler(logging.StreamHandler(sys.stdout))



class NoMoreItemSlots(Exception):
    pass

class Main(FileSystemEventHandler):

    def __init__(self):
        logger.info("Starting main")
        self.onTimeout = False
        self.loldir = misc.get_lol_dir()
        self.listener_dir = os.path.join(self.loldir, "Screenshots")
        self.screenshot_dir_created = False
        logger.info("Go tlol dir !")
        self.config = configparser.ConfigParser()
        logger.info("Reading config")
        self.config.read(self.loldir + os.sep +"Config" + os.sep + "game.cfg")

        while True:
            try:
            # res = 1440,810
                res = int(self.config['General']['Width']), int(self.config['General']['Height'])
                break
            except KeyError as e:
                print(repr(e))
                os.remove(os.path.join(os.getenv('LOCALAPPDATA'), "League IQ", "loldir"))
                misc.get_lol_dir()

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

        too_many_screenshots = len(glob.glob(self.loldir+os.sep + "Screenshots" + os.sep + "*")) > 300

        if too_many_screenshots:
            Tk().withdraw()
            messagebox.showinfo("Warning",
                                f"The screenshots folder at {self.loldir}\\Screenshots has over 300 screenshots. League IQ may stop working if the folder grows too large. Make sure to delete old screenshots.")




        # self.res_converter = ui_constants.ResConverter(1920, 1200, 0.48)
        # self.res_converter = ui_constants.ResConverter(1440, 900, 0.48)
        logger.info("Loading res cvt")
        self.res_converter = ui_constants.ResConverter(*res, hud_scale=hud_scale, summ_names_displayed=show_names_in_sb)

        self.item_manager = ItemManager()
        if Main.shouldTerminate():
            return
        with open(app_constants.asset_paths["champ_vs_roles"], "r") as f:
            self.champ_vs_roles = json.load(f)

        logger.info("Now loading models!")
        dll_hook = CPredict()
        # dll_hook = None

        self.next_item_model_standard = NextItemModel("standard", dll_hook=dll_hook)
        self.next_item_model_standard.load_model()
        if Main.shouldTerminate():
            return
        self.next_item_model_late = NextItemModel("late", dll_hook=dll_hook)
        self.next_item_model_late.load_model()
        if Main.shouldTerminate():
            return
        self.next_item_model_starter = NextItemModel("starter", dll_hook=dll_hook)
        self.next_item_model_starter.load_model()
        if Main.shouldTerminate():
            return
        self.next_item_model_first_item = NextItemModel("first_item", dll_hook=dll_hook)
        self.next_item_model_first_item.load_model()
        if Main.shouldTerminate():
            return
        self.next_item_model_boots = NextItemModel("boots", dll_hook=dll_hook)
        self.next_item_model_boots.load_model()
        if Main.shouldTerminate():
            return
        self.champ_img_model = ChampImgModel(self.res_converter, dll_hook=dll_hook)
        self.champ_img_model.load_model()
        if Main.shouldTerminate():
            return
        self.item_img_model = ItemImgModel(self.res_converter, dll_hook=dll_hook)
        self.item_img_model.load_model()
        if Main.shouldTerminate():
            return
        self.self_img_model = SelfImgModel(self.res_converter, dll_hook=dll_hook)
        self.self_img_model.load_model()
        if Main.shouldTerminate():
            return
        self.kda_img_model = KDAImgModel(self.res_converter, dll_hook=dll_hook)
        self.kda_img_model.load_model()
        if Main.shouldTerminate():
            return
        self.tesseract_models = MultiTesseractModel([LvlImgModel(self.res_converter),
                                                     CSImgModel(self.res_converter),
                                                     CurrentGoldImgModel(self.res_converter)])
        logger.info("All models loaded!")
        self.previous_champs = None
        self.previous_kda = None
        self.previous_cs = None
        self.previous_lvl = None
        self.previous_role = None

        self.boots_ints = ItemManager().get_boots_ints()
        self.ward_int = self.item_manager.lookup_by("name", "Control Ward")["int"]
        self.removable_items = ["Control Ward", "Health Potion", "Refillable Potion", "Corrupting Potion",
                                "Cull", "Doran's Blade", "Doran's Shield", "Doran's Ring",
                                "Rejuvenation Bead", "The Dark Seal", "Faerie Charm",
                                "Elixir of Wrath", "Elixir of "
                                                   "Iron",
                                "Elixir of Sorcery"]
        
        num_full_items = [0, 1, 2, 3, 4, 5]
        thresholds = [0, 0.1, 0.25, .7, 1.1]
        max_leftover_gold_threshold = [349, 499, 1999, 1999, 1999, 1999]
        self.threshold = 0.3
        self.gold_tolerance = 50
        self.max_leftover_gold_threshold = 500
        self.skipped = False
        self.skipped_item = None
        self.force_late_after_standard = False
        self.force_boots_network_after_first_item = False
        self.max_leftover_gold_thresholds_dict = dict(zip(num_full_items, max_leftover_gold_threshold))
        
        self.commonality_to_items = dict()
        for i in range(len(thresholds)-1):
            self.commonality_to_items[(thresholds[i], thresholds[i + 1])] = num_full_items[i]
        self.commonality_to_items = RangeKeyDict(self.commonality_to_items)
        logger.info("init complete!")
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
            lol = heavy_imports.Item(id=3040, region="EUW").name[-1]
        except Exception as e:
            print(e)
            print(f"Connection error. Retry in {timeout}")
            time.sleep(timeout)
            Main.test_connection(5)


    @staticmethod
    def swap_teams(team_data):
        return np.concatenate([team_data[5:], team_data[:5]], axis=0)


    def summoner_items_slice(self, role):
        return np.s_[
               role * game_constants.MAX_ITEMS_PER_CHAMP:role * game_constants.MAX_ITEMS_PER_CHAMP + game_constants.MAX_ITEMS_PER_CHAMP]


    def get_blackout_items(self, summ_items_counter):
                completes = self.item_manager.get_completes()
                for item in summ_items_counter:
                    item_full = self.item_manager.lookup_by("int", item)
                    if item in completes and not ("multiples_allowed" in item_full and item_full["multiples_allowed"]):
                        subcomponents = heavy_imports.Item(id=(int(item_full["id"])), region="EUW").builds_from
                        # print(subcomponents)
                        for subcomponent in subcomponents:
                            # print(subcomponent)
                            sub = self.item_manager.lookup_by("id", str(subcomponent.id))["int"]
                            if sub in completes:
                                yield sub
                        yield item


    def all_items_counter2items_list(self, counter, lookup):
        items_id = [[], [], [], [], [], [], [], [], [], []]
        for i in range(10):
            items_id[i] = self.items_counter2items_list(counter[i], lookup)
        return items_id


    def items_counter2items_list(self, summ_items, lookup):
        result = []
        for item_key in summ_items:
            id_item = self.item_manager.lookup_by(lookup, item_key)
            result.extend([id_item] * summ_items[item_key])
        return result


    def predict_next_item(self, model=None, role=None, champs=None, items=None, cs=None, lvl=None, kda=None,
                          delta_items=Counter()):
        
                
        if role is None:
            role = self.role
        if champs is None:
            champs = self.champs
        if items is None:
            items = self.items
        if cs is None:
            cs = self.cs
        if lvl is None:
            lvl = self.lvl
        if kda is None:
            kda = self.kda
        if model is None:
            model = self.next_item_model
        elif model == "standard":
            model = self.next_item_model_standard
        elif model == "late":
            if np.any(np.isin(list(items[role].keys()), list(ItemManager().get_full_item_ints()))):
                model = self.next_item_model_late
                print("using LATE network")
            else:
                print("using FIRST ITEM network")
                model = self.next_item_model_first_item
        champs_int = [int(champ["int"]) for champ in champs]
        items_id = self.all_items_counter2items_list(items, "int")
        items_id = [[int(item["id"]) for item in summ_items] for summ_items in items_id]
        summ_owned_completes = None
        if self.network_type == "late" or self.network_type == "first_item":
            # at beginning of game dont buy potion first item. buy starter item first
            if items[role]:
                summ_owned_completes = list(self.get_blackout_items(items[role] + delta_items))

        return model.predict_easy(role, champs_int, items_id, cs, lvl, kda, self.current_gold,
                                  summ_owned_completes)


    def build_path(self, next_item, current_gold=None, items=None):
        if items is None:
            items = self.items[self.role]

        if current_gold is None:
            current_gold = self.current_gold

        if next_item["name"] == "Empty":
            return [], [], []

        # if self.network_type == "first_item":
        #     return [next_item], [items], 0, False
        cass_item = heavy_imports.Item(id=(int(next_item["id"])), region="EUW")
        l = cass_item.name
        if not list(cass_item.builds_from):
            if current_gold >= cass_item.gold.base:
                return [next_item], [items + Counter({next_item["int"]: 1})], [cass_item.gold.base]
            else:
                # raise InsufficientGold(next_item)
                return [], [], []
        items_by_id = Counter({int(self.item_manager.lookup_by("int", item_id)["id"]): qty for item_id,
                                                                                               qty in items.items()})

        # TODO: this is bad. the item class should know when to return main_img or id
        next_items, abs_items = build_path_for_gold(heavy_imports.Item(id=(int(next_item["main_img"]) if
                                                                  "main_img" in
                                                                  next_item else
                                                                  int(next_item["id"])), region="EUW"), items_by_id,
                                                    current_gold)
        next_items = [self.item_manager.lookup_by("id", str(item_.id)) for item_ in next_items]
        abs_items = [Counter([self.item_manager.lookup_by("id", str(item))["int"] for item, qty in
                              abs_items_counter.items() for _ in range(qty)]) for abs_items_counter in abs_items]
        cost = [heavy_imports.Item(id=(int(item["main_img"]) if "main_img" in
                                                         item else int(item["id"])),
                              region="EUW").gold.base for item in next_items]

        return next_items, abs_items, cost


    def remove_low_value_items(self, items, blacklist):
        items = Counter(items)
        removal_index = 0
        delta_items = Counter()
        six_items = None
        delta_six = None
        while itemslots_left(items) <= 0:
            if itemslots_left(items) == 0:
                six_items = Counter(items)
                delta_six = Counter(delta_items)
            if removal_index >= len(self.removable_items):
                break
            item_to_remove = self.item_manager.lookup_by("name", self.removable_items[removal_index])['int']
            if item_to_remove in items and item_to_remove not in blacklist:
                delta_items += Counter({item_to_remove: items[item_to_remove]})
            items -= delta_items
            removal_index += 1

        return items, delta_items, six_items, delta_six


    def swap_teams_all(self):
        logger.info("Switching teams!")
        self.champs = self.swap_teams(self.champs)
        self.items = self.swap_teams(self.items)
        self.lvl = self.swap_teams(self.lvl)
        self.cs = self.swap_teams(self.cs)
        self.kda = self.swap_teams(self.kda)
        self.role -= 5


    def predict_with_threshold(self, delta_items=Counter(), model=None):
        next_items, confidences = self.predict_next_item(delta_items=delta_items, model=model)
        tmp = sorted(np.transpose([next_items, confidences]), key=lambda a: a[1])
        tmp = [[next_item, confidence] for (next_item, confidence) in tmp if next_item["name"]!="Empty"]
        if tmp == []:
            return [], [], []
        next_items, confidences = np.transpose(tmp)
        for item, confidence in zip(next_items[::-1], confidences[::-1]):
            items_buy_seq, abs_items, cost = self.build_path(item, current_gold=self.current_gold + self.gold_tolerance)
            if items_buy_seq == []:
                continue
            # if cost[0] <= self.current_gold + self.gold_tolerance:
                # return item, next_items, abs_items, item_cost

            if confidence < self.threshold and self.network_type == "standard":
                logger.info("Super low confidence in that one. Falling back to late game network.")
                self.network_type = "late"
                self.model = self.next_item_model_late
                item = self.predict_next_item(model="late", delta_items=delta_items)[0][0]
                items_buy_seq, abs_items, cost = self.build_path(item)

            return items_buy_seq, abs_items, cost
        return [], [], []


    # def select_affordable_item(self, items, tolerance):
    #     item_cost_lookup = dict()
    #     for item in items[::-1]:
    #         if item["int"] in item_cost_lookup:
    #             item_cost = item_cost_lookup[item["int"]]
    #         else:
    #             next_items, abs_items, item_cost = self.build_path(item, current_gold=10000)
    #             item_cost_lookup[item["int"]] = item_cost
    #         if item_cost <= current_gold + tolerance:
    #             return item, next_items, abs_items, item_cost
    #     raise InsufficientGold()


    def try_item_reduction(self, blacklist):
        items_five, delta_five, items_six, delta_six = self.remove_low_value_items(self.items[self.role], blacklist)
        next_items, abs_items, cost = [], [self.items[self.role]], 0
        next_item = None
        delta_items = None

        for items_reduction, deltas in zip([items_six, items_five], [delta_six, delta_five]):
            if items_reduction is None:
                continue
            # copied_items = [Counter(summ_items) for summ_items in self.items]
            # copied_items[self.role] = items_reduction
            self.items[self.role] = items_reduction
            delta_items = deltas

            next_items, abs_items, cost = self.predict_with_threshold(delta_items=delta_items)
            if next_items == [] or next_items[0]["name"] == "Empty":
                continue
            next_item = next_items[0]

            if itemslots_left(abs_items[-1]) >= 0:
                break
        return next_item, delta_items, next_items, abs_items, cost


    def true_completes_owned(self):
        return len(list(self.item_manager.extract_full_items(self.items[self.role])))


    def select_right_network(self):
        num_true_completes_owned = self.true_completes_owned()
        champ_vs_role_commonality = self.champ_vs_roles[str(self.champs[self.role]["int"])].get(
            game_constants.ROLE_ORDER[self.role], 0)
        logger.info(f"champ vs roles commonality: {champ_vs_role_commonality}")
        allowed_items = self.commonality_to_items[champ_vs_role_commonality]
        self.max_leftover_gold_threshold = self.max_leftover_gold_thresholds_dict[num_true_completes_owned]
        if self.role == 4:
            allowed_items -= 1
        if num_true_completes_owned < allowed_items and not self.force_late_after_standard:
            self.network_type = "standard"
            self.next_item_model = self.next_item_model_standard
            logger.info("USING STANDARD GAME MODEL")
        else:
            if self.items[self.role] == Counter():
                self.network_type = "starter"
                self.next_item_model = self.next_item_model_starter
                logger.info("USING STARTER GAME MODEL")
            elif np.any(np.isin(list(self.items[self.role].keys()), list(ItemManager().get_full_item_ints()))):
                self.network_type = "late"
                self.next_item_model = self.next_item_model_late
                logger.info("USING LATE GAME MODEL")
            elif self.force_boots_network_after_first_item:
                self.network_type = "boots"
                self.next_item_model = self.next_item_model_boots
                logger.info("USING BOOTS GAME MODEL")
            else:
                self.network_type = "first_item"
                self.next_item_model = self.next_item_model_first_item
                logger.info("USING FIRST ITEM GAME MODEL")


    def analyze_champ(self):
        if self.role > 4:
            self.swap_teams_all()
        result = []
        while self.current_gold > 0:
            self.select_right_network()
            if itemslots_left(self.items[self.role]) <= 0:
                next_item, delta_items, next_items, abs_items, cost = self.try_item_reduction(Counter(
                    [i["int"] for i in result]))
            else:
                delta_items = None
                next_items, abs_items, cost = self.predict_with_threshold()
                if next_items == []:
                    if self.current_gold >= self.max_leftover_gold_threshold and self.true_completes_owned() < 3 and \
                            not self.skipped:
                        self.skip_item()
                        if delta_items:
                            self.add_deltas_back(delta_items)
                        continue
                    return self.pad_result(result)

                next_item = next_items[0]
            if self.is_end_of_buy(next_item, delta_items, next_items):
                if self.network_type != "starter" and self.current_gold >= self.max_leftover_gold_threshold and \
                        self.true_completes_owned() < 3 and not self.skipped:
                    self.skip_item()
                    if delta_items:
                        self.add_deltas_back(delta_items)
                    continue
                else:
                    return self.pad_result(result, next_items, abs_items)

            self.current_gold -= sum(cost)
            self.items[self.role] = abs_items[-1].copy()
            result.extend(next_items)
            current_summ_items = [self.item_manager.lookup_by("int", item) for item in self.items[self.role]]
            if delta_items:
                self.add_deltas_back(delta_items)
        return self.pad_result(result)


    def skip_item(self):
        if self.network_type == "standard":
            self.force_late_after_standard = True
            logger.info("skip from standard to late")
        elif self.network_type == "first_item":
            self.force_boots_network_after_first_item = True
            self.skipped = True
            logger.info("skip first_item  to boots")
        else:
            self.skipped = True
            next_item = self.predict_next_item(model="late")[0][0]
            abs_items = self.build_path(next_item, 4000)[1]
            self.items[self.role] = abs_items[-1].copy()
            self.skipped_item = next_item
            logger.info("skip from late to next late")


    def pad_result(self, result, next_items=None, abs_items=None):
        if not result and hasattr(self, "network_type") and self.network_type not in ["starter", "first_item"]:
            self.network_type = "late"
            next_item = self.predict_next_item(model="late")[0][0]

            next_items, abs_items, _ = self.build_path(next_item, self.current_gold + 30)
            if next_items:
                return next_items
            else:
                if self.skipped_item:
                    print(f"skipped item returned is {self.skipped_item}")
                    return [self.skipped_item]
                else:
                    return [next_item]
        else:
            return self.add_aux_items(result, next_items, abs_items)


    def is_end_of_buy(self, next_item, delta_items, next_items):
        return (next_items == [] or next_item == [] \
            or self.network_type == "standard" and next_item["name"] == "Empty") \
            or (self.items[self.role].get(self.ward_int, 0) >= 1 and next_item["int"] == self.ward_int) \
            or (self.contains_elixir(self.items[self.role]) and self.contains_elixir(Counter({next_item["int"]: 1}))) \
            or delta_items and ((next_item["int"] in delta_items and (next_item['int'] != self.ward_int))
                                or (next_item["name"] in self.removable_items)) \
            or (self.network_type in ["starter"] and self.current_gold == 500)


    def add_aux_items(self, result, next_items, abs_items):
        if self.network_type == "starter":
            print("abs starter aux")
            print(abs_items)
            print(self.items)
            if abs_items != []:
                self.items[self.role] = abs_items[-1]
            print(self.items)
            result.extend(next_items)
            self.current_gold -= heavy_imports.Item(id=(int(next_items[0]["id"])), region="EUW").gold.base


        if not np.any(["Elixir" in item["name"] for item in result]) and self.current_gold >= 500:
            elixir = self.predict_next_item(model='late')[0][0]
            if "Elixir" in elixir["name"]:
                result.append(elixir)
                self.current_gold -= heavy_imports.Item(id=(int(elixir["id"])), region="EUW").gold.base

        while self.network_type != "standard" and self.current_gold > 0 and itemslots_left(self.items[self.role]) > 0:
            print("abs starter aux111")
            next_extra_item , _, _ = self.predict_with_threshold(model="standard")
            print(next_extra_item)
            if next_extra_item != [] and (next_extra_item[0]["name"] in {"Control Ward", "Health Potion"} or \
                (next_extra_item[0]["name"] in {"Refillable Potion"} and self.network_type == "starter")):
                self.buy_one_off_item(next_extra_item[0], result)
            else:
                break
        if result != []:
            return result
        else:
            return [self.predict_next_item(model="late")[0][0]]


    def buy_one_off_item(self, item, result):
        print("one off")
        result.append(item)
        self.current_gold -= heavy_imports.Item(id=(int(item["id"])), region="EUW").gold.base
        self.items[self.role] += Counter({item["int"]: 1})


    def add_deltas_back(self, delta_items):
        for delta_item in delta_items:
            if delta_item in self.items[self.role] and delta_item != self.ward_int:
                self.items[self.role][delta_item] = self.items[self.role][delta_item] - delta_items[delta_item]
            else:
                self.items[self.role] += Counter({delta_item: delta_items[delta_item]})


    def contains_elixir(self, items):
        elixir_ints = [self.item_manager.lookup_by("name", "Elixir of Wrath")["int"], self.item_manager.lookup_by(
            "name", "Elixir of Iron")["int"], self.item_manager.lookup_by("name", "Elixir of Sorcery")["int"]]
        return np.any(np.isin(list(items.keys()), elixir_ints))


    def deflate_items(self, items):
        comp_pool = Counter()
        for item in items:
            item = heavy_imports.Item(id=(int(item["id"])), region="EUW")
            comps = Counter([str(item_comp.id) for item_comp in list(item.builds_from)])
            if comps:
                comp_pool -= Counter(comps)
            comp_pool += Counter({str(item.id): 1})
        result = self.items_counter2items_list(comp_pool, "id")
        result_sorted = sorted(result, key=lambda a: heavy_imports.Item(id=(int(a["id"])), region="EUW").gold.total,
                               reverse=True)
        return result_sorted


    def recipe_cost(self, next_items):
        return sum([heavy_imports.Item(id=(int(next_item["main_img"]) if "main_img" in
                                                                next_item else int(next_item["id"])),
                              region="EUW").gold.base for next_item in next_items])


    def on_created(self, event):
        logger.info("Got event for file %s" % event.src_path)
        # pr.enable()

        # prevent keyboard mashing
        if self.onTimeout:
            return
        file_path = event.src_path
        
        if self.listener_dir == self.loldir:        
            if file_path[-11:] == "Screenshots":
                self.screenshot_dir_created = True
                return
            else:
                return
        


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


    def repair_failed_predictions(self, predictions, lower, upper):
        assert (len(predictions) == 10)
        for i in range(len(predictions)):
            # this pred is wrong
            if predictions[i] is None or upper < predictions[i] or predictions[i] < lower:
                opp_index = (i + 5) % 10
                opp_pred_valid = predictions[opp_index] > lower and predictions[opp_index] < upper
                if opp_pred_valid:
                    predictions[i] = predictions[opp_index]
                else:
                    predictions[i] = sum(predictions) / 10
        return predictions


    def process_image(self, img_path):

        logger.info('you pressed tab + f12 ' + img_path)

        try:
            logger.info("Now trying to predict image")
            screenshot = heavy_imports.cv.imread(img_path)
            # utils.show_coords(screenshot, self.champ_img_model.coords, self.champ_img_model.img_size)
            logger.info("Trying to predict champ imgs")

            self.champs = list(self.champ_img_model.predict(screenshot))
            logger.info(f"Champs: {self.champs}\n")

            try:
                self.kda = list(self.kda_img_model.predict(screenshot))
            except Exception as e:
                self.kda = [[0, 0, 0]] * 10
            logger.info(f"KDA:\n {self.kda}\n")
            self.kda = np.array(self.kda)
            self.kda[:, 0] = self.repair_failed_predictions(self.kda[:, 0], 0, 25)
            self.kda[:, 1] = self.repair_failed_predictions(self.kda[:, 1], 0, 25)
            self.kda[:, 2] = self.repair_failed_predictions(self.kda[:, 2], 0, 25)
            tesseract_result = list(self.tesseract_models.predict(screenshot))
            try:
                self.lvl = tesseract_result[:10]
            except Exception as e:
                logger.info(e)
                self.lvl = [0] * 10
            self.lvl = self.repair_failed_predictions(self.lvl, 1, 18)

            try:
                self.cs = tesseract_result[10:20]
            except Exception as e:
                logger.info(e)
                self.cs = [0] * 10
            self.cs = self.repair_failed_predictions(self.cs, 0, 400)
            try:
                self.current_gold = tesseract_result[-1]
            except Exception as e:
                logger.info(e)
                self.current_gold = 500

            if self.current_gold > 4000:
                self.current_gold = 4000
            elif self.current_gold < 0 or self.current_gold is None:
                self.current_gold = 500

            logger.info(f"Lvl:\n {self.lvl}\n")
            logger.info(f"CS:\n {self.cs}\n")
            logger.info(f"Current Gold:\n {self.current_gold}\n")
            logger.info("Trying to predict item imgs. \nHere are the raw items: ")
            self.items = list(self.item_img_model.predict(screenshot))
            self.items = [self.item_manager.lookup_by("int", item["int"]) for item in self.items]
            logger.info("Here are the converted items:")
            for i, item in enumerate(self.items):
                logger.info(f"{divmod(i, 7)}: {item}")
            logger.info("Trying to predict self imgs")
            self.role = self.self_img_model.predict(screenshot)
            logger.info(self.role)


            def prev_champs2champs(prev_champs):
                repaired_champs_int = [max(pos_counter, key=lambda k: pos_counter[k]) for pos_counter in
                                       prev_champs]
                return [ChampManager().lookup_by("int", champ_int) for champ_int in repaired_champs_int]


            champs_int = [champ["int"] for champ in self.champs]

            # sometimes we get incorrect champ img predictions. we need to detect this and correct for it by taking
            # the previous prediction
            if not self.previous_champs:
                self.previous_champs = [Counter({champ_int: 1}) for champ_int in champs_int]
                self.previous_kda = self.kda
                self.previous_cs = self.cs
                self.previous_lvl = self.lvl
                self.previous_role = self.role

            else:
                champ_overlap = np.sum(np.equal(self.champs, prev_champs2champs(self.previous_champs)))
                # only look at top 3 kdas since the lower ones often are overlapped
                k_increase = np.all(np.greater_equal(self.kda[:3, 0], self.previous_kda[:3, 0]))
                d_increase = np.all(np.greater_equal(self.kda[:3, 1], self.previous_kda[:3, 1]))
                a_increase = np.all(np.greater_equal(self.kda[:3, 2], self.previous_kda[:3, 2]))
                # cs_increase = np.all(np.greater_equal(cs, self.previous_cs))
                # lvl_increase = np.all(np.greater_equal(cs, self.previous_cs))
                # all_increased = k_increase and d_increase and a_increase and cs_increase and lvl_increase

                # this is still the same game
                if champ_overlap > 7 and k_increase and d_increase and a_increase:
                    print("SAME GAME. taking previous champs")
                    self.champs = prev_champs2champs(self.previous_champs)
                    self.previous_champs = [prev_champs_counter + Counter({champ_int: 1}) for
                                            champ_int, prev_champs_counter
                                            in zip(champs_int, self.previous_champs)]
                # this is a new game
                else:
                    self.previous_champs = [Counter({champ_int: 1}) for champ_int in champs_int]
                self.previous_kda = self.kda
                self.previous_cs = self.cs
                self.previous_lvl = self.lvl
                self.previous_role = self.role


        except FileNotFoundError as e:
            print(e)
            return
        except Exception as e:
            print(e)
            traceback.print_exc()
            return

        # remove items that the network is not trained on, such as control wards
        self.items = [item if (item["name"] != "Warding Totem (Trinket)" and item[
            "name"] != "Farsight Alteration" and item["name"] != "Oracle Lens") else self.item_manager.lookup_by("int",
                                                                                                                 0) for
                 item in
                 self.items]

        # we don't care about the trinkets
        self.items = np.delete(self.items, np.arange(6, len(self.items), 7))

        self.items = np.array([summ_items["int"] for summ_items in self.items])
        self.items = np.reshape(self.items, (game_constants.CHAMPS_PER_GAME, game_constants.MAX_ITEMS_PER_CHAMP))
        self.items = [Counter(summ_items) for summ_items in self.items]
        for summ_items in self.items:
            del summ_items[0]

        # if np.any(self.cs != 0):
        #     self.current_gold += 30


        try:
            items_to_buy = self.analyze_champ()
            items_to_buy = self.deflate_items(items_to_buy)
            logger.info(f"This is the result for summ_index {self.role}: ")
            logger.info(items_to_buy)
            out_string = ""
            if items_to_buy and items_to_buy[0]:
                out_string += str(items_to_buy[0]["id"])
            for item in items_to_buy[1:]:
                out_string += "," + str(item["id"])
        except Exception as e:
            print("Unable to predict next item")
            print(e)
            print(traceback.print_exc())
            out_string = "0"
        # with open(os.path.join(os.getenv('LOCALAPPDATA'), "League IQ", "last"), "w") as f:
        #     f.write(out_string)
        
        self.skipped = False
        self.skipped_item = None
        self.force_late_after_standard = False
        self.force_boots_network_after_first_item = False


    @staticmethod
    def shouldTerminate():
        return os.path.isfile(os.path.join(os.getenv('LOCALAPPDATA'), "League IQ", "terminate"))
    

    def run(self):
        observer = Observer()
        logger.info(f"Now listening for screenshots at: {self.listener_dir}")
        observer.schedule(self, path=self.listener_dir)
        try:
            observer.start()
        except FileNotFoundError:
            self.listener_dir = self.loldir
            logger.info(f"Error. Screenshots dir does not exist. Now listening for screenshots at: {self.listener_dir}")
            observer.unschedule_all()
            observer.stop()   
            observer = Observer()
            observer.schedule(self, path=self.listener_dir)
            observer.start()
            
        try:
            with open(os.path.join(os.getenv('LOCALAPPDATA'), "League IQ", "ai_loaded"), 'w') as f:
                f.write("true")
            while not Main.shouldTerminate():
                time.sleep(1)
                if self.screenshot_dir_created:
                    print("screenshot dir created!")
                    observer.unschedule_all()
                    observer.stop()
                    observer = Observer()
                    self.listener_dir = os.path.join(self.loldir, "Screenshots")
                    screenshot_path = glob.glob(self.listener_dir+"/*")[-1]
                    self.process_image(screenshot_path)
                    observer.schedule(self, path=self.listener_dir)
                    observer.start()
                    self.screenshot_dir_created = False
            observer.stop()
            os.remove(os.path.join(os.getenv('LOCALAPPDATA'), "League IQ", "terminate"))
        except KeyboardInterrupt:
            observer.stop()
        observer.join()


# m = Main()
# m.run()

# m.process_image(f"test_data/screenshots/Screen445.png")
# for i in range(386,396):
#     m.process_image(f"test_data/screenshots/Screen{i}.png")

# m.run_test_games()

# pr = cProfile.Profile()

# dataloader = data_loader.SortedNextItemsDataLoader(app_constants.train_paths[
#                                                                      "next_items_processed_elite_sorted_uninf"])
# X, Y = dataloader.get_train_data()
# m = NextItemModel("starter")
# # X = X[Y==2]
# # X_ = X[:, 1:]
# # X_ = X_[500:700]
# m.output_logs(X[:20].astype(np.float32))

#
# blob = heavy_imports.cv.imread("blob.png", heavy_imports.cv.IMREAD_GRAYSCALE )
# heavy_imports.cv.imshow("blob", blob)
# heavy_imports.cv.waitKey(0)
# ret, thresholded = heavy_imports.cv.threshold(blob, 0, 255, heavy_imports.cv.THRESH_BINARY + heavy_imports.cv.THRESH_OTSU)
# heavy_imports.cv.imshow("thresholded", thresholded)
# heavy_imports.cv.waitKey(0)
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


# import tflearn
# from train_model.network import ChampEmbeddings
# from tflearn.data_utils import to_categorical
# import tensorflow as tf
#
# model = tflearn.DNN(ChampEmbeddings().build())
# model.load('models/best/next_items/starter/my_model1809')
# data_input = tflearn.input_data(shape=[None, 1+177], name='input')
# image_batch = np.reshape(np.concatenate([[82], np.sum(to_categorical([ 23 , 57 , 52 , 47, 125, 120],
#                                                                   nb_classes=ItemManager().get_num("int")), axis=0)],
#                                          axis=0), (1,-1)).astype(np.float32)
# d = model.evaluate(image_batch, np.reshape([1.], (-1, 1)))

# feed_dict = tflearn.utils.feed_dict_builder(image_batch , None, [data_input], None)
# graph = tf.get_default_graph()
# [op.values() for op in graph.get_operations()]
# res = model.predictor.evaluate(feed_dict=feed_dict, ops=['Reshape_2:0'], batch_size=1)

# feed_dict = feed_dict_builder(X, Y, self.inputs, self.targets)
# ops = [o.metric for o in self.train_ops]
# return self.predictor.evaluate(feed_dict, ops, batch_size)
# print("hi")

# with tf.Graph().as_default():
#     with tf.Session() as sess:
#         tflearn.is_training(False, session=sess)
#         model = tflearn.DNN(ChampEmbeddings().build(), session=sess)
#         sess.run(tf.global_variables_initializer())
#         try:
#             model.load('models/best/next_items/starter/my_model109', create_new_session=False)
#         except Exception as e:
#             print("Unable to open best model files")
#             raise e
#         print("hi")
