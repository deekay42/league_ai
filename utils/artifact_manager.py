import json
from abc import ABC, abstractmethod
from constants import app_constants
import cv2 as cv
import traceback


class ArtifactManager(ABC):

    def __init__(self, data_path, assets_path=None, supported_lookups={"id", "name", "int", "img_int"}):
        self._by = dict()
        self.supported_lookups = supported_lookups

        for lookup in self.supported_lookups:
            self._by[lookup] = dict()
        self.data_path = data_path
        self.base_dict = None
        self.assets_path = assets_path
        self.parse_data()
        self.build_lookup_dicts()

    def lookup_by(self, lookup, val):
        try:
            return self._by[lookup][val]
        except KeyError as e:
            print(f"ERROR: There was an error looking up the given key {lookup} and the value {val}")
            print(repr(e))
            print(traceback.format_exc())


    def parse_data(self):
        with open(self.data_path) as f:
            try:
                self.base_dict = json.load(f)
            except json.decoder.JSONDecodeError as e:
                print("Error parsing base JSON file")
                print(e)
                raise e


    def build_lookup_dicts(self):
        int_counter = 0
        img_int_counter = 0
        for json_artifact in self.base_dict.values():
            self._handle_vanilla_artifacts(json_artifact, int_counter, img_int_counter)
            int_counter += 1
            img_int_counter += 1


    def get_num(self, lookup):
        return len(self._by[lookup])


    def _handle_vanilla_artifacts(self, json_artifact, int_counter, img_int_counter):
        json_artifact["int"] = int_counter
        json_artifact["img_int"] = img_int_counter
        for lookup in self.supported_lookups:
            self._by[lookup][json_artifact[lookup]] = json_artifact


    def _handle_virtual_artifacts(self, json_virtual, img_int_counter):
        main_img_id = json_virtual["main_img"]
        main_artifact = self._by["id"][main_img_id]
        json_virtual["int"] = main_artifact["int"]
        json_virtual["img_int"] = img_int_counter
        for lookup in self.supported_lookups - {"int"}:
            self._by[lookup][json_virtual[lookup]] = json_virtual


    def get_imgs(self):
        return {artifact["img_int"]: cv.imread(self.assets_path + artifact["name"] + ".png") for artifact in
                self._by["img_int"].values()}


    def get_ints(self):
        return self._by["int"]


class ChampManager:
    instance = None


    def __init__(self):
        if not ChampManager.instance:
            ChampManager.instance = ChampManager.__ChampManager()


    def __getattr__(self, name):
        return getattr(self.instance, name)


    class __ChampManager(ArtifactManager):

        def __init__(self):
            super().__init__(data_path=app_constants.asset_paths["champ_json"],
                             assets_path=app_constants.asset_paths["champ_imgs"])


        def build_lookup_dicts(self):
            virtuals = []
            int_counter = 0
            img_int_counter = 0
            for json_artifact in self.base_dict.values():
                if "main_img" in json_artifact:
                    virtuals.append(json_artifact)
                else:
                    self._handle_vanilla_artifacts(json_artifact, int_counter, img_int_counter)
                    int_counter += 1
                    img_int_counter += 1

            for json_virtual in virtuals:
                self._handle_virtual_artifacts(json_virtual, img_int_counter)
                img_int_counter += 1


class ItemManager:
    instance = None


    def __init__(self):
        if not ItemManager.instance:
            ItemManager.instance = ItemManager.__ItemManager()


    def __getattr__(self, name):
        return getattr(self.instance, name)


    class __ItemManager(ArtifactManager):

        def __init__(self):
            super().__init__(data_path=app_constants.asset_paths["item_json"],
                             assets_path=app_constants.asset_paths["item_imgs"])


        def build_lookup_dicts(self):
            virtuals = []
            unbuyables = []
            int_counter = 0
            img_int_counter = 0
            for json_artifact in self.base_dict.values():
                if "main_img" in json_artifact:
                    virtuals.append(json_artifact)
                elif "buyable_id" in json_artifact:
                    unbuyables.append(json_artifact)
                else:
                    self._handle_vanilla_artifacts(json_artifact, int_counter, img_int_counter)
                    int_counter += 1
                    img_int_counter += 1
            for json_virtual in virtuals:
                self._handle_virtual_artifacts(json_virtual, img_int_counter)
                img_int_counter += 1
            for json_unbuyable in unbuyables:
                self._handle_unbuyable_items(json_unbuyable, img_int_counter)
                img_int_counter += 1

            # first element must be the empty item!
            assert (self._by["id"]['0']["int"] == 0)
            assert (self._by["id"]['0']["name"] == "Empty")


        def _handle_unbuyable_items(self, json_unbuyable, img_int_counter):
            buyable_id = json_unbuyable["buyable_id"]
            if buyable_id != "0":
                buyable_item = self._by["id"][buyable_id]
                json_unbuyable["int"] = buyable_item["int"]
            else:
                json_unbuyable["int"] = 0
            json_unbuyable["img_int"] = img_int_counter

            for lookup in self.supported_lookups - {"int"}:
                self._by[lookup][json_unbuyable[lookup]] = json_unbuyable


        def get_imgs(self):
            return {item["img_int"]: cv.imread(self.assets_path + item["id"] + ".png") for item in
                    self._by["img_int"].values()}


        def get_num_completes(self):
            return sum([1 if "completion" in artifact and (artifact["completion"]=="complete" or artifact[
                "completion"]=="semi") else 0 for artifact in self._by["int"].values()])


        def get_completes(self):
            return [artifact for artifact in self._by["int"].values() if "completion" in artifact and (artifact[
                                                                                                           "completion"] == "complete" or artifact[
                "completion"] == "semi")]


class SelfManager:
    instance = None


    def __init__(self):
        if not SelfManager.instance:
            SelfManager.instance = SelfManager.__SelfManager()


    def __getattr__(self, name):
        return getattr(self.instance, name)


    class __SelfManager(ArtifactManager):

        def __init__(self):
            super().__init__(data_path=app_constants.asset_paths["self_json"],
                             assets_path=app_constants.asset_paths["self_imgs"])


class SpellManager:
    instance = None


    def __init__(self):
        if not SpellManager.instance:
            SpellManager.instance = SpellManager.__SpellManager()


    def __getattr__(self, name):
        return getattr(self.instance, name)


    class __SpellManager(ArtifactManager):

        def __init__(self):
            super().__init__(data_path=app_constants.asset_paths["spell_json"])
