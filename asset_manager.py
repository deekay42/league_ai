import glob
import json

import cv2 as cv


class AssetManager:
    instance = None

    def __init__(self):
        if not AssetManager.instance:
            AssetManager.instance = AssetManager.__AssetManager()

    def __getattr__(self, name):
        return getattr(self.instance, name)

    class __AssetManager:
        def __init__(self):
            self.champ_dict_by_str = dict()
            self.champ_dict_by_id = dict()
            self.champ_dict_by_img_int = dict()
            self.champ_dict_by_champ_int = dict()
            self.item_dict_by_str = dict()
            self.item_dict_by_id = dict()
            self.item_dict_by_img_int = dict()
            self.item_dict_by_item_int = dict()
            self.spell_dict = dict()
            self.build_item_dict()
            self.build_champ_dict()
            self.build_spell_dict()

            self.base_item_path = "../assets/items/"
            self.base_champ_path = "../assets/champs/"
            self.base_self_path = "../assets/self_indicator/"

        def build_spell_dict(self):
            with open('res/spell2id.json') as f:
                try:
                    spell2id = json.load(f)
                except json.decoder.JSONDecodeError as e:
                    print("Error parsing spell2id JSON file")
                    print(e)
                    raise e

            for spell in spell2id.values():
                self.spell_dict.update(
                    dict.fromkeys([('str', spell['name']), ('id', spell['id']), ('int', spell['int'])], spell))

        def build_champ_dict(self):
            with open('res/champ2id.json') as f:
                try:
                    champ2id = json.load(f)
                except json.decoder.JSONDecodeError as e:
                    print("Error parsing champ2id JSON file")
                    print(e)
                    raise e

            deriveds = []
            champ_int_counter = 0
            img_int_counter = 0
            for champ in champ2id.values():
                if "main_img" in champ:
                    deriveds.append(champ)
                else:
                    champ["champ_int"] = champ_int_counter
                    champ["img_int"] = img_int_counter

                    self.champ_dict_by_str[champ['name']] = \
                        self.champ_dict_by_id[champ['id']] = \
                        self.champ_dict_by_champ_int[champ['champ_int']] = \
                        self.champ_dict_by_img_int[champ['img_int']] = champ

                    champ_int_counter += 1
                    img_int_counter += 1

            for derived in deriveds:
                main_img_id = derived["main_img"]
                main_champ = self.champ_dict_by_id[main_img_id]
                derived["champ_int"] = main_champ['champ_int']
                derived["img_int"] = img_int_counter

                self.champ_dict_by_str[derived['name']] = \
                    self.champ_dict_by_id[derived['id']] = \
                    self.champ_dict_by_img_int[derived['img_int']] = derived

                img_int_counter += 1

        def build_item_dict(self):
            with open('res/item2id.json') as f:
                try:
                    item2id = json.load(f)
                except json.decoder.JSONDecodeError as e:
                    print("Error parsing item2id JSON file")
                    print(e)
                    raise e

            unbuyables = []
            deriveds = []

            item_int_counter = 0
            img_int_counter = 0
            for item in item2id.values():

                if "buyable_id" in item:
                    unbuyables.append(item)
                elif "main_img" in item:
                    deriveds.append(item)
                else:
                    item["item_int"] = item_int_counter
                    item["img_int"] = img_int_counter

                    self.item_dict_by_str[item['name']] = \
                        self.item_dict_by_id[item['id']] = \
                        self.item_dict_by_item_int[item['item_int']] = \
                        self.item_dict_by_img_int[item['img_int']] = item


                    item_int_counter += 1
                    img_int_counter += 1

            for unbuyable in unbuyables:
                buyable_id = unbuyable["buyable_id"]
                if buyable_id != "0":
                    buyable_item = self.item_dict_by_id[buyable_id]
                    unbuyable["item_int"] = buyable_item["item_int"]
                else:
                    unbuyable["item_int"] = 0

                unbuyable["img_int"] = img_int_counter

                self.item_dict_by_str[unbuyable['name']] = \
                    self.item_dict_by_id[unbuyable['id']] = \
                    self.item_dict_by_img_int[unbuyable['img_int']] = unbuyable

                img_int_counter += 1

            for derived in deriveds:
                main_img_id = derived["main_img"]
                main_item = self.item_dict_by_id[main_img_id]
                derived["item_int"] = main_item["item_int"]
                derived["img_int"] = img_int_counter

                self.item_dict_by_str[derived['name']] = \
                    self.item_dict_by_id[derived['id']] = \
                    self.item_dict_by_img_int[derived['img_int']] = derived

                img_int_counter += 1

            # first element must be the empty item!
            assert (self.item_id2item_int(0) == 0)
            assert (self.item_id2str(0) == "Empty")

        def get_num_champs(self):

            return len(self.champ_dict_by_champ_int)

        def get_num_champ_imgs(self):
            return len(self.champ_dict_by_img_int)

        def get_num_items(self):
            return len(self.item_dict_by_item_int)

        def get_num_item_imgs(self):
            return len(self.item_dict_by_img_int)

        def get_num_spells(self):
            return len(self.spell_dict)

        def spell_id2int(self, id):
            return self.spell_dict.get(('id', id)).get('int')

        # cvt next item train data to ints
        def item_id2item_int(self, id):
            try:
                return self.item_dict_by_id[str(id)]['item_int']
            except Exception as e:
                print("Problem in itemid2int")
                print(repr(e))
                print(id)
                raise e

        def item_id2str(self, id):
            return self.item_dict_by_id[str(id)]['name']

        def item_str2img_int(self, name):
            return self.item_dict_by_str[name]["img_int"]

        def champ_id2champ_int(self, id):
            return self.champ_dict_by_id[id]["champ_int"]

        def champ_int2id(self, i):
            return self.champ_dict_by_champ_int[i]["id"]

        def champ_str2img_int(self, name):
            return self.champ_dict_by_str[name]["img_int"]

        def img_int2champ_str(self, img_int):
            return self.champ_dict_by_img_int[img_int]["name"]

        def img_int2item_str(self, img_int):
            return self.item_dict_by_img_int[img_int]["name"]

        def get_item_imgs(self):
            return {item["img_int"]: cv.imread(self.base_item_path + item["id"] + ".png") for item in self.item_dict_by_img_int.values()}

        def get_champ_imgs(self):
            return {champ["img_int"]: cv.imread(self.base_champ_path + champ["name"] + ".png") for champ in self.champ_dict_by_img_int.values()}

        def get_self_imgs(self):
            self_img_paths = sorted(glob.glob(self.base_self_path + "*.png"))
            counter = 0
            self_imgs = {counter: cv.imread(template_path) for template_path in self_img_paths}
            return self_imgs