import glob
import numpy as np
import cv2 as cv
import json
import math


def getItemTemplateDict():
    with open('res/item2id') as f:
        item2id = json.load(f)
    id2item = dict(map(reversed, item2id.items()))


    item_paths = glob.glob("res/item/*.png")
    item_names = [item_name[9:-4] for item_name in item_paths]
    item_templates = [cv.imread(template_path) for template_path in item_paths]
    item_template_height = item_templates[0].shape[1]

    item_names = [id2item[int(item)] for item in item_names]
    result = dict(zip(item_names, item_templates))

    # add empty item slot
    result["Empty"] = np.array([[[7, 13, 12]] * item_template_height] * item_template_height,
                               dtype=np.uint8)

    return result

class Converter:
    def __init__(self):
        with open('res/item2id') as f:
            item2id = json.load(f)
        ids = sorted(list(item2id.values()))
        seq_num = np.arange(0, len(ids), dtype=np.uint16)
        self.item_id2int_dict = dict(zip(ids, seq_num))

        with open('res/champ2id') as f:
            champ2id = json.load(f)
        ids = sorted(list(champ2id.values()))
        seq_num = np.arange(0, len(ids), dtype=np.uint16)
        self.champ_id2int_dict = dict(zip(ids, seq_num))

    def item_id2int(self, id):
        if id == [] or id == 0:
            return 0
        # there are some items in the training files that have since been deleted
        if id == 3711:
            id = 1412
        elif id == 1409:
            id = 1413
        elif id == 1408:
            id = 1412
        elif id == 1410:
            id = 1402
        elif id == 3034:
            id = 1037
        elif id == 1418:
            id = 1419
        elif id == 2420:
            id = 2423
        elif id == 2420:
            id = 2423
        elif id == 2421:
            id = 2424
        elif id == 2045:
            id = 0
        elif id == 2049:
            id = 0
		#this is really bad TODO:
        elif id == 2301:
            id = 3092
        elif id == 2302:
            id = 3092
        elif id == 2303:
            id = 3092



        return self.item_id2int_dict[id]

    def champ_id2int(self, id):
        return self.champ_id2int_dict[id]


def getSelfTemplateDict():
    self_paths = glob.glob("res/self_indicator/*.png")
    self_names = [name[18:-4] for name in self_paths]
    self_templates = [cv.imread(template_path) for template_path in self_paths]
    result = dict(zip(self_names, self_templates))

    return result


def getSpellTemplateDict():
    spell_path = 'res/sprite/spell0.png'
    spell_template = cv.imread(spell_path)
    spell_height_png = 48
    spell_templates = [spell_template[0:spell_height_png, 0 * spell_height_png:1 * spell_height_png],
                       spell_template[0:spell_height_png, 1 * spell_height_png:2 * spell_height_png],
                       spell_template[0:spell_height_png, 4 * spell_height_png:5 * spell_height_png],
                       spell_template[0:spell_height_png, 5 * spell_height_png:6 * spell_height_png],
                       spell_template[0:spell_height_png, 6 * spell_height_png:7 * spell_height_png],
                       spell_template[0:spell_height_png, 7 * spell_height_png:8 * spell_height_png],
                       spell_template[0:spell_height_png, 8 * spell_height_png:9 * spell_height_png],
                       spell_template[1 * spell_height_png: 2 * spell_height_png,
                       4 * spell_height_png:5 * spell_height_png],
                       spell_template[1 * spell_height_png: 2 * spell_height_png,
                       7 * spell_height_png:8 * spell_height_png]]
    spell_names = ["Barrier", "Cleanse", "Ignite", "Exhaust", "Flash", "Ghost", "Heal", "Smite", "Teleport"]

    # spell_templates = [cv.resize(template, size, interpolation=cv.INTER_AREA) for
    #                    template in spell_templates]

    return dict(zip(spell_names, spell_templates))


def getChampTemplateDict():
    champ_paths = glob.glob('res/champion/*')
    champ_templates = [cv.imread(template_path) for template_path in champ_paths]
    # champ_templates = [champ_template[CHAMP_IMG_BLACK_BORDER_WIDTH:-CHAMP_IMG_BLACK_BORDER_WIDTH, CHAMP_IMG_BLACK_BORDER_WIDTH:-CHAMP_IMG_BLACK_BORDER_WIDTH] for champ_template in champ_templates]
    # champ_slot_height = champ_templates[0].shape[0]

    # mini_circle_x_offset = int(champ_slot_height * SMALL_CIRCLE_Y_OFFSET / CHAMP_CIRCLE)
    # mini_circle_y_offset = int(champ_slot_height * SMALL_CIRCLE_X_OFFSET / CHAMP_CIRCLE)
    # mini_circle_radius = int((champ_slot_height * CHAMP_SMALL_CIRCLE / CHAMP_CIRCLE) // 2)
    #
    # circle_img = np.zeros((champ_slot_height, champ_slot_height), np.uint8)
    # cv.circle(circle_img, (champ_slot_height // 2,  champ_slot_height // 2), champ_slot_height // 2, 255, thickness=-1)
    # cv.circle(circle_img, (mini_circle_radius-mini_circle_x_offset, champ_slot_height - mini_circle_radius+mini_circle_y_offset), mini_circle_radius, 0,
    #           thickness=-1)
    # cv.circle(circle_img, (champ_slot_height - mini_circle_radius+mini_circle_x_offset, champ_slot_height - mini_circle_radius+mini_circle_y_offset),
    #           mini_circle_radius, 0,
    #           thickness=-1)
    # champ_templates = [cv.bitwise_and(template, template, mask=circle_img) for template in champ_templates]
    #
    champ_names = [champ_name[13:-4] for champ_name in champ_paths]


    # champ_templates = [cv.resize(template, size, interpolation=cv.INTER_AREA) for
    #                         template in champ_templates]

    return dict(zip(champ_names, champ_templates))


def generateItemCoordinates(box_size, left_x, right_x, y_diff, top_left_trinket_y):
    item_slots_coordinates = np.zeros((2, 5, 7, 2), dtype=np.int64)
    x_offsets = (left_x, right_x)
    for x_offset, team in zip(x_offsets, range(2)):
        for player in range(5):
            for item in range(7):
                item_slots_coordinates[team][player][item] = (
                    int(x_offset + item * box_size), int(top_left_trinket_y + player * y_diff))
    return item_slots_coordinates


def generateSpellCoordinates(box_size, left_x, right_x, y_diff, top_left_spell_y):
    spell_slots_coordinates = np.zeros((2, 5, 2, 2), dtype=np.int64)
    x_offsets = (left_x, right_x)
    for x_offset, team in zip(x_offsets, range(2)):
        for player in range(5):
            spell_slots_coordinates[team][player][0] = (
                int(x_offset), int(top_left_spell_y + player * (y_diff + box_size + 1)))
            spell_slots_coordinates[team][player][1] = (
                int(x_offset), int(top_left_spell_y + box_size + player * (y_diff + box_size)) + 1)
    return spell_slots_coordinates

def generateSpellCoordinatesLarge(box_size, left_x, right_x, y_diff, top_left_spell_y):
    spell_slots_coordinates = np.zeros((2, 5, 2, 2), dtype=np.int64)
    x_offsets = (left_x, right_x)
    for x_offset, team in zip(x_offsets, range(2)):
        for player in range(5):
            spell_slots_coordinates[team][player][0] = (
                int(x_offset), int(top_left_spell_y + player * y_diff))
            spell_slots_coordinates[team][player][1] = (
                int(x_offset), spell_slots_coordinates[team][player][0][1] + box_size + 1)
    return spell_slots_coordinates

def generateChampCoordinates(left_x, right_x, y_diff, top_left_spell_y):
    champ_slots_coordinates = np.zeros((2, 5, 2), dtype=np.int64)
    x_offsets = (left_x, right_x)
    for x_offset, team in zip(x_offsets, range(2)):
        for player in range(5):
            champ_slots_coordinates[team][player] = (
                int(x_offset), int(top_left_spell_y + player * y_diff))
    return champ_slots_coordinates

# spells need to be sorted
def generateChampionCoordsBasedOnSpellCoords(left_side_spells, right_side_spells):
    champ_slot_coords = np.zeros((2, 5, 2), dtype=np.int64)
    spell_width = math.fabs(left_side_spells[0][1][1] - left_side_spells[0][0][1])
    for team_index, team in zip(range(2), [left_side_spells, right_side_spells]):
        x_offset = team[0][0][0]
        for top_spell in range(5):
            champ_slot_coords[team_index][top_spell][0] = x_offset + 2 * spell_width
            champ_slot_coords[team_index][top_spell][1] = left_side_spells[top_spell][0][1]
    return champ_slot_coords

def champ_int2string():
    champ_imgs = init_champ_data_for_training()
    champ_imgs2 = getChampTemplateDict()
    return dict(zip(champ_imgs.keys(), champ_imgs2.keys()))

def item_int2string():
    item_imgs = init_item_data_for_training()
    item_imgs2 = getItemTemplateDict()
    return dict(zip(item_imgs.keys(), item_imgs2.keys()))

def spell2int():
    spells = {"Barrier": 0, "Cleanse": 1, "Ignite": 2, "Exhaust": 3, "Flash": 4, "Ghost": 5, "Heal": 6, "Smite": 7,
              "Teleport": 8}
    return spells

def int2spell():
    spells = {0: "Barrier", 1: "Cleanse", 2: "Ignite", 3: "Exhaust", 4: "Flash", 5: "Ghost", 6: "Heal", 7: "Smite", 8:
              "Teleport"}
    return spells

def init_item_data_for_training():
    item_imgs = getItemTemplateDict()
    imgs = list(item_imgs.values())
    newkeys = np.arange(0, len(list(item_imgs.keys())))
    return dict(zip(newkeys, imgs))

def init_self_data_for_training():
    self_imgs = getSelfTemplateDict()
    imgs = list(self_imgs.values())
    newkeys = [1,0]
    return dict(zip(newkeys, imgs))


def init_spell_data_for_training(image_size):
    spell_imgs = getSpellTemplateDict()
    imgs = list(spell_imgs.values())
    newkeys = np.arange(0, len(list(spell2int().keys())))
    imgs = [cv.resize(img, image_size, interpolation=cv.INTER_AREA) for img in imgs]
    return dict(zip(newkeys, imgs))


def init_champ_data_for_training():
    item_imgs = getChampTemplateDict()
    imgs = list(item_imgs.values())
    newkeys = np.arange(0, len(list(item_imgs.keys())))
    return dict(zip(newkeys, imgs))
    
def cvtHrzt(std, current):
    return std
    
def cvtVert(std, current):
    return std
    
def getResolution():
    return 1440,900
    # TODO:






