import glob
import numpy as np
import cv2 as cv
import json
import math

SCOREBOARD_SCALING = 0.5

STD_SCOREBOARD_WIDTH = 870
STD_SCOREBOARD_HEIGHT = 352
SCOREBOARD_LEFT_X_OFFSET = 285
SCOREBOARD_Y_OFFSET = 174
SCOREBOARD_BORDER_WIDTH = 8

SCOREBOARD_INNER_LEFT_X_OFFSET = 10
SCOREBOARD_INNER_RIGHT_X_OFFSET = 437
SCOREBOARD_INNER_Y_BOT_OFFSET = 287
SCOREBOARD_INNER_TILE_WIDTH = 425
SCOREBOARD_INNER_TILE_HEIGHT = 57
SCOREBOARD_TOP_TILE_Y_OFFSET = 10
SCOREBOARD_ITEM_BORDER_WIDTH = 10
SCOREBOARD_INNER_BORDER_WIDTH = 3

SCOREBOARD_HEAD_Y_OFFSET = 9
SCOREBOARD_HEAD_Y_OFFSET = 44
SCOREBOARD_HEAD_LEFT_X_OFFSET = 27


CHAMP_LEFT_X_OFFSET = 335
CHAMP_RIGHT_X_OFFSET = 813
CHAMP_Y_OFFSET = 275
CHAMP_Y_DIFF = 64
CHAMP_CIRCLE = 34
CHAMP_SMALL_CIRCLE = 17
CHAMP_SIZE = 23
SMALL_CIRCLE_X_OFFSET = 5
SMALL_CIRCLE_Y_OFFSET = 6
CHAMP_IMG_BLACK_BORDER_WIDTH = 4
CHAMP_MIN_SIZE = 20
CHAMP_MAX_SIZE = 50

ITEM_LEFT_X_OFFSET = 516
ITEM_RIGHT_X_OFFSET = 992
ITEM_Y_DIFF = 64
ITEM_Y_OFFSET = 276
ITEM_SIZE = 24
ITEM_MIN_SIZE = 15
ITEM_MAX_SIZE = 40
ITEM_X_DIFF = 28

SPELL_LEFT_X_OFFSET = 276
SPELL_RIGHT_X_OFFSET = 754
SPELL_Y_DIFF = 41
SPELL_Y_DIFF_LARGE = 64
SPELL_Y_OFFSET = 267
SPELL_SIZE = 22
SPELL_MIN_SIZE = 15
SPELL_MAX_SIZE = 40

SELF_INDICATOR_LEFT_X_OFFSET = 247
SELF_INDICATOR_RIGHT_X_OFFSET = 723
SELF_INDICATOR_Y_DIFF = 64
SELF_INDICATOR_Y_OFFSET = 300
SELF_INDICATOR_SIZE = 15



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


