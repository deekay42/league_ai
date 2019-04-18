import math

import numpy as np


def generateItemCoordinates(box_size, left_x, right_x, y_diff, top_left_trinket_y, x_offset, y_offset):
    item_slots_coordinates = np.zeros((2, 5, 7, 2), dtype=np.int64)
    total_x_offsets = (left_x + x_offset, right_x + x_offset)
    for total_x_offset, team in zip(total_x_offsets, range(2)):
        for player in range(5):
            for item in range(7):
                item_slots_coordinates[team][player][item] = (
                    round(total_x_offset + item * box_size), round(top_left_trinket_y + y_offset + player * y_diff))
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
                round(x_offset), round(top_left_spell_y + player * y_diff))
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


def chunks(l, n):
    n = max(1, n)
    return [l[i:i + n] for i in range(0, len(l), n)]


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
