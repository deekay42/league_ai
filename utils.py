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

    champ_names_old = ['TwistedFate', 'Akali', 'Sivir', 'KogMaw', 'Nidalee', 'MegaGnar', 'Kled', 'Rengar', 'Tristana', 'Volibear',
     'Leona', 'Amumu', 'Katarina', 'Sona', 'VelKoz', 'Nautilus', 'Rumble', 'Veigar', 'Teemo', 'Annie', 'Blitzcrank',
     'Alistar', 'Rhaast', 'Kassadin', 'Ziggs', 'MasterYi', 'LeeSin', 'Karthus', 'Varus', 'Vi', 'Udyr', 'Nami',
     'Orianna', 'Talon', 'Rakan', 'Sejuani', 'Yasuo', 'Karma', 'Ryze', 'Gangplank', 'Kindred', 'LeBlanc', 'Zed',
     'Syndra', 'Warwick', 'Ashe', 'Malzahar', 'Cassiopeia', 'Ekko', 'Morgana', 'Lulu', 'Vayne', 'KhaZix', 'Jinx',
     'XinZhao', 'Aatrox', 'Thresh', 'KaiSa', 'Ahri', 'Lucian', 'Ezreal', 'Maokai', 'Jhin', 'Skarner', 'Trundle',
     'Jayce', 'Zoe', 'Swain', 'Elise', 'Hecarim', 'Irelia', 'Fizz', 'Singed', 'Dr.Mundo', 'Camille', 'Soraka',
     'JarvanIV', 'Nocturne', 'Fiddlesticks', 'Rammus', 'Quinn', 'Kennen', 'Gnar', 'Zac', 'Mordekaiser', 'Renekton',
     'Anivia', 'Nasus', 'Pyke', 'Shyvana', 'ShadowAssassin', 'Tryndamere', 'Bard', 'Kalista', 'MissFortune', 'Caitlyn',
     'Janna', 'AurelionSol', 'Kayle', 'Urgot', 'Xerath', 'Corki', 'Malphite', 'Ornn', 'Graves', 'Garen', 'Nunu',
     'RekSai', 'Zilean', 'Ivern', 'Kayn', 'Pantheon', 'Illaoi', 'Twitch', 'Draven', 'Diana', 'TahmKench', 'Viktor',
     'Xayah', 'Wukong', 'Galio', 'Vladimir', 'Shen', 'Lux', 'Braum', 'Olaf', 'Taric', 'Zyra', 'Lissandra',
     'Evelynn', 'Poppy', 'Gragas', 'Azir', 'Darius', 'Jax', 'Heimerdinger', 'Shaco', 'Fiora', 'Yorick', 'Sion', 'Brand',
     'Riven', 'ChoGath', 'Taliyah']
    # TODO: This is really bad
    # need to have a single source of truth for champ names, items, etc.
    item_names_old = ["Sorcerer's Shoes", "Mikael's Crucible", 'Travel Size Elixir of Wrath', 'Crystalline Bracer', 'The Hex Core mk-2', 'Chain Vest', 'Quicksilver Sash', 'Hexdrinker', 'Cloak of Agility', 'The Hex Core mk-1', 'Edge of Night', 'Righteous Glory', 'Pilfered Health Potion', 'The Obsidian Cleaver', 'Last Whisper', 'Boots of Swiftness', "Bami's Cinder", 'Salvation', 'Sweeping Lens (Trinket)', 'Lost Chapter', 'Adaptive Helm', 'Blasting Wand', "Zhonya's Hourglass", "Randuin's Omen", "Youmuu's Ghostblade", 'Null-Magic Mantle', 'Sapphire Crystal', 'Maw of Malmortius', 'Warding Totem (Trinket)', 'Pilfered Potion of Rouge', 'Circlet of the Iron Solari', "Lord Dominik's Regards", 'Frozen Mallet', 'Guardian Angel', "Seeker's Armguard", 'Hextech Protobelt-01', 'Pickaxe', 'Hextech Gunblade', 'Long Sword', 'Duskblade of Draktharr', 'Blade of the Ruined King', "Death's Dance", 'Locket of the Iron Solari', 'Infernal Mask', "Zhonya's Paradox", 'Mortal Reminder', 'Rod of Ages', 'Infinity Edge', 'Iceborn Gauntlet', 'Elixir of Sorcery', 'Trinity Fusion', 'Travel Size Elixir of Sorcery', "Shurelya's Reverie", 'Hextech Revolver', "Liandry's Torment", 'Bilgewater Cutlass', 'Gargoyle Stoneplate', 'Travel Size Elixir of Iron', 'Enchantment: Bloodrazor: Sabre', 'Elixir of Iron', "Dead Man's Plate", 'Glacial Shroud', 'Hextech GLP-800', 'Oblivion Orb', "Death's Daughter", 'Rapid Firecannon', 'Sheen', 'Health Potion', "Executioner's Calling", 'Amplifying Tome', 'Haunting Guise', 'Vampiric Scepter', 'Muramana', 'Ohmwrecker', 'Stormrazor', 'Raise Morale', 'Fire at Will', "Warmog's Armor", "Targon's Brace", 'Sunfire Cape', "Seraph's Embrace", 'Fiendish Codex', 'Serrated Dirk', "Brawler's Gloves", 'Void Staff', "Knight's Vow", 'Kircheis Shard', "Mejai's Soulstealer", 'Remnant of the Ascended', "Nomad's Medallion", "Warden's Mail", 'Zeal', 'Remnant of the Watchers', 'Cull', "Doran's Shield", "Guinsoo's Rageblade", "Doran's Blade", "Hunter's Machete", 'The Dark Seal', 'Mana Potion', 'Essence Reaver', "Poacher's Dirk", "Luden's Echo", 'Trinity Force', "Zeke's Convergence", 'Phage', 'Twin Shadows', 'Statikk Shiv', "Wit's End", "Runaan's Hurricane", 'Spellbinder', "Jaurim's Fist", 'Phantom Dancer', 'Negatron Cloak', 'Recurve Bow', "Caulfield's Warhammer", 'Dagger', "Doran's Ring", 'Minion Dematerializer', 'Ninja Tabi', "Sterak's Gage", "Rabadon's Deathcap", 'Bramble Vest', "Zz'Rot Portal", "Banshee's Veil", "Rylai's Crystal Scepter", 'Boots of Mobility', 'Ancient Coin', 'Tiamat', 'Thornmail', 'Total Biscuit of Rejuvenation', "Spellthief's Edge", 'Needlessly Large Rod', "Nashor's Tooth", 'Stinger', 'Lich Bane', 'Forbidden Idol', 'Relic Shield', 'Ardent Censer', 'Elixir of Wrath', 'Ravenous Hydra', 'Banner of Command', "Stalker's Blade", 'Broken Stopwatch', 'Tear of the Goddess', 'Frozen Heart', 'Mercurial Scimitar', 'Aegis of the Legion', "Mercury's Treads", 'Refillable Potion', 'The Black Cleaver', 'Spirit Visage', 'Frostfang', "Skirmisher's Sabre", 'Kindlegem', 'Stopwatch', 'Corrupting Potion', 'Redemption', 'Aether Wisp', "Hunter's Potion", 'Slightly Magical Boots', 'The Bloodthirster', 'Abyssal Mask', 'Enchantment: Warrior', 'Enchantment: Runic Echoes', 'Control Ward', 'Farsight Alteration', 'B. F. Sword', 'Faerie Charm', "Athene's Unholy Grail", "Giant's Belt", "Hunter's Talisman", 'Enchantment: Cinderhulk', 'Chalice of Harmony', 'The Black Spear', 'Prototype Hex Core', 'Pilfered Stealth Ward', "Rabadon's Deathcrown", 'Rejuvenation Bead', 'Enchantment: Runic Echoes: Sabre', 'Enchantment: Bloodrazor', "Archangel's Staff", 'Sly Sack of Gold', 'Enchantment: Warrior: Sabre', "Spectre's Cowl", 'Molten Edge', 'Raptor Cloak', 'Perfect Hex Core', 'Oracle Alteration', 'Enchantment: Cinderhulk: Sabre', 'Titanic Hydra', "Berserker's Greaves", 'Catalyst of Aeons', 'Manamune', 'Boots of Speed', 'Ionian Boots of Lucidity', 'Cloth Armor', 'Ruby Crystal', 'Morellonomicon', 'Forgefire Cape', 'Remnant of the Aspect', 'Empty']

    champ_int2string_old = dict(zip(np.arange(0, len(champ_names_old)), champ_names_old))
    item_int2string_old = dict(zip(np.arange(0, len(item_names_old)), item_names_old))

    def __init__(self):
        with open('res/item2id') as f:
            item2id = json.load(f)

        self.item_dict = dict()
        for item in item2id.values():
            self.item_dict.update(dict.fromkeys([('str', item['name']), ('id', item['id']), ('int', item['int'])], item))

        with open('res/champ2id') as f:
            champ2id = json.load(f)

        self.champ_dict = dict()
        for champ in champ2id.values():
            self.champ_dict.update(dict.fromkeys([('str', champ['name']), ('id', champ['id']), ('int', champ['int'])], champ))

        with open('res/spell2id') as f:
            spell2id = json.load(f)

        self.spell_dict = dict()
        for spell in spell2id.values():
            self.spell_dict.update(dict.fromkeys([('str', spell['name']), ('id', spell['id']), ('int', spell['int'])], spell))

    def spell_id2int(self, id):
        return self.spell_dict.get(('id', id)).get('int')

    def champ_int2id(self, id):
        return self.champ_dict.get(('int', id)).get('id')

    def item_int2id(self, i):
        return self.item_dict.get(('int', i)).get('id')

    def item_int2string(self, i):
        return self.item_dict.get(('int', i)).get('name')

    def item_id2string(self, i):
        if i == 2420:
            i = 2423
        return self.item_dict.get(('id', i)).get('name')

    def item_id2int(self, id):
        if id == [] or id == 0:
            return 0
		#this is really bad TODO:
        # elif id == 2301:
        #     id = 3092
        # elif id == 2302:
        #     id = 3092
        # elif id == 2303:
        #     id = 3092
        return self.item_dict.get(('id', id)).get('int')

    def champ_id2int(self, id):
        return self.champ_dict.get(('id', id)).get('int')

    def champ_int2string(self, i):
        return self.champ_dict.get(('int', i)).get('name')

    def champ_string2id(self, champ_string):
        return self.champ_dict.get(('str', champ_string)).get('id')

    def item_string2id(self, item_string):
        return self.item_dict.get(('str', item_string)).get('id')

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


    # champ_templates = [cv.resize(template, s  ize, interpolation=cv.INTER_AREA) for
    #                         template in champ_templates]

    return dict(zip(champ_names, champ_templates))


def generateItemCoordinates(box_size, left_x, right_x, y_diff, top_left_trinket_y, x_offset, y_offset):
    item_slots_coordinates = np.zeros((2, 5, 7, 2), dtype=np.int64)
    total_x_offsets = (left_x + x_offset, right_x+x_offset)
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
    return 1920,1080
    # TODO:

def summ_names_displayed():
    return True

def chunks(l, n):
    n = max(1, n)
    return [l[i:i + n] for i in range(0, len(l), n)]