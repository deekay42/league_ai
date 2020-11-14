import copy
import glob
import os
import sys
from collections import Counter
from tkinter import Tk
from tkinter import messagebox
from tkinter.filedialog import askdirectory

import cv2 as cv
import numpy as np

from utils.artifact_manager import ItemManager
from constants import game_constants,app_constants
# from train_model import data_loader
from train_model.input_vector import Input
import psutil


def findProcessIdByName(processName):
    '''
    Get a list of all the PIDs of a all the running process whose name contains
    the given string processName
    '''
    listOfProcessObjects = []
    #Iterate over the all the running process
    for proc in psutil.process_iter():
        try:
            pinfo = proc.as_dict(attrs=['pid', 'name', 'create_time'])
            # Check if process name contains the given name string.
            if processName.lower().replace(' ', '') in pinfo['name'].lower().replace(' ', '') :
                return pinfo['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied , psutil.ZombieProcess) :
            pass
    else:
        return -1

def num_itemslots(items):
    if not items:
        return 0
    wards = ItemManager().lookup_by("name", "Control Ward")["int"]
    hpots = ItemManager().lookup_by("name", "Health Potion")["int"]
    redpot = ItemManager().lookup_by("name", "Elixir of Wrath")["int"]
    bluepot = ItemManager().lookup_by("name", "Elixir of Sorcery")["int"]
    ironpot = ItemManager().lookup_by("name", "Elixir of Iron")["int"]
    num_single_slot_items = int(items.get(wards, 0) > 0) + int(items.get(hpots, 0) > 0)
    reg_item_keys = (set(items.keys()) - {hpots, wards, redpot, bluepot, ironpot})
    num_reg_items = sum([items[key] for key in reg_item_keys])
    return num_single_slot_items + num_reg_items


def itemslots_left(items=None):
    return game_constants.MAX_ITEMS_PER_CHAMP - num_itemslots(items)


def iditem2intitems(items):
    return Counter({ItemManager().lookup_by("id", str(itemid))["int"]: qty for itemid, qty in items.items()})


def show_coords_all(img_source, champ_coords, champ_size, item_coords, item_size, self_coords, self_size):
    img = copy.deepcopy(img_source)
    for coord in champ_coords:
        cv.rectangle(img, tuple(coord), (coord[0] + champ_size, coord[1] + champ_size), (255, 0, 0), 1)
    for coord in item_coords:
        cv.rectangle(img, tuple(coord), (coord[0] + item_size, coord[1] + item_size), (255, 0, 0), 1)
    for coord in self_coords:
        cv.rectangle(img, tuple(coord), (coord[0] + self_size, coord[1] + self_size), (255, 0, 0), 1)
    cv.imshow("lol", img)
    cv.waitKey(0)


# def plot_hist():
#     import matplotlib.pyplot as plt
#     from sklearn.preprocessing import power_transform, minmax_scale
#     dataloader_elite = data_loader.SortedNextItemsDataLoader(app_constants.train_paths[
#                                                                  "next_items_processed_elite_sorted_inf"])
#     X_elite, Y_elite = dataloader_elite.get_train_data()

#     d = np.reshape(X_elite[:, Input.current_gold_start:Input.current_gold_end], (-1,1))
#     # d = np.clip(d, game_constants.min_clip["cs"], game_constants.max_clip["cs"])
#     # d = power_transform(d, method='yeo-johnson')
#     # d = minmax_scale(d)

#     # An "interface" to matplotlib.axes.Axes.hist() method
#     n, bins, patches = plt.hist(x=d, bins='auto', color='#0504aa',
#                                 alpha=0.7, rwidth=0.85)
#     plt.grid(axis='y', alpha=0.75)
#     plt.xlabel('Value')
#     plt.ylabel('Frequency')
#     plt.title('My Very Own Histogram')
#     maxfreq = n.max()
#     plt.ylim(ymax=maxfreq)
#     plt.show()
#     print("lol")


def show_coords(img_source, coords, size_x, size_y):
    img = copy.deepcopy(img_source)
    for coord in coords:
        cv.rectangle(img, tuple(coord), (int(coord[0] + size_x), int(coord[1] + size_y)), (255, 0, 0), 1)
    cv.namedWindow('image', cv.WINDOW_NORMAL)
    cv.resizeWindow('image', 800, 450)
    cv.imshow("image", img)
    cv.waitKey(0)


def remove_old_files(path):
    old_filenames = glob.glob(path + '*')
    for filename in old_filenames:
        os.remove(filename)


def preprocess(img, med_it, kernel):
    for _ in range(med_it):
        img = cv.medianBlur(img, kernel)
    return img


def chunks(l, n):
    n = max(1, n)
    return [l[i:i + n] for i in range(0, len(l), n)]


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def get_lol_dir():
    while True:
        if os.path.exists(os.path.join(os.getenv('LOCALAPPDATA'), "League IQ", "loldir")):
            with open(os.path.join(os.getenv('LOCALAPPDATA'), "League IQ", "loldir"), "r") as f:
                loldir = f.read()
        else:
            loldir = "C:/Riot Games/League of Legends"
        if not (os.path.isdir(loldir + "/Game")):
            query_lol_dir()
        else:
            return loldir



def query_lol_dir():
    Tk().withdraw()
    messagebox.showinfo("Information",
                        "We were unable to locate your League of Legends installation. Please select your main League of Legends folder.")
    loldir = askdirectory(initialdir="C:", title="Please select your main League of Legends folder")

    while not (os.path.isdir(loldir + "/Game")):
        messagebox.showinfo("Information", "That wasn't it. Select the folder that has the Game folder in it.")
        loldir = askdirectory(initialdir="C:", title="Please select your main League of Legends folder")
        if loldir == "":
            sys.exit()
    print(loldir)
    screenshot_dir = loldir + "/Screenshots"
    try:
        os.makedirs(screenshot_dir)
    except Exception as e:
        print(e)
    with open(os.path.join(os.getenv('LOCALAPPDATA'), "League IQ", "loldir"), "w") as f:
        f.write(loldir)


# no need to shuffle here. only costs time. shuffling will happen during training before each epoch

def uniform_shuffle(l1, l2):
    assert len(l1) == len(l2)
    rng_state = np.random.get_state()
    np.random.shuffle(l1)
    np.random.set_state(rng_state)
    np.random.shuffle(l2)
