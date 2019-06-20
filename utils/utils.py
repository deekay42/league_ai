import glob
import os
from tkinter import Tk
from tkinter import messagebox
from tkinter.filedialog import askdirectory
import copy

import cv2 as cv


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


def show_coords(img_source, coords, size):
    img = copy.deepcopy(img_source)
    for coord in coords:
        cv.rectangle(img, tuple(coord), (coord[0] + size, coord[1] + size), (255, 0, 0), 1)
    cv.imshow("lol", img)
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
        if not os.path.isdir(loldir + "/RADS"):
            query_lol_dir()
        else:
            return loldir


def query_lol_dir():
    Tk().withdraw()
    messagebox.showinfo("Information",
                        "We were unable to locate your League of Legends installation. Please select your main League of Legends folder.")
    loldir = askdirectory(initialdir="C:", title="Please select your main League of Legends folder")
    
    while not os.path.isdir(loldir + "/RADS"):
        messagebox.showinfo("Information", "That wasn't it. Select the folder that has the RADS folder in it.")
        loldir = askdirectory(initialdir="C:", title="Please select your main League of Legends folder")]
        if loldir == "":
            exit()
    print(loldir)
    screenshot_dir = loldir + "/Screenshots"
    try:
        os.makedirs(screenshot_dir)
    except Exception as e:
        print(e)
    with open(os.path.join(os.getenv('LOCALAPPDATA'), "League IQ", "loldir"), "w") as f:
        f.write(loldir)
