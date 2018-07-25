import numpy as np
import pyautogui
import cv2 as cv
import keyboard
import glob
import os
from predict import Predictor

def take_screenshot():
    image = pyautogui.screenshot()
    image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
    return image

    
def take_windows_screenshot():
    winpath = 'L:\Spiele\Screenshots\*'
    list_of_files = glob.glob(winpath) # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    screenshot = cv.imread(winpath*latest_file)
    return screenshot
    
predict = Predictor()
while True:
    keyboard.wait('tab')
    print('you pressed tab')
    screenshot = take_screenshot()
    champs, spells, items, self_ = predict(screenshot)
    print(champs)
    print(spells)
    print(items)
    print(self_)
# cv.imshow('fds', screenshot)
# cv.waitKey(0)