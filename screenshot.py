import numpy as np
import pyautogui
import cv2 as cv
import keyboard
import glob
import os
from predict import Predictor
import time

def take_screenshot():
    image = pyautogui.screenshot()
    image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
    return image

    
def take_windows_screenshot():
    folder = 'L:\Spiele\Screenshots\*'
    list_of_files = glob.glob(folder) # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    print(latest_file)
    screenshot = cv.imread("L:\Spiele\Screenshots\Screen15.png")
    return screenshot
    

predict = Predictor()
while True:
    keyboard.wait('tab+f12')
    print('you pressed tab + f12')
    time.sleep(2)
    screenshot = take_windows_screenshot()
    champs, spells, items, self_ = predict(screenshot)
    print(champs)
    print(spells)
    print(items)
    print(self_)