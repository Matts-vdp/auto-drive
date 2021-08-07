import cv2
import keyboard
import numpy as np
import pyautogui
import win32gui
from time import sleep, time_ns

width, height = 100, 100
path = "images/"

# used to take screenchots of a window
def screenshot(window_title=None):
    hwnd = win32gui.FindWindow(None, window_title)
    if hwnd:
        x, y, x1, y1 = win32gui.GetClientRect(hwnd)
        x, y = win32gui.ClientToScreen(hwnd, (x, y))
        x1, y1 = win32gui.ClientToScreen(hwnd, (x1 - x, y1 - y))
        im = pyautogui.screenshot(region=(x, y, x1, y1))
        img_np = np.array(im)
        return img_np
    else:
        print('Window not found!')

# Used to make the image ready to be stored on disk and used as training data
def preprocess(img):
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# used to check wich keys are pressed
def getKeys():
    k = ["left", "right"]
    for key in k:
        if keyboard.is_pressed(key):
            return key
    return "none"

# main loop
# saves the taken images in a folder corresponding to the pressed key
def takeScreen():
    print("press s to start")
    keyboard.wait("s")
    sleep(1)
    print("started")
    i = 0
    while True:
        if keyboard.is_pressed("s"):
            break
        img = screenshot("car (DEBUG)")
        key = getKeys()
        img = preprocess(img)
        filename = path + key + "/" + str(i) + ".png"
        i += 1 
        if i%1000 == 0:
            print(i)
        cv2.imwrite(filename, img)
    print("stopped")

if __name__ == "__main__":
    takeScreen()