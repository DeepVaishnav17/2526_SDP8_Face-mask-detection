import pyautogui
import time


time.sleep(5)
for i in range (100):
    pyautogui.hotkey("ctrl","v")
    pyautogui.press("enter")
    time.sleep(0.5)


