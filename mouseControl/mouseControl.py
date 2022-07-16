from turtle import screensize
import pyautogui

print(pyautogui.size())

screenWidth, screenHeight = pyautogui.size()

xMid = screenWidth / 2
yMid = screenHeight / 2

print(xMid, yMid)

for i in range(0,5):
    pyautogui.moveTo(xMid + 100, yMid, 1)
    pyautogui.moveTo(xMid, yMid + 100, 1)
    pyautogui.moveTo(xMid - 100, yMid, 1)
    pyautogui.moveTo(xMid, yMid - 100, 1)