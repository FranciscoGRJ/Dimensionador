# Importing libraries for the project
import cv2
import numpy as np

webcam = False
path='1.jpg'
cap = cv2.VideoCapture(0)
cap.set(10,160) #Set property 10 (brightness)
cap.set(3,1920)
cap.set(4,1080)

while True:
    success, img =cap.read()
    
    cv2.imshow('original', img)
    cv2.waitKey(1)
