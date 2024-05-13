# Importing libraries for the project
import cv2
import numpy as np
import utils

webcam = False
path='1.jpg'
cap = cv2.VideoCapture(0)
cap.set(10,160) #Set property 10 (brightness)
cap.set(3,1920) #Set property 3 (frame width)
cap.set(4,1080) #Set porperty 4 (frame lenght)

while True:
    if webcam : 
        success, img =cap.read()
    else:
        img = cv2.imread(path)

    img, contours = utils.getContours(img,
                                           cannyThreshold=[150,150], #Threshold for certanty
                                           showCanny=True,           #Show Canny
                                           draw=True,                #Draw contours
                                           minArea=50000,            #Minimum Area to be detected
                                           filter=4)                 #Filter based on number of corners
    if len(contours) != 0:
        biggest = contours[0][2]
        print(biggest)
        
    img = cv2.resize(img,(0,0),None,0.5,0.5)
    cv2.imshow('original', img)
    cv2.waitKey(1)

#test comment
