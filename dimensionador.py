# Importing libraries for the project
import cv2
import numpy as np
import utils
import time 

webcam = False
path='2.jpg'
cap = cv2.VideoCapture(0)
cap.set(10,160) #Set property 10 (brightness)
cap.set(3,1920) #Set property 3 (frame width)
cap.set(4,1080) #Set porperty 4 (frame lenght)

scaleFactor = 3
referenceWidth = 210*scaleFactor
referenceHeigth = 297*scaleFactor

newHeigth = 0
newWidth= 0
logHeight = 0
logWidth = 0


while True:
    
    if newWidth !=0 or newHeigth != 0:
        print("enter delay")
        time.sleep(5)
    
    if webcam : 
        success, img =cap.read()
    else:
        img = cv2.imread(path)

    img, contours = utils.getContours(img,
                                           minArea=50000,            #Minimum Area to be detected
                                           filter=4)                 #Filter based on number of corners
    if len(contours) != 0:
        biggest = contours[0][2]
        #print(biggest)
        imgWarp = utils.warpImage (img,biggest,referenceWidth,referenceHeigth)
        #cv2.imshow('A4', imgWarp)
        
        img2, contours2 = utils.getContours(imgWarp,
                                           cannyThreshold=[50,50], #Threshold for certanty
                                           draw=False,                #Draw contours
                                           minArea=2000,            #Minimum Area to be detected
                                           filter=4)                 #Filter based on number of corners
        if len(contours) != 0:
            for obj in contours2:
                cv2.polylines(img2,[obj[2]],True,(0,255,0),2)
                newPoint = utils.reorder(obj[2])
                newWidth = round((utils.findDis(newPoint[0][0]//scaleFactor,newPoint[1][0]//scaleFactor)/10),1)
                newHeigth = round((utils.findDis(newPoint[0][0]//scaleFactor,newPoint[2][0]//scaleFactor)/10),1)
                
                cv2.arrowedLine(img2,(newPoint[0][0][0], newPoint[0][0][1]),(newPoint[1][0][0],newPoint[1][0][1]), (255,0,255),3,8,0,0.05)
                cv2.arrowedLine(img2,(newPoint[0][0][0], newPoint[0][0][1]),(newPoint[2][0][0],newPoint[2][0][1]), (255,0,255),3,8,0,0.05)
                x,y,w,h = obj[3]
                cv2.putText(img2, '{}cm'.format(newWidth), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255,0,255),2)
                cv2.putText(img2, '{}cm'.format(newHeigth), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255,0,255),2)
                
             
                
                
        cv2.imshow('found',img2)
        
    img = cv2.resize(img,(0,0),None,0.5,0.5)
    
    #cv2.imshow('original', img)
    k = cv2.waitKey(1) & 0xFF
    #END APP
    if k == ord('q'):
        break


