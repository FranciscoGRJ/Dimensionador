import cv2
import numpy as np

def getContours(img,cannyThreshold=[100,100],showCanny=False, minArea = 1000, filter = 0, draw =False):
    imgGrayScale = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlurr = cv2.GaussianBlur(imgGrayScale,(5,5),1)
    imgCanny = cv2.Canny(imgBlurr,cannyThreshold[0],cannyThreshold[1])
    kernel =np.ones((5,5))
    imgDialation =cv2.dilate(imgCanny,kernel,iterations=2)
    imgThreshold= cv2.erode(imgDialation,kernel,iterations=1)
    if showCanny:
        cv2.imshow('canny',imgThreshold)

    contours,hiearchy = cv2.findContours(imgThreshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    finalContours = []
    
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            perimiter =cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,0.02*perimiter,True)
            boundingBox = cv2.boundingRect(approx)
            
            if filter > 0:
                if len(approx) == filter:
                    finalContours.append([len(approx),area,approx,boundingBox,i])
            else:
                finalContours.append([len(approx),area,approx,boundingBox,i])
    
    finalContours = sorted(finalContours,key=lambda x:x[1], reverse=True)
    if draw:
        for con in finalContours:
            cv2.drawContours(img,con[4],-1,(0,0,255),3)
    
    return img, finalContours
