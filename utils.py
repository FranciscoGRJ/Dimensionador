import cv2
import numpy as np

def getContours(img,cannyThreshold=[100,100],showCanny=False, minArea = 1000, filter = 0, draw =False):
    imgGrayScale = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlurr = cv2.GaussianBlur(imgGrayScale,(5,5),1)
    imgCanny = cv2.Canny(imgBlurr,cannyThreshold[0],cannyThreshold[1])
    kernel =np.ones((5,5))
    imgDialation =cv2.dilate(imgCanny,kernel,iterations=3)
    imgThreshold= cv2.erode(imgDialation,kernel,iterations=2)
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

def reorder (myPoints):
    print(myPoints.shape)
    myNewPoints=np.zeros_like(myPoints)
    myPoints=myPoints.reshape((4,2))
    add= myPoints.sum(1)
    myNewPoints[0] = myPoints[np.argmin(add)]
    myNewPoints[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis = 1)
    myNewPoints[1] =myPoints[np.argmin(diff)]
    myNewPoints[2] =myPoints[np.argmax(diff)]
    return myNewPoints

def warpImage (img,points,width,heigth,pad = 20):
    #print(points)
    points = reorder(points)
    point1 =np.float32(points)
    point2 = np.float32([[0,0],[width,0],[0,heigth],[width,heigth]])
    matrix =cv2.getPerspectiveTransform(point1,point2)
    imageWrapped = cv2.warpPerspective(img,matrix,[width,heigth])
    
    imageWrapped = imageWrapped[pad:imageWrapped.shape[0]-pad, pad:imageWrapped.shape[1]-pad]
    
    return imageWrapped

def findDis (point1, point2):
    return ((point2[0]-point1[0])**2 + (point2[1]-point1[1])**2)**0.5
