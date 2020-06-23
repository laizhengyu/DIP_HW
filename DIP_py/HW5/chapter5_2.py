# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 23:15:21 2020

@author: LZY
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np


def func(img,size,gray_th):
    #MORPH_ELLIPSE , MORPH_RECT
    k1=cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))
    k2=cv2.getStructuringElement(cv2.MORPH_RECT,(size+deltasize,size+deltasize))
    
    #MORPH_TOPHAT
    imgOpen_1 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel=k1)
    imgOpen_2 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel=k2)

    diffimg=imgOpen_2-imgOpen_1
    _,diffimg=cv2.threshold(diffimg,gray_th,255,cv2.THRESH_BINARY)   
    
    count=np.sum(diffimg)/255

    diffValue=count/((size+1)*(size+1))
#    print(diffValue)
    value.append(diffValue)
    s.append(size)
    return s,value
    
    
img=cv2.imread("Chapter5_2.bmp")
img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
img = cv2.GaussianBlur(img, (5, 5), 0.5)  
#------------------------------------------------
#_,img=cv2.threshold(img,40,255,type=cv2.THRESH_BINARY)


#size=5
maxsize=50
deltasize=1
gray_th=40
value=[]
s=[]
for size in range(5,maxsize,1):
    
    if size <20:
        deltasize=1
        gray_th=40
    else:
        deltasize=1
        gray_th=20
        
#    if size <20:
#        deltasize=1
#        gray_th=20
#    else:
#        deltasize=2
#        gray_th=20  
        
        
#MORPH_ELLIPSE , MORPH_RECT
    k1=cv2.getStructuringElement(cv2.MORPH_RECT,(size-deltasize,size-deltasize))
    k2=cv2.getStructuringElement(cv2.MORPH_RECT,(size+deltasize,size+deltasize))
    
    #MORPH_ELLIPSE
#    _,thres_img = cv2.threshold(img,80,255,cv2.THRESH_BINARY)
#    imgOpen_1 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel=k1)
###    _,th1=cv2.threshold(imgOpen_1,20,255,cv2.THRESH_BINARY)
##    
#    imgOpen_2 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel=k2)
###    _,th2=cv2.threshold(imgOpen_2,20,255,cv2.THRESH_BINARY)
##    
#    diffimg=imgOpen_2-imgOpen_1

#    area=np.sum(diffimg)
    
    #MORPH_TOPHAT
    imgOpen_1 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel=k1)
    #_,th1=cv2.threshold(imgOpen_1,40,255,cv2.THRESH_BINARY)
    
    imgOpen_2 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel=k2)
    #_,th2=cv2.threshold(imgOpen_2,40,255,cv2.THRESH_BINARY)
    
    #diffimg=th2-th1
    
    diffimg=imgOpen_2-imgOpen_1
    _,diffimg=cv2.threshold(diffimg,gray_th,255,cv2.THRESH_BINARY)   
    



#    #th=60
#    imgOpen_1 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel=k1)
#    _,th1=cv2.threshold(imgOpen_1,70,255,cv2.THRESH_BINARY)
#    
#    imgOpen_2 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel=k2)
#    _,th2=cv2.threshold(imgOpen_2,70,255,cv2.THRESH_BINARY)
#    
#    diffimg=th2-th1
    
    
    count=np.sum(diffimg)/255
    
    
    
#    count=0
#    for i in range(img.shape[0]):
#        for j in range (img.shape[1]):
#            if img[i,j]>128:
#                count=count+1
    
    
    diffValue=count/((size)*(size))
#    diffValue=count/(3.14*(size*size)/4)
#    
    #_,thres_img = cv2.threshold(diffimg,20,255,cv2.THRESH_BINARY)
    
#    diffValue =( cv2.sumElems(diffimg)[0] / 255 )/size

    
#    print(diffValue)
    value.append(diffValue)
    s.append(size)
'''
--------------------------------------------------------------
'''
#plt.figure()
#plt.subplot(231)
#plt.title("srcimg")
#plt.imshow(img,cmap='gray')
#plt.subplot(232)
#plt.title("open1")
#plt.imshow(imgOpen_1,cmap="gray")
#plt.subplot(233)
#plt.title("open2")
#plt.imshow(imgOpen_2,cmap="gray")
#plt.subplot(234)
#plt.title("diff")
#plt.imshow(diffimg,cmap="gray")
#plt.subplot(235)
#plt.title("th")
#plt.imshow(thres_img,cmap="gray")

plt.figure()
plt.bar(s,value,0.5)