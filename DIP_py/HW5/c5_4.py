# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 17:12:16 2020

@author: LZY
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

def func(img,size,gray_th):
    #MORPH_ELLIPSE , MORPH_RECT
    k1=cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))
    k2=cv2.getStructuringElement(cv2.MORPH_RECT,(size+2,size+2))
    
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






img=cv2.imread("Chapter5_1.bmp")
#preprocessing
img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
img = cv2.GaussianBlur(img, (5, 5), 0.5)  
img=img/255
#parameter
maxsize=50
gray_th=40/255
value=[]
s=[]

for size in range(3,maxsize,1):
    s,value=func(img,size,gray_th)
    
    
plt.figure()
plt.bar(s,value,0.5)
    
    