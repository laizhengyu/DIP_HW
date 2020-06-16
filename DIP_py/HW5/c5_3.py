# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 17:17:37 2020

@author: LZY
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

img=cv2.imread("Chapter5_2.bmp")
#mask1=cv2.imread('mask1.bmp')
#mask2=cv2.imread('mask2.bmp')
#
#mask=(mask1+mask2)

#img=np.clip(img-mask,0,255)
img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
img = cv2.GaussianBlur(img, (5, 5), 0.5)  
gray_th=20
size=7
delsize=1
k1=cv2.getStructuringElement(cv2.MORPH_RECT,(size-delsize,size-delsize))
k2=cv2.getStructuringElement(cv2.MORPH_RECT,(size+delsize,size+delsize))

imgOpen_1 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel=k1)
#ero_img1=cv2.erode(img,k1)
#_,th1=cv2.threshold(imgOpen_1,40,255,cv2.THRESH_BINARY)

imgOpen_2 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel=k2)
#ero_img2=cv2.erode(img,k2)
#_,th2=cv2.threshold(imgOpen_2,40,255,cv2.THRESH_BINARY)

#diffimg=th2-th1

diffimg=imgOpen_2-imgOpen_1
#diffimg=ero_img2-ero_img1
_,diffimg=cv2.threshold(diffimg,gray_th,255,cv2.THRESH_BINARY)

count=np.sum(diffimg)/255
diffValue=count/(size*size)
print('gray_th=',gray_th,'size=',size)
print('diffvalue=',diffValue)

plt.figure()
plt.subplot(231)
plt.title("srcimg")
plt.imshow(img,cmap='gray')
plt.subplot(232)
plt.title("open1")
plt.imshow(imgOpen_1,cmap="gray")
plt.subplot(233)
plt.title("open2")
plt.imshow(imgOpen_2,cmap="gray")
plt.subplot(234)
plt.title("diff")
plt.imshow(diffimg,cmap="gray")
#plt.subplot(235)
#plt.title("erosion")
#plt.imshow(ero_img1,cmap="gray")