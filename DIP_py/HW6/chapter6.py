# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 11:29:09 2020

@author: LZY
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np


img=cv2.imread("MRA.pgm")



#gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)


plt.figure()
plt.subplot(231)
plt.imshow(img)
plt.subplot(232)
plt.hist(img.ravel(),256,[0,256])







#plt.figure()
#plt.imshow(gray,cmap='gray')