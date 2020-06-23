# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 10:18:31 2020

@author: LZY
"""

#demo of threshold gray by adjusting trackbar

import cv2
import numpy as np
img=cv2.imread("MRA.pgm",0)


img_w=np.size(img,0)
img_h=np.size(img,1)
cv2.namedWindow('image', 0)
cv2.resizeWindow('image', 720, 480)


cv2.createTrackbar('gray','image',0,255,lambda x: None)

while(True):
    num = cv2.getTrackbarPos('gray','image')
    ret, thresh1 = cv2.threshold(img, num, 255, cv2.THRESH_BINARY)
    cv2.imshow('image', thresh1)
    if cv2.waitKey(1) == 27:
        break


cv2.destroyAllWindows()
