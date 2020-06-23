# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 11:29:09 2020

@author: LZY
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np



img=cv2.imread('MRA.pgm',0)



def func1(img):   
    #选取ROI区域，对ROI区域做otsu，把得到的阈值th2在原图中做otsu
    ROI_img=img[10:60,30:110]
    th1, binary_roi = cv2.threshold(ROI_img, 0, 255,  cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th2,otsu_binary=cv2.threshold(img, th1, 255,  cv2.THRESH_BINARY)
    return th2,otsu_binary,ROI_img,binary_roi



def OTSU(img_gray):
    
    max_g = 0
    suitable_th = 0
    th_begin = 28
    th_end = 50
    
    
    for threshold in range(th_begin, th_end):
        bin_img = img_gray > threshold
        bin_img_inv = img_gray <= threshold
        fore_pix = np.sum(bin_img)
        back_pix = np.sum(bin_img_inv)
        if 0 == fore_pix:
            break
        if 0 == back_pix:
            continue
 
        w0 = float(fore_pix) / img_gray.size
        u0 = float(np.sum(img_gray * bin_img)) / fore_pix
        w1 = float(back_pix) / img_gray.size
        u1 = float(np.sum(img_gray * bin_img_inv)) / back_pix
        
        # intra-class variance
        g = w0 * w1 * (u0 - u1) * (u0 - u1)
        if g > max_g:
            max_g = g
            suitable_th = threshold
    
    
    
    img_w=np.size(img_gray,0)
    img_h=np.size(img_gray,1)
    otsu_binary=np.zeros((img_w,img_h))    
    for i in range(img_w):
       for j in range(img_h):
           if img_gray[i,j]>suitable_th:
               otsu_binary[i,j]=255
           else:
               otsu_binary[i,j]=0    
        
    return suitable_th,otsu_binary


def gray_point(img):
    points=0
    a=plt.hist(img.ravel(),256,[0,256])[0]
    for i in range(255,1,-1):
        if points<num_fore:
            points=points+a[i]
        else:
            bg_gray=i
            break
    return bg_gray
#传统Otsu
th,otsu_binary=cv2.threshold(img, 0, 255,  cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#rect ROI
th1,otsu_binary,ROI_img,binary_roi=func1(img)
plt.figure(2)
plt.subplot(221),plt.imshow(img,cmap='gray'),plt.title('src img')
plt.subplot(222),plt.imshow(ROI_img,'gray'),plt.title('rect ROI')
plt.subplot(223),plt.imshow(binary_roi,'gray'),plt.title('ROI binary')
plt.subplot(224),plt.imshow(otsu_binary,'gray'),plt.title('RCOtsu 2')




#ROI
_, temp = cv2.threshold(img, 20, 255,  cv2.THRESH_BINARY)
k=cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
mask=cv2.dilate(temp,k)/255
ROI_img1=mask*img
ROI_img1=ROI_img1.astype(np.uint8)

num_mask=np.sum(mask)
num_fore=0.05*num_mask
bg_ratio=360/num_mask


#my otsu
my_th,my_otsu=OTSU(img)
roi1_th,roi_rcotsu=OTSU(img)

#bg_gray=gray_point(roi_rcotsu)

    
    
#imshow
plt.figure(1)
plt.subplot(221),plt.imshow(img,'gray'),plt.title('src img')
plt.subplot(222),plt.imshow(mask,'gray'),plt.title('mask')
plt.subplot(223),plt.imshow(roi_rcotsu,'gray'),plt.title('RCOtsu 2')
plt.subplot(224),plt.hist(img.ravel(),256,[0,256]),plt.title('hist')


#img_w=np.size(img,0)
#img_h=np.size(img,1)
#cv2.namedWindow('img', 0)
#cv2.resizeWindow('img', 2*img_w, 2*img_w)
#cv2.imshow('img',img)
