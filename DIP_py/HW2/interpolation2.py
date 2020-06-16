import numpy as np
import matplotlib.pyplot as plt

import cv2
from skimage.measure import compare_ssim
from skimage.measure import compare_mse
from skimage.measure import compare_psnr


def get_mean_std(img):
    h,w=img.shape[:2]
    img_mean=np.mean(img[:,:,0])
    img_std=np.std(img[:,:,0])
    return img_mean,img_std



img=cv2.imread('Chapter2_1.pgm')
height, width = img.shape[:2]

#extend border 
scale_img=cv2.copyMakeBorder(img,height//2,height//2,width//2,width//2,cv2.BORDER_CONSTANT,value=[0,0,0])
scale_h,scale_w=scale_img.shape[:2]
Rotation_degree=-15.0
center=(scale_w//2,scale_h//2)

#Rotation Matrix
Rotate_M=cv2.getRotationMatrix2D(center,Rotation_degree,scale=1)
Rotate_back=cv2.getRotationMatrix2D(center,-Rotation_degree,scale=1)


Nearest_img=cv2.warpAffine(scale_img,Rotate_M,(scale_h,scale_w),flags=cv2.INTER_NEAREST)
Bilinear_img=cv2.warpAffine(scale_img,Rotate_M,(scale_h,scale_w),flags=cv2.INTER_LINEAR)
Cubic_img=cv2.warpAffine(scale_img,Rotate_M,(scale_h,scale_w),flags=cv2.INTER_CUBIC)


#rotate back by using the matrix of Rotae_back
Nearest_back=cv2.warpAffine(Nearest_img,Rotate_back,(scale_h,scale_w),flags=cv2.INTER_NEAREST)
Bilinear_back=cv2.warpAffine(Bilinear_img,Rotate_back,(scale_h,scale_w),flags=cv2.INTER_LINEAR)
Cubic_back=cv2.warpAffine(Cubic_img,Rotate_back,(scale_h,scale_w),flags=cv2.INTER_CUBIC)

Nearest_back=Nearest_back[height//2:height//2+height,width//2:width//2+width]
Bilinear_back=Bilinear_back[height//2:height//2+height,width//2:width//2+width]
Cubic_back=Cubic_back[height//2:height//2+height,width//2:width//2+width]



#mse
nearest_mse=compare_mse(img,Nearest_back)
bilinear_mse=compare_mse(img,Bilinear_back)
cubic_mse=compare_mse(img,Cubic_back)

#ssim
nearest_ssim=compare_ssim(img,Nearest_back,multichannel=True)
bilinear_ssim=compare_ssim(img,Bilinear_back,multichannel=True)
cubic_ssim=compare_ssim(img,Cubic_back,multichannel=True)

#psnr
nearest_psnr=compare_psnr(img,Nearest_back)
bilinear_psnr=compare_psnr(img,Bilinear_back)
cubic_psnr=compare_psnr(img,Cubic_back)
print('Nearest:','MSE=',nearest_mse,'SSIM=',nearest_ssim,'PSNR=',nearest_psnr)
print('Bilinear:','MSE=',bilinear_mse,'SSIM=',bilinear_ssim,'PSNR=',bilinear_psnr)
print('Cubic:','MSE=',cubic_mse,'SSIM=',cubic_ssim,'PSNR=',cubic_psnr)


##旋转后图像
plt.figure()
plt.subplot(2,2,1)
plt.title('Source img')
plt.imshow(img,cmap='gray')
plt.subplot( 2, 2, 2 )
plt.title( 'Nearest_img' )
plt.imshow( Nearest_img )
plt.subplot( 2, 2, 3 )
plt.title( 'Bilinear_img' )
plt.imshow( Bilinear_img )
plt.subplot( 2, 2,4 )
plt.title( 'Cubic_img' )
plt.imshow( Cubic_img )
#
#
##复原后图像
plt.figure()
plt.subplot(2,2,1)
plt.title('Source img')
plt.imshow(img)
plt.subplot( 2, 2, 2 )
plt.title( 'Nearest_back' )
plt.imshow( Nearest_back )
plt.subplot( 2, 2, 3 )
plt.title( 'Bilinear_back' )
plt.imshow( Bilinear_back )
plt.subplot( 2, 2,4 )
plt.title( 'Cubic_back' )
plt.imshow( Cubic_back )
#
#
##灰度直方图
#plt.figure()
#plt.subplot(2,2,1)
#plt.title('Source img')
#plt.hist(img.ravel(), 256, [0, 256])
#plt.show()
#plt.subplot( 2, 2, 2 )
#plt.title( 'Nearest_back' )
#plt.hist(Nearest_back.ravel(), 256, [0, 256])
#plt.show()
#plt.subplot( 2, 2, 3 )
#plt.title( 'Bilinear_back' )
#plt.hist(Bilinear_back.ravel(), 256, [0, 256])
#plt.show()
#plt.subplot( 2, 2,4 )
#plt.title( 'Cubic_back' )
#plt.hist(Cubic_back.ravel(), 256, [0, 256])
#plt.show()
#
#
###傅里叶变换
#f1 = np.fft.fft2(img[:,:,0])
#f2 = np.fft.fft2(Nearest_back[:,:,0])
#f3 = np.fft.fft2(Bilinear_back[:,:,0])
#f4 = np.fft.fft2(Cubic_back[:,:,0])
#
#fshift1 = np.fft.fftshift(f1)       
#fshift2 = np.fft.fftshift(f2)  
#fshift3 = np.fft.fftshift(f3)  
#fshift4 = np.fft.fftshift(f4)  
#
#fimg1 = np.log(np.abs(fshift1))
#fimg2 = np.log(np.abs(fshift2))
#fimg3 = np.log(np.abs(fshift3))
#fimg4 = np.log(np.abs(fshift4))
#
#plt.figure()
#plt.subplot(2,2,1)
#plt.title('Source img')
#plt.imshow(fimg1,cmap='gray')
#plt.subplot( 2, 2, 2 )
#plt.title( 'Nearest_back' )
#plt.imshow(fimg2,cmap='gray')
#plt.subplot( 2, 2, 3 )
#plt.title( 'Bilinear_back' )
#plt.imshow(fimg3,cmap='gray')
#plt.subplot( 2, 2,4 )
#plt.title( 'Cubic_back' )
#plt.imshow(fimg4,cmap='gray')
