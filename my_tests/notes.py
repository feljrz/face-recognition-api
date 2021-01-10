#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 20:33:07 2020

@author: felipe
"""


# def imshow(im, cmap='gray'):
#     if len(im.shape) == 2 or (len(im.shape) == 3 and im.shape[2] == 1):
#         plt.imshow(im, cmap=cmap, norm=plt.Normalize(vmin=0, vmax=255))
#     else:
#         plt.imshow(cv2.cv2tColor(im, cv2.COLOR_BGR2RGB))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils

def load_image(path, gray=True):
    im = cv2.imread(path)
    if gray: im = cv2.cv2tColor(im, cv2.COLOR_BGR2GRAY)
    return im
    

def resizeImage(image, scale=0.75):
    x = int(image.shape[0] * scale)
    y = int(image.shape[1] * scale)
    return cv2.resize(image, dsize=(x, y), interpolation=cv2.INTER_AREA)
    # return cv2.resize(image, (x,y), interpolation=cv2.INTER_AREA)

def imshow(im, grid=False, cmap="gray"):
    plt.imshow(im, cmap=cmap, norm=plt.Normalize(vmin=0, vmax=255))
    
    
def compare_images(im1, im2):
    plt.Figure(figsize=(10, 10))
    plt.subplot(121)
    imshow(im1)
    plt.subplot(122)
    imshow(im2)

def gauss_filter(image):
    return cv2.GaussianBlur(image.copy(), (11, 11), 0)
    
def threshold_binary(image, minval, maxval=255):
    return cv2.threshold(image.copy(), minval, maxval, cv2.THRESH_BINARY)[1]

def threshold_adaptive(image, maxval, method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C):
    return cv2.adaptiveThreshold(src=image.copy(), maxValue=maxval, adaptiveMethod=method, thresholdType=cv2.THRESH_BINARY, blockSize=11, C=2)




#%%
#%%

#Opencv2 notes

#Leitura

image = cv2.imread("images/obama.jpg")
obama = cv2.imread("images/obama.jpg")

fraude = load_image("images/fraude_1.jpeg")
np_image = np.array(image)
image= cv2.cv2tColor(image, cv2.COLOR_BGR2GRAY)
# plt.imshow(image)
#image = cv2.cv2tColor(image,cv2.COLOR_BGR2RGB)
# plt.imshow(image, cmap=cmap, norm=plt.Normalize(vmin=0, vmax=255))
# cv2.waitKey(0)
# cv2.imshow('Image',image)
#%%
#%%
#Resize
new_image = resizeImage(image, 0.5)
# plt.subplot(121)
# imshow(image, cmap="gray")
# plt.subplot(122)
# imshow(new_image, cmap="gray")

image = new_image.copy()

#%%
#%%
#Gaussian Filter
blur_image = cv2.GaussianBlur(image, (11, 11), 0)
# imshow(blur_image)
#%%
#Thresh
thresh = image.copy()
thresh = cv2.threshold(thresh, 50, 255, cv2.THRESH_BINARY)[1] #Basico
# compare_images(image, thresh)
thresh2 = image.copy()
thresh2 = cv2.adaptiveThreshold(src=thresh2, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=11, C=2)
# compare_images(thresh, thresh2)
#%%
#%%
#Contour
fraude_blur = gauss_filter(fraude)
# fraude_thresh = threshold_adaptive(fraude_blur, 255) MUDADO
fraude_thresh = threshold_binary(fraude_blur, 170, 255)
# imshow(bkk)

#%%
#%%
# contours 2 
cnts = cv2.findContours(fraude_thresh.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
grab_cnt = imutils.grab_contours(cnts) #cv2.findContours(fraude_thresh.copy(), v.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

cp_fraude = fraude_thresh.copy()
for x in cnts[0]:
    cv2.drawContours(cp_fraude, [x], -1, (0, 0, 255), thickness=8) 
    
compare_images(fraude_thresh, cp_fraude)
#%%
#%%
compare_images(threshold_adaptive(cp_fraude, 255), fraude_thresh)
imshow(threshold_adaptive(cp_fraude, 255))

#%%
#%%
#Estudo dos contornos
blankimage = np.ones((fraude.shape[0], fraude.shape[1], 3), np.uint8) * 255
for x in cnts[0]:
    cv2.drawContours(blankimage, [x], -1, (0, 0, 255), thickness=8) 

imshow(blankimage)
compare_images(fraude_thresh, blankimage)
#%%
#BITWISE

blank = np.zeros(obama.shape, dtype='uint8')
circle = cv2.circle(blank.copy(), (200, 200), 200, 255, -1)
rectangle = cv2.rectangle(blank.copy(), (30, 30), (370, 370), 255, -1)
plt.subplot(121)
plt.imshow(rectangle, norm=plt.Normalize(vmin=0, vmax=255))
plt.subplot(122)
plt.imshow(circle)
plt.show()

#Bitwise AND
bitwise_and = cv2.bitwise_and(circle, rectangle)
plt.imshow(bitwise_and)

#Bitwise OR
bitwise_or = cv2.bitwise_or(circle, rectangle)
plt.imshow(bitwise_or)

#Bitwise XOR
bitwise_xor = cv2.bitwise_xor(circle, rectangle)
plt.imshow(bitwise_xor)

#Bitwise NOT
bitwise_not = cv2.bitwise_not(circle, rectangle)
plt.imshow(bitwise_not)


#%%
#Mask 
masked = cv2.bitwise_and(obama, obama, mask=circle)
plt.imshow(masked)
#%%



