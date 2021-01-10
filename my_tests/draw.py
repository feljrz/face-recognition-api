#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 21:04:37 2020

@author: felipe
"""

import cv2
import matplotlib.pyplot as plt 
import numpy as np


image = cv2.imread("images/obama.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.subplot(121)
plt.imshow(image)

#%%
blank = np.zeros(image.shape, dtype='uint8')
plt.subplot(122)

blank[200:450, 350:600] = 0, 255, 0 
blank[250: 400, 400:550] = 255, 0 ,0
new_img = blank[200:450, 350:600]
plt.subplot(121)
plt.imshow(blank)
plt.subplot(122)
plt.imshow(new_img)
#%%
#Draw with cv2
im = np.zeros((1080, 720), dtype='uint8')
centerX, centerY = im.shape[0]//2, im.shape[1]//2
cv2.rectangle(im, (centerX, centerY), (400,600), (255, 0 ,255), 2)
cv2.circle(im, (centerX, centerY), 100, (255, 255, 255), -1)
plt.imshow(im)
#%%
#Gaussian Filter
blur = cv2.GaussianBlur(image, (3,3), 0)
plt.imshow(blur)


#%%
#Edge
canny = cv2.Canny(blur, 100, 200)
plt.subplot(121)
plt.imshow(canny)
plt.subplot(122)
plt.imshow(cv2.Canny(image, 100, 200))
#%%
#Dilating
dilated = cv2.dilate(canny, (3, 3), iterations=2)
plt.imshow(dilated)

#%%
#Eroding
eroded = cv2.erode(dilated, (3, 3), iterations=2)
plt.imshow(eroded)
#%%
#Resize
resized = cv2.resize(canny, (1200, 900), interpolation=cv2.INTER_CUBIC)
plt.imshow(resized)
#%%
#Crop
crop_image = canny[0:400, 400:600]
plt.imshow(crop_image)
#%%
#Utilizando a binarização da imagem também é possível encontrar os contornor porém é menos eficiente
#Lembre-se que: (pixel > thresh = 255) e (pixel < thresh = 0)
ret, thresh = cv2.threshold(image, thresh=125, maxval=255, type=cv2.THRESH_BINARY)
plt.imshow(thresh)

#%%
# findCountours
# cv2.RETR_LIST - lista com todos os contornos, TREE - cortornos em hierarquia, EXTERNAL - contornos externos
# cv2.CHAIN_APPROX_NONE - não faz aproximação entre os pontos retornoando todos, SIMPLE retorna 2 pontos que definem a reta
# drawContours - recebe a imagem, lista dos contornos, indices (-1 todos), cor

contours, hierarchies = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
print(f"{len(contours)} found")

cv2.drawContours(blank, contours[300], -1, (255, 255, 255))
plt.subplot(121)
plt.imshow(blank)


#%%















