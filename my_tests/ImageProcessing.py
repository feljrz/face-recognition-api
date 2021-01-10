#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 10:36:30 2020

@author: felipe
"""

import os
import face_recognition.api as face_recognition
import multiprocessing
import itertools
import sys
import PIL.Image
import numpy as np
import cv2
import imutils
import requests
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd


def graph_size(n):
    """Função auxiliar para definir tamanho dos gráficos"""
    return (n*(1+5**0.5)/2, n)

# obama1= cv2.imread("../../examples/obama.jpg")
# obama2 = cv2.imread("../../examples/obama2.jpg")
#obamaG1 = cv2.cv2tColor(obama1, cv2.COLOR_BGR2GRAY)
#obamaG2 = cv2.cv2tColor(obama2, cv2.COLOR_BGR2GRAY)
folder = "images"

# url = "https://us.123rf.com/450wm/milkos/milkos1710/milkos171000380/87123751-happy-redhead-woman-surfing-the-web-and-talking-on-mobile-at-home.jpg?ver=6"
# response  = requests.get(url)
# img_req = PIL.Image.open(BytesIO(response.content))
# plt.imshow(img_req)
# face_recognition.load_image_file(img_req, mode='RGB')

# with open("req.jpg", 'wb') as f:
#     f.write(img_req)
    
    
def printImage(images, grid=False):
    for image in images:
        #image = cv2.cv2tColor(image, cv2.COLOR_BGR2GRAY)
        if grid: plt.grid(True)
        plt.imshow(image)
        
    
def resizeImage(image, **kwargs):
    if 'scale' in kwargs.keys():
        scale = kwargs.get('scale')
        x = int(image.shape[0] * scale)
        y = int(image.shape[1] * scale)
    return cv2.resize(image, dsize=(x, y), interpolation=cv2.INTER_AREA)
        
def loadFolderImages(folder, resize=True, verbose=True):
    images = {}
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        # img = face_recognition.load_image_file(filename)
        if img is not None:
            if verbose: print(type(img), img.shape)
            img_cp = img.copy()
            img_cp = resizeImage(img_cp, scale=0.1)
            img_cp = cv2.cvtColor(img_cp, cv2.COLOR_BGR2GRAY)
            # img = img.resize(img, (128, 128))
            images.update({filename: img_cp})
    return images

def plotSample(images, labels):
    size = len(images)
    plot_size = np.array(list((3,4))) * 4
    plt.figure(figsize=tuple(plot_size)) 
    for i in range(12):
        plt.subplot(3,4, i+1)
        plt.imshow(images[i])
        plt.yticks([])
        plt.xticks([])
        plt.xlabel(labels[i])
    plt.show()
    

im_dic = loadFolderImages(folder, verbose=False)
dic_images = list(im_dic.values())

# images_test = np.empty(dic_images.shape)

# for k in dic_images :
#     k = k[np.newaxis, :]
#     print(k.shape)
#     images_test.append(k)
#     plt.imshow(k)
    


#images = resize_image(image, minSize)
# images = np.concatenate(list(dic_images))
# plt.imshow(images_test[5])
# labels = list(dic_images.keys())
# plotSample(images, labels)
%
labels = list(im_dic.keys())
plotSample(dic_images, labels)
%
data = np.load("Dataset/face_images.npy")
