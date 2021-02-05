#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 07:53:55 2020

@author: felipe
"""

import face_recognition as fr
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
import sklearn 
import math

%matplotlib qt

def resizeImage(image, **kwargs):
    if 'scale' in kwargs.keys():
        scale = kwargs.get('scale')
        x = int(image.shape[0] * scale)
        y = int(image.shape[1] * scale)
    return cv2.resize(image, dsize=(x, y), interpolation=cv2.INTER_AREA)

def compare_images(im1, im2):
    plt.Figure(figsize=(10, 10))
    plt.subplot(121)
    plt.imshow(im1)
    plt.subplot(122)
    plt.imshow(im2)

def load_image(path, gray=True):
    im = cv2.imread(path)
    if gray: im = cv2.cv2tColor(im, cv2.COLOR_BGR2GRAY)
    return im

def flat_images(images):
    #Flatten the images array
    n, m = images[0].shape
    k = len(images)
    flat_images = np.zeros((n * m, k), dtype='uint8').T
    
    for i in range(k):
        flat_images[i] = images[i].flatten()
    return flat_images

def reshape_images(images, original):
    #original must be a tensor
    n, n, k = original
    return images.reshape((n, m, k))


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
    
def get_labels(y_pred, labels):
    dic_labels = {}
    index_pred = [i[0] for i in y_pred[1]] 
    for i, name in enumerate(labels):
        dic_labels.update({i:name})   
    both = set(index_pred).intersection(list(dic_labels.keys()))
    name_pred = [dic_labels[i] for i in both]
    return name_pred

#%%



# def imshow(im, cmap='gray'):
#     if len(im.shape) == 2 or (len(im.shape) == 3 and im.shape[2] == 1):
#         plt.imshow(im, cmap=cmap, norm=plt.Normalize(vmin=0, vmax=255))
#     else:
    #         plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

# obama = cv2.imread("images/obama.jpg")
# obama = cv2.cvtColor(obama, cv2.COLOR_BGR2RGB)

# blank = np.zeros(obama.shape, dtype='uint8')
# circle = cv2.circle(blank.copy(), (200, 200), 200, 255, -1)
# rectangle = cv2.rectangle(blank.copy(), (30, 30), (370, 370), 255, -1)

# #%%
# face_loc = fr.face_locations(obama, model="hog") #Return top, right, bottom, left
# rec_face_1 = (face_loc[0][3], face_loc[0][0])
# rec_face_2 = (face_loc[0][1], face_loc[0][2])

# # rec_fraud_1 = tuple([int(x * 1.2) for x in rec_face_1])
# # rec_fraud_2 = tuple([int(x * 1.2) for x in rec_face_2])

# rec_fraud_1 = (int(face_loc[0][3] * 0.75), int(face_loc[0][0] * 0.75))
# rec_fraud_2 = (int(face_loc[0][1] * 1.25), int(face_loc[0][2] * 1.25))

# cv2.rectangle(obama, rec_face_1, rec_face_2, (255 ,255, 255), 3)
# cv2.rectangle(obama, rec_fraud_1, rec_fraud_2, (255 ,255, 255), 4)
# plt.imshow(obama)
# #%%
# test = resizeImage(obama.copy(), (200, 200))


#%%
#Realizando leitura do dataset

train_dir = "/home/felipe/Documents/Alianca/my_tests/archive/lfw-deepfunneled/lfw-deepfunneled"

path = []
images = []
data = []
for sub_folder in os.listdir(train_dir):
    for filename in os.listdir(os.path.join(train_dir, sub_folder)):
        image_path = train_dir+'/'+sub_folder+'/'+filename
        data.append({"Name": sub_folder, 'Path': image_path})
        
df = pd.DataFrame(data)
print("kkk")

#%%
#Catch only users that had more than 10 images
df_aux = df.groupby(df['Name']).filter(lambda x: len(x) > 10)

#%%%
# Reading local of an image
for x in df_aux['Path']:
    im = cv2.imread(x)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    images.append(im)
    


print("Fim")

#%%
# Catch the face locations
face_locations = []
for im in images:
    face_locations.append(fr.face_locations(im, 2, 'hog'))
print("Fim face location")
#%%    
# Decode face locations and detect pics that had more than one face 

invalid_pics = []
for i, j in enumerate(face_locations):
    if(len(j) != 1):
        print(f"There are less or more than one person in image index: {i}")
        face_locations.pop(i)
        invalid_pics.append(i)
        images.pop(i) #removing from images
    # else:
    #     encoded = fr.face_encodings(images[i], known_face_locations=())    
print("Fim2")
#%%    
#Encoding faces
# encoded = []
# for x in images:
#     encoded.append(fr.face_encodings(x))

#%%
flat_images = flat_images(images)

#%%

#Making a KNN model to classify users after the face detection
X = flat_images.copy()
y = list(df_aux['Name'])
for i in invalid_pics: y.pop(i) #removing invalid index from y


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state=42)





#%%
from sklearn.neighbors import KNeighborsClassifier
neighbors = int(math.sqrt(len(X_train)))

model = KNeighborsClassifier(n_neighbors= neighbors, weights='distance', algorithm='auto', p=2, metric='minkowski', n_jobs = -1)
model.fit(X_train, y_train)

print("Fim treino knn")
#%%
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, init="k-means++", n_init= 8, max_iter=1000)
kmeans.fit(X_train, y_train)

#%%


#Predict
# new_user_image_path = "/home/felipe/Documents/Alianca/my_tests/archive/lfw-deepfunneled/lfw-deepfunneled"
# new_image = load_image(new_user_image_path)
# y_pred = kmeans.fit_predict(X_test)
y_pred_nb = model.kneighbors(X_test, n_neighbors=1)
#%%
# Get the labels of predictions
labels_pred = get_labels_pred(y_pred_nb, y)
plotSample(X_test.reshape(-1, 250, 250), labels_pred)
print("Fim print")


#%%
#Check matches
are_matches = [y_pred_nb[0][i] <= 0.6 for i in range(X_test.shape[0])]

#%%

#%%
closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]




