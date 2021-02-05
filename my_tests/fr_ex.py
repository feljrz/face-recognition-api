#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 07:53:55 2020
@author: felipe
"""
#https://stackoverflow.com/questions/30988033/sending-live-video-frame-over-network-in-python-opencv

import face_recognition as fr
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
import sklearn
import math
import pickle #Estudar sobre
from flask import Flask, render_template, Response


#%%

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
    if gray: im = cv2.cv2tColor(im, cv2.COLOR_BGR2RGB)
    return im

def flat_images(images):
    #Flatten the images array
    n, m = images[0].shape
    k = len(images)
    flat_images = np.zeros((n * m, k), dtype='uint8').T
   
    for i in range(k):
        flat_images[i] = images[i].flatten()
    return flat_images


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
   
#Lmebrar que esta recebe a predição com distância = True
#Errado
def get_labels(labels, knn_pred):
   names = []
   index_pred = [i for i in knn_pred[1][0]]
   for i in index_pred:
       names.append(labels[i])
   return names

def read_dir(directory, for_one=False, retrieve_one_image=False):
    path = []
    images = []
    data = []
    
    if not(for_one):
        for sub_folder in os.listdir(directory):
            for filename in os.listdir(os.path.join(directory, sub_folder)):
                image_path = directory+'/'+sub_folder+'/'+filename
                data.append({"Name": sub_folder, 'Path': image_path})
                if retrieve_one_image:
                    break
    else:
        for image in os.listdir(directory):
            image_path = os.path.join(directory,image)
            data.append({"Name": directory.split('/').pop(), "Path":image_path})            
     
    df = pd.DataFrame(data)
    return df

def read_image(path):
    images = []
    for x in path:
        im = cv2.imread(x)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        images.append(im)
    return images


def knn_train(X_train, y_train, model_save_path=None, threads=-1):
    from sklearn.neighbors import KNeighborsClassifier
    n_neighbors = int(math.sqrt(len(X_train))) #Sera susbtituído pelo elbow method
    
    model = KNeighborsClassifier(n_neighbors= n_neighbors, weights='distance', algorithm='auto', p=2, metric='minkowski', n_jobs = threads)
    model.fit(X_train, y_train)
    if model_save_path:
        save_binary(model, model_save_path)
    
    return model

def knn_predict_(image, model_path=None, verbose=False):
    if model_path:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
   
    y_pred = model.kneighbors(image, n_neighbors=1)
   
    return y_pred

def save_binary(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_binary(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def unpaking_array(*args):
    new_list = list
    
    
def first_train(df):
    df["Face Location"] = df["Image"].apply(lambda x: fr.face_locations(x, 2, model="hog"))
    df["Face Encoding"] = df.apply(lambda x: fr.face_encodings(x["Image"], x["Face Location"]), axis=1)
    model = knn_train([x[0] for x in df['Face Encoding']], df['Name'].values, model_save_path)
    
    return model

    


# Realizando leitura do dataset
model_save_path = "./knn_model.clf"
train_dir= "archive/lfw-deepfunneled"

#%%
df = read_dir(train_dir, retrieve_one_image=True)
df['Image'] = read_image(df['Path'])
print("Fim Leitura")

#%%
# Catch only users that had more than 10 images
# df = df.groupby(df['Name']).filter(lambda x: len(x) > 10)
df = df.iloc[:1500, :]

#%%
#Testando map do pandas apply

df = first_train(df)



#%%
# Catch the face locations
# We can increase number of itterations to increse precision
# face_locations = []
# face_encoding = []
# for im in df['Image'].values:
#     fl = fr.face_locations(im, 2, 'hog')
#     face_encoding.append(fr.face_encodings(im, fl))
#     face_locations.append(fl)
# print("Fim face location")

# df['Face Location'] = face_locations
# df['Face Encoding'] = face_encoding
#%%

# SOMENTE EM TESTE
# Detect and remove images that had more than one face

# df = df[df.apply(lambda x: len(x['Face Location']) == 1, axis=1)]
# df.to_csv('1000_images_1_face.csv') # Substituir pelo pickle para manter o tipo de dados
save_binary(df, 'df_with_me')

#%%
#Reading Dataframe
#For tests you can start here
df = load_binary("bkp/df_with_me")
# model = load_binary(model_save_path)


#%%
# Encoding images flatting them
# flat_images = flat_images(df['Images']))
# df['Images'] = list(flat_images)

#%%
# Separate data in train and test
# LEMBRE-SE QUE O TREINO REAL DEVERÁ CONTER TODOS OS USUÁRIOS

# X = .copy()
# y = list(df['Name'].copy())

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split([x[0] for x in df['Face Encoding'].values], list(df['Name'].values), test_size= 0.25, random_state=42)

############## UTILIZAR ESTE ########################

# from sklearn.model_selection import train_test_split
# df_train, df_test = train_test_split(df, test_size= 0.25, random_state=42)


#%%

#Train
# from sklearn.neighbors import KNeighborsClassifier
# #Realizar Elbow method

# n_neighbors = int(math.sqrt(len(X_train)))
# model = KNeighborsClassifier(n_neighbors= n_neighbors, weights='distance', algorithm='auto', p=2, metric='minkowski', n_jobs = -1)
# model.fit(X_train, y_train)

# save_model(model, model_save_path)
# Lembre que df_train será substituído por df

model = knn_train([x[0] for x in np.array(df['Face Encoding'])], df['Name'].values, model_save_path)

#%%
#Predict
app = Flask(__name__)
cap = cv2.VideoCapture('../../../Video/faces.mp4')
while(cap.isOpened()):
    ret, frame = cap.read()
    # ukn_face_loc = fr.face_locations(frame) #return top right bottom left
    ukn_face_encode = fr.face_encodings(frame)
    
    # person_pred = model.predict(np.array(ukn_face_encode).reshape(-1, 1))
    
    # cv2.rectangle(frame, (ukn_face_loc[0], ukn_face_loc[3]),
    #               (ukn_face_loc[2], ukn_face_loc[1]), (0, 255, 0), 2)
    
    # frame = cv2.putText(frame, person_pred, ((ukn_face_loc[2]/2 + 10), (ukn_face_loc[2]/2 + 10)) , cv2.FONT_HERSHEY_SIMPLEX 
    #                     , 1, (0, 255, 0), 2, cv2.LINE_AA)
        
    cv2.imshow('Video Frame', frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()


#%%
#Send to Browser


temp = []
for pos, x in enumerate(df["Face Encoding"]):
    if len(x) > 1:
        for j in x:
            temp.append(j)
    else:
        temp.append(x)
    
    
    
  





    
#%%
#Teste on Machine

# webcam = cv2.VideoCapture(0)
# while True:
#     ret, frame = webcam.read()
#     if ret == True:
#         small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        
        
#         cv2.imshow("Frame", small_frame)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break

# webcam.release()
# cv2.destroyAllWindows()







#%%


#Pode ser substituido por cnn// Realizar encoding do frame
unk_face_frame = np.array(df['Face Encoding'].iloc[200]).reshape(1, -1)
closest_neighbors = model.kneighbors(unk_face_frame, n_neighbors=5)

y_pred = model.predict(unk_face_frame)


#%%
candidate_names = get_labels(df['Name'].values,  closest_neighbors)
#Select distinct names
candidate_names = list(set(candidate_names)) 

df_candidates = df[df['Name'].isin(candidate_names)] # Lembre que isto só é possível pois Boolean é filho uma extensão do tipo int

#%%
#Check the best match for candidates

face_distances = fr.face_distance(np.array([x[0] for x in df_candidates['Face Encoding'].values]), unk_face_frame)
inconsistent_image_count = sum(x > 0.6 for x in face_distances)



#%%
#Segunda verificação
#Selecionando o index no df total

if(inconsistent_image_count < 2):
    best_match_index = np.argmin(face_distances)
    candidate_index = df_candidates.index[best_match_index]
    person_df = df[df.index == candidate_index]
    name = ' '.join(person_df['Name'].iloc[0].split('_'))
    print(f'Bem vindo {name}')

else:
    print("Você ainda não está cadastrado")
    


#%%




