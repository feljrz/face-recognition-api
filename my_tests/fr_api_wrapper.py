#Testando o o que será colocado em threads_cam.py


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
    flaten_images = np.zeros((n * m, k), dtype='uint8').T
   
    for i in range(k):
        flaten_images[i] = images[i].flatten()
    return flaten_images


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

def get_label(frame, candidates, tolerance):
    global df
    name = None

    candidates_subset = df.iloc[candidates[0]]
    distances = []
    for encode in candidates_subset['Face Encoding']:
        distances.append(np.linalg.norm(frame - encode[0]))

    count = len([dst for dst in distances if dst > tolerance])

    if count > 2:
        name = "Unknow"

    # MUST BE FIXED
    # else:
        # index = np.argmin(distances)
        # name = candidates_subset.iloc[index]['Name']

    return [name], count

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
    print("End of training")
    
    return model

def knn_predict(image, model_path=None, verbose=False):
    if model_path:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
   
    y_pred = model.kneighbors(image, n_neighbors=1)
   
    return y_pred

def save_binary(pkl_file, path):
    with open(path, 'wb') as f:
        pickle.dump(pkl_file, f)

def load_binary(path):
    with open(path, 'rb') as f:
        pkl_file = pickle.load(f)
    return pkl_file

  
def first_train(train_dir, model_save_path=None, df_save_path=None):
    df = read_dir(train_dir, retrieve_one_image=False)
    df = df.sort_values(["Name"]) #@@@ Here because of the Dataset @@@
    df = df.iloc[:4500, :] #@@@ Here because of the Dataset @@@
    df['Image'] = read_image(df['Path'])
    print(df.tail(15))
    print("End reading")

    print("Start Location")
    df["Face Location"] = df["Image"].apply(lambda x: fr.face_locations(x, 2, model="hog"))
    print("End Location")
    
    print("Start Encoding")
    df["Face Encoding"] = df.apply(lambda x: fr.face_encodings(x["Image"], x["Face Location"]), axis=1)
    print("End Encoding")
    df = df[df.apply(lambda x: len(x['Face Encoding']) == 1, axis= 1)] #@@@@ Here because of the Dataset @@@
    if df_save_path:
        save_binary(df, df_save_path) 
    print(df.head(10))

    #Encode array has to be a 'pure' array to sklearn train
    # I won't find another way to do with thease piece of code maybe i am dumb
    X_train = []
    for elem in df['Face Encoding'].values:
        X_train.append(elem[0]) 

    model = knn_train(X_train, df['Name'].values, model_save_path)
    print("End of first train")
    return model, df

# Realizando leitura do dataset
# Lembrar de definir esta variáveis via linha de comnado
model_save_path = "./knn_model.clf"
train_dir = "archive/lfw-deepfunneled"
df_save_path = 'bkp/dataset_example.pkl'
df = pd.read_pickle(df_save_path)  #Lembrar de definir o carregamento facultativo









