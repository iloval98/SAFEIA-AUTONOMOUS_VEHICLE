#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Code original par Thibault Neuveu (thibo73800 sous Github)
# 10/11/2020 - Ajout modifications par Frederic Deschamps (commentaires et sortie graphique)

import csv
import cv2
import os
import numpy as np
import random
import matplotlib.pyplot as plt


from random import shuffle

from sklearn.model_selection import train_test_split

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Cropping2D, Lambda, Dropout



DATA_PATH = "data/driving_log.csv"
DATA_IMG = "data/"

def build_model():
    """
        Build keras model
    """
    # Attention la couche lambda peut poser problemes a certains outils 
    # Il peut etre necessaire de la supprimer. Dans ce cas augmenter 
    # le nombre d epochs. Mais cela ne sera pas suffisant pour finir le 
    # premier circuit.
    model = Sequential()
    model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape = (160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape = (160, 320, 3)))
    model.add(Conv2D(8, 9, strides=(4, 4), padding="same", activation="elu"))
    model.add(Conv2D(16, 5, strides=(2, 2), padding="same", activation="elu"))
    model.add(Conv2D(32, 4, strides=(1, 1), padding="same", activation="elu"))
    model.add(Flatten())
    model.add(Dropout(.6))
    model.add(Dense(1024, activation="elu"))
    model.add(Dropout(.3))
    model.add(Dense(1))

    #ada = optimizers.Adagrad(lr=0.001)
    model.compile(loss="mse", optimizer="adam",metrics=['accuracy','mean_squared_error'])

    return model

def get_data(log_content, index_list, batch_size, strict=True):
    """
        Return Data from the simulator
        **input:
            *log_content (2dim Array) Data from the log file
            *index_list: Index to used to create each batch
            *batch_size (Int) Size of each batch
    """
    images, rotations = [], []
    while True:
        if not strict:
            shuffle(index_list)
        for index in index_list:
            # Futur angle correction
            angle_correction = [0., 0.25, -.25]
            # Randomly select one of of three images
            i = random.choice([0, 1, 2]) # [Center, Left, Right]
            img = cv2.imread(os.path.join(DATA_IMG, log_content[index][i]).replace(" ", ""))

            if img is None: continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Get the rotation
            rotation = float(log_content[index][3])

            #if not strict and random.random() > np.sqrt(rotation ** 2) + 0.1:
            #    continue

            # Apply correction
            rotation = rotation + angle_correction[i]
            # 1/2: random roation of the image
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
                rotation = rotation * -1
            # Append the new image/rotation to the current batch
            images.append(img)
            rotations.append(rotation)
            # Yield the new batch
            if len(images) >= batch_size:
                yield np.array(images), np.array(rotations)
                # Next batch
                images, rotations = [], []

def main():
    """
        Main function to train the model
    """
    # Open log csv
    with open(DATA_PATH, "r") as f:
        content = [line for line in csv.reader(f)]
    # Split in train/validation set
    random_indexs = np.array(range(len(content)))
    train_index, valid_index = train_test_split(random_indexs, test_size=0.15)
    model = build_model()

    print("Train size = %s" % len(train_index))
    print("Valid size = %s" % len(valid_index))

    BATCH_SIZE = 64 # avant 64
    # model.fit_genrator
    history = model.fit_generator(
        generator=get_data(content, train_index, BATCH_SIZE, strict=False),
        steps_per_epoch=len(train_index) / BATCH_SIZE,
        validation_data=get_data(content, valid_index, BATCH_SIZE, strict=True),
        validation_steps=len(valid_index) / BATCH_SIZE,
        epochs=30) # FDE - Fonctionne avec 12 epochs, test avec 30 pour ameliorer comportement sur second circuit.

    model.save("model_2.h5")
    
    print('XXXXX  SCORING  XXXXX') 
    
    # summary
    print(model.summary())
    # Accurracy
  

    # Scoring
    print(history.history.keys())

    # FDE - summarize history for mean squared error
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('model Mean Squared Error')
    plt.ylabel('MSDF')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


  
   
    # FDE - summarize history for loss
    plt.plot(history.history['loss'],history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('val_Loss')
    plt.xlabel('loss')
    plt.legend(['loss', 'val_oss'], loc='upper left')
    plt.show()

    # FDE - summarize for ac loss
    plt.plot(history.history['val_acc'],history.history['val_loss'])
    plt.title('acc & loss')
    plt.ylabel('val_Loss')
    plt.xlabel('val_acc')
    plt.legend(['val_acc', 'val_oss'], loc='upper left')
    plt.show()
   
    


if __name__ == '__main__':
    main()
