import os
import random
from tqdm import tqdm
import xml.etree.ElementTree as ET
import cv2 
import shutil
import pickle
import keras
import numpy as np
from keras import regularizers
from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten 
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from classifier_train import get_img_array


def get_model():
    # fine-tuning (ТОНКАЯ ОТЛАДКА МОДЕЛИ VGG16 ДЛЯ НАШИХ НУЖД)
    vgg = keras.applications.vgg16.VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    vgg.trainable = False
    flatten = vgg.output
    flatten = Flatten()(flatten)
    bboxHead = Dense(128, activation="relu")(flatten)
    bboxHead = Dense(128, activation="relu")(flatten)
    bboxHead = Dense(64, activation="relu")(bboxHead)
    bboxHead = Dense(32, activation="relu")(bboxHead)
    bboxHead = Dense(4, activation="sigmoid")(bboxHead)
    model = Model(inputs=vgg.input, outputs=bboxHead)

    drop = .25
    kernal_reg = regularizers.l2(.001)
    optimizer = Adam(lr = .0001)


    model.compile(loss = 'mse', optimizer = optimizer, metrics = ['accuracy'])

    return model