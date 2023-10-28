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
from main import get_img_array
from detector import get_model


def parse_xml(fn):
    root = ET.parse(fn).getroot()
    w  = float(root.find('size').find('width').text)
    h = float(root.find('size').find('height').text)
    
    bboxs = []
    for obj in root.findall('object'):
        # ОТНОСИТЕЛЬНЫЙ РАЗМЕР
        xmin = float(obj.find('bndbox').find('xmin').text) # 0 -> 1 
        ymin = float(obj.find('bndbox').find('ymin').text) # 0 -> 1 
        xmax = float(obj.find('bndbox').find('xmax').text) # 0 -> 1 
        ymax = float(obj.find('bndbox').find('ymax').text) # 0 -> 1 
        bboxs.append((xmin / w, ymin / h,(xmax-xmin) / w, (ymax-ymin) / h))
    keep = True
    if len(bboxs) == 1:
        bboxs.append(bboxs[0]) # duplicate || zeros
    
    if len(bboxs) > 2 or len(bboxs) == 0:
        keep = False

    return bboxs, keep

def get_tts(): # get test tranin split
    DIM = (224, 224) 
     
    names = os.listdir('./Detector/JPEGImages/')
  
    pre_paths = [f'./Detector/JPEGImages/{i}' for i in names]

    try:
        with open("parsed_detector_labels.pkl", "rb") as f:
            pre_labels = pickle.load(f)
    except:
        pre_labels = [parse_xml(f'./Detector/Annotations/{i.removesuffix(".jpg") + ".xml"}') for i in names]
        with open("parsed_detector_labels.pkl", "wb") as f:
            pickle.dump(pre_labels, f)
    

    paths = []
    labels = []
    for i, (label, keep) in enumerate(pre_labels):
        if keep:
            labels.append(label)
            paths.append(pre_paths[i])
    pre_labels = labels
    pre_paths = paths

    paths = []
    labels = []

    percentage = 1.
    if percentage == 1.:
        paths = pre_paths
        labels = pre_labels
    else:
        for i in range(len(pre_labels)):
            if random.random() < percentage: #0 -> 1
                labels.append(pre_labels[i])
                paths.append(pre_paths[i])
    labels = np.array(labels)
    labels = labels.reshape(labels.shape[0], 8)
    print("PATHS PREPARED!")
    x_train, x_test, y_train, y_test = train_test_split(paths, labels, train_size = .90, random_state = 10)

    new_x_train = get_img_array(x_train, DIM)
    print("TRAIN ARRAY CONVERTED!")
    new_x_test = get_img_array(x_test, DIM)
    print("IMAGE ARRAY CONVERTED!")

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_test = np.array(y_test)
    y_train = np.array(y_train)
    tts = (new_x_train, new_x_test, y_train, y_test)
    return tts




if __name__ == "__main__":
    # ЕСЛИ ПЕРЕСТАЕТ УЧИТЬСЯ ТО СТОПАЕМ
    early_stopping = EarlyStopping(monitor='val_loss', verbose = 1, patience=10, min_delta = .00075)

    # СОХРАНЕНИЕ ЛУЧШЕГО ПОКОЛЕНИЯ, ОЦЕНИЯВАЯ ПО val_loss
    model_checkpoint = ModelCheckpoint('detector.h5', verbose = 1, save_best_only=True,
                                    monitor = 'val_loss')
    
    # УМЕНЬШЕНИЕ LEARNING RATE ЕСЛИ НАЧИНАЕТ ПЛОХО УЧИТЬСЯ
    lr_plat = ReduceLROnPlateau(patience = 2, mode = 'min') 
    epochs = 10000
    batch_size = 512 # ОТНОСИТЕЛЬНОЕ КОЛИЧЕСТВО (КОЛ-ВО / САЙЗ?)
    model = get_model()

    x_train, x_test, y_train, y_test = get_tts()

    model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, callbacks = [early_stopping, model_checkpoint, lr_plat], validation_data = (x_test, y_test), verbose= 1)



