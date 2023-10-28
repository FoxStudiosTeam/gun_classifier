import os
import numpy as np 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing import image 
from keras import regularizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten 
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import random



# Возвращает модель
def get_model(): 
    inp_shape = (150, 150, 3)
    act = 'relu'
    drop = .25
    kernal_reg = regularizers.l1(.001)
    optimizer = Adam(lr = .0001)    
    model = Sequential() 
    model.add(Conv2D(64, kernel_size=(3,3),activation=act, input_shape = inp_shape, 
                     kernel_regularizer = kernal_reg,
                     kernel_initializer = 'he_uniform', padding = 'same', name = 'Input_Layer'))
    model.add(MaxPooling2D(pool_size=(2, 2),  strides = (3, 3)))
    model.add(Conv2D(64, (3, 3), activation=act, kernel_regularizer = kernal_reg, 
                     kernel_initializer = 'he_uniform', padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides = (3, 3))) 
    model.add(Conv2D(128, (3, 3), activation=act, kernel_regularizer = kernal_reg, 
                     kernel_initializer = 'he_uniform', padding = 'same'))
    model.add(Conv2D(128, (3, 3), activation=act, kernel_regularizer = kernal_reg, 
                     kernel_initializer = 'he_uniform', padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides = (3, 3)))
    model.add(Flatten())
    model.add(Dense(128, activation=act))
    model.add(Dense(64, activation=act))
    model.add(Dense(32, activation=act))
    model.add(Dropout(drop))

    # СИГМОИД, Т.К ОДИН ВЫХОД
    model.add(Dense(1, activation='sigmoid', name = 'Output_Layer'))

    # БИНАРНАЯ КРОСС ЭНТРОПИЯ, Т.К ОДИН ВЫХОД
    model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return model 


