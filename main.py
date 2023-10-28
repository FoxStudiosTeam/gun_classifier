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
from classifier import get_model


def get_image_value(path, dim): # ЧТЕНИЕ И РЕСАЙЗ ИЗОБРАЖЕНИЯ
    img = image.load_img(path, target_size = dim)
    img = image.img_to_array(img)
    return img / 255

def get_img_array(img_paths, dim): # ПЕРЕВОД В МАССИВ ИЗОБРАЖЕНИЙ
    final_array = []
    from tqdm import tqdm
    for path in tqdm(img_paths):
        img = get_image_value(path, dim)
        final_array.append(img)
    final_array = np.array(final_array)  
    return final_array

def get_tts(): # get test tranin split
    DIM = (150,150) 
    np.random.seed(10)        
    o2_paths = [f'./Separated/FinalImages/other2/{i}' for i in os.listdir('./Separated/FinalImages/other2')] 
    o2_labels = [0 for i in range(len(o2_paths))]
    np.random.shuffle(o2_paths)
    o3_paths = [f'./Separated/FinalImages/other3/{i}' for i in os.listdir('./Separated/FinalImages/other3')] 
    o3_labels = [0 for i in range(len(o3_paths))]
    np.random.shuffle(o3_paths)

    guns_paths = []
    for _, path in enumerate(os.listdir('./Separated/FinalImages/Gun')):
        guns_paths.append(f'./Separated/FinalImages/Gun/{path}')

    guns_labels = [1 for i in range(len(guns_paths))]
    np.random.shuffle(guns_paths)

    pistol_paths = [f'./Separated/FinalImages/Pistol/{i}' for i in os.listdir('./Separated/FinalImages/Pistol')] 
    pistol_labels = [1 for i in range(len(pistol_paths))]
    np.random.shuffle(pistol_paths)
    rifle_paths = [f'./Separated/FinalImages/Rifle/{i}' for i in os.listdir('./Separated/FinalImages/Rifle')] 
    rifle_labels = [1 for i in range(len(rifle_paths))]   

    neg_paths = []
    for _, path in enumerate(os.listdir('./Separated/FinalImages/NoWeapon')):
        neg_paths.append(f'./Separated/FinalImages/NoWeapon/{path}')
    np.random.shuffle(neg_paths)

    pistol_labels = [1 for i in range(len(pistol_paths))]
    rifle_labels = [1 for i in range(len(rifle_paths))]
    neg_labels = [0 for i in range(len(neg_paths))]
    pre_paths = pistol_paths + rifle_paths + neg_paths + guns_paths + o2_paths + o3_paths
    pre_labels = pistol_labels + rifle_labels + neg_labels + guns_labels + o2_labels + o3_labels
    paths = []
    labels = []

    percentage = 0.05

    for i in range(len(pre_labels)):
        if random.random() < percentage: #0 -> 1
            labels.append(pre_labels[i])
            paths.append(pre_paths[i])

    x_train, x_test, y_train, y_test = train_test_split(paths, labels, stratify = labels, train_size = .90, random_state = 10)

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
    model_checkpoint = ModelCheckpoint('my_model.h5', verbose = 1, save_best_only=True,
                                    monitor = 'val_loss')
    
    # УМЕНЬШЕНИЕ LEARNING RATE ЕСЛИ НАЧИНАЕТ ПЛОХО УЧИТЬСЯ
    lr_plat = ReduceLROnPlateau(patience = 2, mode = 'min') 
    epochs = 10000
    batch_size = 512 # ОТНОСИТЕЛЬНОЕ КОЛИЧЕСТВО (КОЛ-ВО / САЙЗ?)
    model = get_model()

    x_train, x_test, y_train, y_test = get_tts()

    model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, callbacks = [early_stopping, model_checkpoint, lr_plat], validation_data = (x_test, y_test), verbose= 1)