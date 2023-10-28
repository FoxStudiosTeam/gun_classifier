from keras.applications import VGG16
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import time
import random
from classifier_train import get_img_array
from tqdm import tqdm
data = []
targets = []
filenames = []


def parse_xml(fn):
    root = ET.parse(fn).getroot()
    #w  = float(root.find('size').find('width').text)
    #h = float(root.find('size').find('height').text)
    
    bboxs = []
    for obj in root.findall('object'):
        # ОТНОСИТЕЛЬНЫЙ РАЗМЕР
        xmin = float(obj.find('bndbox').find('xmin').text)
        ymin = float(obj.find('bndbox').find('ymin').text)
        xmax = float(obj.find('bndbox').find('xmax').text)
        ymax = float(obj.find('bndbox').find('ymax').text)
        bboxs.append((xmin, ymin, xmax, ymax))
    keep = True
    if len(bboxs) != 1:
        keep = False
    return bboxs[0], keep
# loop over the rows

names = os.listdir('./Detector/JPEGImages/')



for filename in tqdm(names):#tqdm():
    # break the row into the filename and bounding box coordinates
    if random.random() > 0.1: continue
    (startX, startY, endX, endY), keep = parse_xml("./Detector/Annotations/" + filename.replace(".jpg", ".xml"))
    if not keep: continue
    # derive the path to the input image, load the image (in OpenCV
    # format), and grab its dimensions
    imagePath = "./Detector/JPEGImages/" + filename
    image = cv2.imread(imagePath)
    (h, w) = image.shape[:2]
    # scale the bounding box coordinates relative to the spatial
    # dimensions of the input image
    startX = float(startX) / w
    startY = float(startY) / h
    endX = float(endX) / w
    endY = float(endY) / h
    # load the image and preprocess it
    target_size=(224, 224)

    image = get_img_array([imagePath], dim=(224, 224))[0]
    # update our list of data, targets, and filenames
    data.append(image)
    targets.append((startX, startY, endX, endY))
    filenames.append(filename)
    # convert the data and targets to NumPy arrays, scaling the input

# pixel intensities from the range [0, 255] to [0, 1]
data = np.array(data, dtype="float32")
targets = np.array(targets, dtype="float32")
# partition the data into training and testing splits using 90% of
# the data for training and the remaining 10% for testing
split = train_test_split(data, targets, filenames, test_size=0.10,
	random_state=42)
# unpack the data split
(trainImages, testImages) = split[:2]
(trainTargets, testTargets) = split[2:4]
(trainFilenames, testFilenames) = split[4:]
# write the testing filenames to disk so that we can use then
# when evaluating/testing our bounding box regressor

print("[INFO] saving testing filenames...")
f = open("FILENAMES", "w")
f.write("\n".join(testFilenames))
f.close()


# load the VGG16 network, ensuring the head FC layers are left off
vgg = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))
# freeze all VGG layers so they will *not* be updated during the
# training process
vgg.trainable = False
# flatten the max-pooling output of VGG
flatten = vgg.output
flatten = Flatten()(flatten)
# construct a fully-connected layer header to output the predicted
# bounding box coordinates
bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid")(bboxHead)
# construct the model we will fine-tune for bounding box regression
model = Model(inputs=vgg.input, outputs=bboxHead)
# initialize the optimizer, compile the model, and show the model
# summary
opt = Adam(lr=1e-4)
model.compile(loss="mse", optimizer=opt)
print(model.summary())
# train the network for bounding box regression
print("[INFO] training bounding box regressor...")
H = model.fit(
	trainImages, trainTargets,
	validation_data=(testImages, testTargets),
	batch_size=32,
	epochs=25,
	verbose=1)
# serialize the model to disk
print("[INFO] saving object detector model...")
model.save("detector_v2.h5", save_format="h5")
# plot the model training history
N = 25
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Bounding Box Regression Loss on Training Set")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("out")



'''






'''