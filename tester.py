import cv2
import numpy as np
from main import get_model

DIM = (150,150) 

model = get_model()
model.load_weights('my_model.h5')

paths = ["C:\\prog\\py\\nn\\Separated\\FinalImages\\NoWeapon\\303.png"]
for path in paths:
    data = cv2.imread(path)
    data = cv2.resize(data, dsize =DIM, interpolation = cv2.INTER_CUBIC)
    data = np.expand_dims(data, axis=0)
    res = model.predict(x= data, batch_size= 1)
    print(f'{path}: {res}')