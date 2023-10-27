import cv2
from main import *

DIM =  (150,150) 

model = get_model()
model.load_weights('my_model.h5')


data = cv2.imread("C:\\prog\\py\\nn\\Separated\\FinalImages\\NoWeapon\\303.png")
data = cv2.resize(data, dsize =DIM, interpolation = cv2.INTER_CUBIC)
data = np.expand_dims(data, axis=0)
res = model.predict(x= data, batch_size= 1)
print(res)


data = cv2.imread("C:\\prog\\py\\nn\\0bd1cce5c59b8d2ac79a613bc71209e2.jpg")
data = cv2.resize(data, dsize =DIM, interpolation = cv2.INTER_CUBIC)
data = np.expand_dims(data, axis=0)
res = model.predict(x= data, batch_size= 1)
print(res)
