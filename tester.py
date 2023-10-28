import cv2
import numpy as np
import classifier, detector

CLASSIFIER_DIM = (150,150) 
DETECTOR_DIM = (224,224) 
cl_model = classifier.get_model()
#cl_model.load_weights('my_model.h5')
dt_model = detector.get_model()
dt_model.load_weights('detector.h5')



paths_to_test = [
    "C:\\prog\\py\\nn\\Separated\\Pistol_2597_3.jpg",
    "C:\\prog\\py\\nn\\Separated\\Pistol_2540_41.jpg",
]
for path in paths_to_test:
    img = cv2.imread(path)
    y, x = img.shape[:2]
    print(x, y)
    data = cv2.resize(img, dsize = DETECTOR_DIM , interpolation = cv2.INTER_CUBIC)
    data = np.expand_dims(data, axis=0)
    res = dt_model.predict(x= data, batch_size= 1)
    x1, y1, x2, y2 = res.reshape(4)
    print(x1, y1, x2, y2)
    x1 *= x
    y1 *= y
    x2 *= x
    y2 *= y
    x1 = min(x1, x2)
    y1 = min(y1, y1)
    x2 = max(x1, x2)
    y2 = max(y1, y2)
    print(x1, y1, x2, y2)
    x1 = int(x1)
    x2 = int(x2)
    y1 = int(y1)
    y2 = int(y2)
    print(x1, y1, x2, y2)
    print(img.shape)

    cutted_image = img.copy()[y1:y2+1, x1:x2+1]
    print(cutted_image)
    if sum(cutted_image.shape[:3]) != 0:
        data = cv2.resize(cutted_image, dsize = CLASSIFIER_DIM, interpolation = cv2.INTER_CUBIC)
        data = np.expand_dims(data, axis=0)
        res = cl_model.predict(x= data, batch_size= 1)
        # [a, b, c]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
        #cv2.putText(img, f'{res}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,0),2)
        cv2.imshow(f'{res}', img)
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
