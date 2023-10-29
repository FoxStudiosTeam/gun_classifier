
import csv
import cv2
import numpy as np
import os
import classifier, detector, detector_v2
from classifier_train import get_img_array
CLASSIFIER_DIM = (150,150) 
DETECTOR_DIM = (224,224) 
cl_model = classifier.get_model()
cl_model.load_weights('classifier_save2.h5') #my_model

dt_model = detector_v2.get_model()
dt_model.load_weights('detector_v2_podgorelo.h5') #detector_v2_podgorelo


with open("out.csv", "w") as f:
    f.write("filename;weapon;classes\n")
    p = "./Тест/test_dataset/test_images_ready/"
    paths = [f'{p}{i}' for i in os.listdir(p) if i.endswith(".jpg")] 
    for path in paths:
        target_size=(224, 224)
        image = get_img_array([path], target_size)

        preds = dt_model.predict(image)[0]
        (startX, startY, endX, endY) = preds
    
        image = cv2.imread(path)
        (h, w) = image.shape[:2]

        startX = int(startX * w)
        startY = int(startY * h)
        endX = int(endX * w)
        endY = int(endY * h)

        
        # show the output image
        

        cutted_image = image.copy()[startY:endY+1, startX:endX+1] # startY+ startX+


        data = cv2.resize(cutted_image, dsize = CLASSIFIER_DIM, interpolation = cv2.INTER_CUBIC)
        data = np.expand_dims(data, axis=0) / 255
        res = cl_model.predict(x= data, batch_size= 1)
        # [a, b, c]
        cv2.rectangle(image, (startX, startY), (startX+endX, startY+endY), #startX+ startY+
        (0, 255, 0), 2)
        cv2.putText(image, f'{res}', (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,0),2)
        print(res)
        res = max(enumerate(res[0].tolist()),key=lambda x: x[1])[0]
        
        if res == 0:
            nowp = False
            wp = []
        else:
            nowp = True
            wp = [['unclassified_weapons', 'short_weapons', 'long_weapons'][res]]
        
        
        #cv2.imshow(f'A', image)
        #cv2.waitKey(0)

        fname = path.split("/")[-1]
        f.write(f"{fname};{nowp};{wp}\n")
