import classifier, detector, detector_v2
import numpy as np
import cv2



#model = classifier.get_model()
#model.load_weights('my_model.h5')
# Захват видеопотока с веб-камеры
#cv2.setUseOptimized(True)
#cv2.setNumThreads(10) #change depending on your computer

CLASSIFIER_DIM = (150,150) 
DETECTOR_DIM = (224,224) 
cl_model = classifier.get_model()
cl_model.load_weights('my_model.h5')
dt_model = detector_v2.get_model()
dt_model.load_weights('detector_v2_podgorelo.h5')
cap = cv2.VideoCapture(0)#"http://127.0.0.1:5000/video_feed"
while True:
    # Получение текущего кадра
    ret, frame = cap.read()
    
    data = cv2.resize(frame, dsize =DETECTOR_DIM, interpolation = cv2.INTER_CUBIC)
    data = np.expand_dims(data, axis=0)
    data = data / 255
    preds = dt_model.predict(data)[0]
    (startX, startY, endX, endY) = preds
 
    (h, w) = frame.shape[:2]

    startX = int(startX * w)
    startY = int(startY * h)
    endX = int(endX * w)
    endY = int(endY * h)


    cutted_image = frame.copy()[startY:endY+1, startX:endX+1] # startY+ startX+
    # todo: обернуть в трай
    if sum(cutted_image.shape[:3]) != 0:
        data = cv2.resize(cutted_image, dsize = CLASSIFIER_DIM, interpolation = cv2.INTER_CUBIC)
        data = np.expand_dims(data, axis=0)
        data = data / 255
        res = cl_model.predict(x= data, batch_size= 1)
        # [a, b, c]
        cv2.rectangle(frame, (startX, startY), (endX, endY), #startX+ startY+
    	(0, 255, 0), 2)
        cv2.putText(frame, f'{res}', (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,0),2)
        cv2.imshow(f'A', frame)
    
    # Прекращение обработки по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break











    #data = cv2.resize(frame, dsize =DIM, interpolation = cv2.INTER_CUBIC)
    #data = np.expand_dims(data, axis=0)

    #ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    #ss.setBaseImage(frame)
    #ss.switchToSelectiveSearchFast()

    #rects = ss.process() 
    
    #data = np.array(frame)
    #data.reshape((3, 150, 1, 1))
    
    #print(data.shape)
    #res = model.predict(x= data, batch_size= 1)
    #print(res)
    # Визуализация класса на кадре
    #cv2.putText(frame, f'{res}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()