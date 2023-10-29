
import numpy as np
import cv2






cap = cv2.VideoCapture("http://61.211.241.239/nphMotionJpeg?Resolution=320x240&Quality=Standard")#"http://127.0.0.1:5000/video_feed"
while True:
    # Получение текущего кадра
    ret, frame = cap.read()
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


