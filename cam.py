from main import *


#res = model.predict(windows)
# NDarray -> ?????
import cv2

DIM =  (150,150) 

model = get_model()
model.load_weights('my_model.h5')
# Захват видеопотока с веб-камеры
cap = cv2.VideoCapture(0)
while True:
    # Получение текущего кадра
    ret, frame = cap.read()
    data = cv2.resize(frame, dsize =DIM, interpolation = cv2.INTER_CUBIC)
    data = np.expand_dims(data, axis=0)
    #data = np.array(frame)
    #data.reshape((3, 150, 1, 1))
    
    #print(data.shape)
    res = model.predict(x= data, batch_size= 1)
    #print(res)
    # Визуализация класса на кадре
    cv2.putText(frame, f'{res}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Отображение кадра
    cv2.imshow('Video Classification', frame)
    
    # Прекращение обработки по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()