import io
from flask import Flask, make_response, render_template, Response, request
import cv2
import numpy as np
import requests
import socket
from flask import Flask, request, jsonify
import classifier, detector_v2
from PIL import Image
app = Flask(__name__)

CLASSIFIER_DIM = (150,150) 
DETECTOR_DIM = (224,224) 
cl_model = classifier.get_model()
cl_model.load_weights('my_model.h5') #! CLASSIFIER.PY
dt_model = detector_v2.get_model()
dt_model.load_weights('detector_v2_podgorelo.h5')


@app.route('/image', methods=['POST'])
def image():
    print(f"New post request from {request.host}")
    try:
        file = request.files['file']
        print(f"Get file with name {file.filename}!")
    except:
        print(f"Corrupted or missied file!")
        return []
    img = Image.open(file)
    img.load()
    background = Image.new("RGB", img.size, (255, 255, 255))
    background.paste(img, mask=img.split()[3])
    img = np.array(background)

    data = cv2.resize(img, dsize = DETECTOR_DIM, interpolation = cv2.INTER_CUBIC)
    data = np.expand_dims(data, axis=0)
    data = data / 255
    preds = dt_model.predict(data)[0]
    (startX, startY, endX, endY) = preds
 
    (h, w) = img.shape[:2]

    startX = int(startX * w)
    startY = int(startY * h)
    endX = int(endX * w)
    endY = int(endY * h)

    cutted_image = img.copy()[startY:endY+1, startX:endX+1] # startY+ startX+
    try:
        data = cv2.resize(cutted_image, dsize = CLASSIFIER_DIM, interpolation = cv2.INTER_CUBIC)
        data = np.expand_dims(data, axis=0)
        data = data / 255
        res = cl_model.predict(x= data, batch_size= 1)
        # [a, b, c]
        cv2.rectangle(img, (startX, startY), (endX, endY), #startX+ startY+
    	(0, 255, 0), 2)
        cv2.putText(img, f'{res}', (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)
        return img
    except:
        return img #todo: обернуть в файл
    

@app.route('/label', methods=['POST', 'OPTIONS'])
def label():
    origin = request.headers.get('Origin')
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Headers', 'x-csrf-token')
        response.headers.add('Access-Control-Allow-Methods',
                            'GET, POST, OPTIONS, PUT, PATCH, DELETE')
        if origin:
            response.headers.add('Access-Control-Allow-Origin', origin)
        else:
            response.headers.add('Access-Control-Allow-Credentials', 'true')
            if origin:
                response.headers.add('Access-Control-Allow-Origin', origin)
        return response
    
    print(f"New post request from {request.host}")
    try:
        file = request.files['file']
        print(f"Get file with name {file.filename}!")
    except:
        print(f"Corrupted or missied file!")
        return {"errors": True}
    img = Image.open(file)
    img.load()
    background = Image.new("RGB", img.size, (255, 255, 255))
    background.paste(img, mask=img.split()[3])
    img = np.array(background)

    data = cv2.resize(img, dsize = DETECTOR_DIM, interpolation = cv2.INTER_CUBIC)
    data = np.expand_dims(data, axis=0)
    data = data / 255
    preds = dt_model.predict(data)[0]
    (startX, startY, endX, endY) = preds
 
    (h, w) = img.shape[:2]

    startX = int(startX * w)
    startY = int(startY * h)
    endX = int(endX * w)
    endY = int(endY * h)

    cutted_image = img.copy()[startY:endY+1, startX:endX+1] # startY+ startX+
    try:
        data = cv2.resize(cutted_image, dsize = CLASSIFIER_DIM, interpolation = cv2.INTER_CUBIC)
        data = np.expand_dims(data, axis=0)
        data = data / 255
        res = cl_model.predict(x= data, batch_size= 1)
        # [a, b, c]
        cv2.rectangle(img, (startX, startY), (endX, endY), #startX+ startY+
    	(0, 255, 0), 2)
        cv2.putText(img, f'{res}', (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)

        labels = ["Иное", "Короткоствольное", "Длинноствольное"]
        
        return {"errors": False, "raw": res.flatten().tolist(), "prediction": labels[max(enumerate(res.tolist()),key=lambda x: x[1])[0]]}
        
    except:
        return {"errors": True}


@app.after_request
def after_request_func(response):
    origin = request.headers.get('Origin')
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Headers', 'x-csrf-token')
        response.headers.add('Access-Control-Allow-Methods',
                            'GET, POST, OPTIONS, PUT, PATCH, DELETE')
        if origin:
            response.headers.add('Access-Control-Allow-Origin', origin)
    else:
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        if origin:
            response.headers.add('Access-Control-Allow-Origin', origin)
    return response


@app.route('/only_classificator', methods=['POST'])
def only_classificator():
    print(f"New post request from {request.host}")
    try:
        file = request.files['file']
        print(f"Get file with name {file.filename}!")
    except:
        print(f"Corrupted or missied file!")
    img = Image.open(file)
    img.load()
    background = Image.new("RGB", img.size, (255, 255, 255))
    background.paste(img, mask=img.split()[3])
    img = np.array(background)

    data = cv2.resize(img, dsize = CLASSIFIER_DIM, interpolation = cv2.INTER_CUBIC)
    data = np.expand_dims(data, axis=0)
    data = data / 255
    res = cl_model.predict(x= data, batch_size= 1)
    return res

#from flask_cors import CORS
if __name__ == '__main__':
    server_addr = "https://api.foxworld.online/neurosocket/cameras/toggle/room"
    ip = "0.0.0.0"
    port = 5010
    ip_port = f'{ip}:{port}'

    requests.post(server_addr, json={
        "name": "Зал 3",
        "code":"foxwatch1",
        "status": "up",
        "ip": "http://foxworld.online:25601"
    })

    app.run(port=port, host=ip)
    print("print!")

    requests.post(server_addr, json={
        "name": "Зал 3",
        "code": "foxwatch1",
        "status": "down",
        "ip": "http://foxworld.online:25601"
    })
