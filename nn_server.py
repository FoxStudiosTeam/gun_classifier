from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import requests
import socket
import classifier, detector, detector_v2
app = Flask(__name__)


CLASSIFIER_DIM = (150,150) 
DETECTOR_DIM = (224,224) 
cl_model = classifier.get_model()
cl_model.load_weights('classifier_save2.h5') #my_model
dt_model = detector_v2.get_model()
dt_model.load_weights('detector_v2_podgorelo.h5') #detector_v2_podgorelo

def gen_frames(camera_url):
    camera = cv2.VideoCapture(camera_url)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            data = cv2.resize(frame, dsize =DETECTOR_DIM, interpolation = cv2.INTER_CUBIC)
            data = np.expand_dims(data, axis=0)
            data = data / 255
            preds = dt_model.predict(data)[0]
            (startX, startY, endX, endY) = preds
            h, w = frame.shape[:2]
            startX = int(startX * w)
            startY = int(startY * h)
            endX = int(endX * w)
            endY = int(endY * h)


            cutted_image = frame.copy()[(startY-15):endY+15, (startX-15):endX+15] # startY+ startX+
            try:
                data = cv2.resize(cutted_image, dsize = CLASSIFIER_DIM, interpolation = cv2.INTER_CUBIC)
                data = np.expand_dims(data, axis=0)
                data = data / 255
                res = cl_model.predict(x= data, batch_size= 1)
                # [a, b, c]
                cv2.rectangle(frame, (startX, startY), (endX, endY), #startX+ startY+
                (0, 255, 0), 2)
                labels = ["OTH", "SHR", "LNG"]
                
                pred = labels[max(enumerate(res.tolist()),key=lambda x: x[1])[0]]
                cv2.putText(frame, f'{pred}', (startX-15, startY-15), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 2)
                
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except:
                pass
            



@app.route('/capture_cam/<camera_ip>')
def video_feed_1arg(camera_ip):
    url = f'http://{camera_ip}'
    for key in request.args.keys():
        url += f'?{key}={request.args[key]}'
    print(url)
    return Response(gen_frames(url), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_cam/<camera_ip>/<a>')
def video_feed_2arg(camera_ip, a):
    url = f'http://{camera_ip}/{a}'
    for key in request.args.keys():
        url += f'?{key}={request.args[key]}'
    print(url)
    return Response(gen_frames(url), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_cam/<camera_ip>/<a>/<b>')
def video_feed_3arg(camera_ip, a, b):
    url = f'http://{camera_ip}/{a}/{b}'
    for key in request.args.keys():
        url += f'?{key}={request.args[key]}'
    return Response(gen_frames(url), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_cam/<camera_ip>/<a>/<b>/<c>')
def video_feed_4arg(camera_ip, a, b, c):
    url = f'http://{camera_ip}/{a}/{b}/{c}'
    for key in request.args.keys():
        url += f'?{key}={request.args[key]}'
    return Response(gen_frames(url), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    server_addr = "https://api.foxworld.online/neurosocket/cameras/toggle/room"
    ip = "0.0.0.0"
    port = 25602
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