import flask
from flask import Flask, render_template, Response
import cv2
import numpy as np
import requests
import socket
from flask import request
app = Flask(__name__)
camera = cv2.VideoCapture(0)


def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    server_addr = "https://api.foxworld.online/neurosocket/cameras/toggle"
    ip = "0.0.0.0"
    port = 6000
    IP = requests.get('https://api.ipify.org/').text
    ip_port = f'{IP}:{port}/video_capture'
    requests.post(server_addr, json={
        "room": "foxwatch1",
        "cam_code": "камера 1",
        "status": "up",
        # "ip": f'{socket.gethostbyname(socket.gethostname())}'
        "ip": ip_port
    })

    # app.run(debug=True, host=ip, port=port)
    app.run(host=ip, port=port)
    print("print!")

    requests.post(server_addr, json={
        "room": "foxwatch1",
        "cam_code": "камера 1",
        "status": "down",
        "ip": ip_port
    })