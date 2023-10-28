from flask import Flask, render_template, Response
import cv2
import numpy as np
import requests
import socket

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
    
    requests.post(server_addr, json={
        "room":"uuid",
        "cam_code":"uuid",
        "status":"up",
        "ip": f'{socket.gethostbyname(socket.gethostname())}'
    })

    app.run(debug=True)
    print("print!")

    requests.post(server_addr, json={
        "room":"uuid",
        "cam_code":"uuid",
        "status":"down",
        "ip": f'{socket.gethostbyname(socket.gethostname())}'
    })