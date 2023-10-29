from flask import Flask, render_template, Response
import cv2
import numpy as np
import requests
import socket



app = Flask(__name__)


# camera = cv2.VideoCapture('rtsp://freja.hiof.no:1935/rtplive/_definst_/hessdalen03.stream')  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
camera = cv2.VideoCapture(3)


def gen_frames():  # generate frame by frame from camera    
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    server_addr = "https://api.foxworld.online/neurosocket/cameras/toggle"

    requests.post(server_addr, json={
        "room":"uuid",
        "cam_code":"uuid",
        "status":"up",
        "ip": f'{socket.gethostbyname(socket.gethostname())}'
    })
    app.run(port=5003)
    print("print!")

    requests.post(server_addr, json={
        "room":"uuid",
        "cam_code":"uuid",
        "status":"down",
        "ip": f'{socket.gethostbyname(socket.gethostname())}'
    })