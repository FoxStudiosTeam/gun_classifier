from flask import Flask, render_template, Response
import cv2
import numpy as np
import requests
import socket

app = Flask(__name__)



def gen_frames(camera_url):
    camera = cv2.VideoCapture(camera_url)
    while True:
        success, frame = camera.read()        
        if not success:
            break
        else:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (100, 100), (w-100, h-100), #startX+ startY+
    	        (0, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/capture_cam/<camera_ip>/<a>')
def video_feed(camera_ip, a):
    return Response(gen_frames(f'http://{camera_ip}/{a}'), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    server_addr = "https://api.foxworld.online/neurosocket/cameras/toggle"
    
    requests.post(server_addr, json={
        "room":"uuid",
        "cam_code":"uuid",
        "status":"up",
        "ip": f'{socket.gethostbyname(socket.gethostname())}'
    })
    
    app.run(port=5005)
    print("print!")

    requests.post(server_addr, json={
        "room":"uuid",
        "cam_code":"uuid",
        "status":"down",
        "ip": f'{socket.gethostbyname(socket.gethostname())}'
    })