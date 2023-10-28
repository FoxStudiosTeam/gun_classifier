from flask import Flask, render_template, Response
import cv2
import numpy as np
import classifier, detector

app = Flask(__name__)

# camera = cv2.VideoCapture('rtsp://freja.hiof.no:1935/rtplive/_definst_/hessdalen03.stream')  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
camera = cv2.VideoCapture(0)

DIM =  (150,150) 

def gen_frames():  # generate frame by frame from camera
    model = classifier.get_model()
    #model.load_weights('detector.h5')
    
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        
        if not success:
            break
        else:
            data = cv2.resize(frame, dsize =DIM, interpolation = cv2.INTER_CUBIC)
            data = np.expand_dims(data, axis=0)
            res = model.predict(x= data, batch_size= 1)
            #print(res)
            # Визуализация класса на кадре
            cv2.putText(frame, f'{res}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
    app.run(debug=True)