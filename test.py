from flask import Flask, render_template, Response
import cv2
import numpy as np
import requests
import socket


app = Flask(__name__)

@app.route('/test')
def test():
    return {0: 0}


if __name__ == '__main__':
    IP = requests.get('https://api.ipify.org/').text
    
    app.run(host="0.0.0.0", port=1001)
    print(IP)