import cv2
import pickle
from io import BytesIO
import time
import requests
from PIL import Image
import numpy as np
from importlib import import_module
import os
from flask import Flask, render_template, Response
from flask import request

from ML.Haar import Haar
import imutils

app = Flask(__name__)
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

def gen(h, height,width, downsample):
    net = cv2.CascadeClassifier('ML/haarcascade_frontalface_alt.xml')

    while True:
      url = f'http://localhost:5000/image.jpg?height={height}&width={width}'
      r = requests.get(url) # replace with your ip address
      curr_img = Image.open(BytesIO(r.content))
      
      frame = cv2.cvtColor(np.array(curr_img), cv2.COLOR_RGB2BGR)
      dwidth = float(width) * (1 - float(downsample))
      dheight = float(height) * (1 - float(downsample))
      frame = imutils.resize(frame, width=int(dwidth), height=int(dheight))

      frame = Haar.face_detect(frame, net)
      
      frame = cv2.imencode('.jpg', frame)[1].tobytes()      
      yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/image.jpg')
def image():
    h = Haar()
    height = request.args.get('height')
    width = request.args.get('width')
    downsample = request.args.get('downsample')
 
    """Returns a single current image for the webcam"""
    return Response(gen(h,height,width, downsample), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port="5001", threaded=True)

while True:
    r = requests.get('http://192.168.1.1:5000/image.jpg') # replace with your ip address
    curr_img = Image.open(BytesIO(r.content))
    curr_img_cv2 = cv2.cvtColor(np.array(curr_img), cv2.COLOR_RGB2BGR)

