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
import imutils
import json
import requests
from flask import Blueprint, request, jsonify, session   

haar = Blueprint('haar', 'haar' ,url_prefix='/haar')


"""use opencv and Haar Cascade for face detection"""
def face_detect(img, net):

        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = net.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
        except Exception as e:
            print("failed HAAR" + str(e))
        return img



encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

def gen(height,width, downsample, camera):
    net = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    while True:
      url = f'http://{camera}:5000/image.jpg?height={height}&width={width}'
      r = requests.get(url) # replace with your ip address
      curr_img = Image.open(BytesIO(r.content))
      
      frame = cv2.cvtColor(np.array(curr_img), cv2.COLOR_RGB2BGR)
      dwidth = float(width) * (1 - float(downsample))
      dheight = float(height) * (1 - float(downsample))
      frame = imutils.resize(frame, width=int(dwidth), height=int(dheight))

      frame = face_detect(frame, net)
      
      frame = cv2.imencode('.jpg', frame)[1].tobytes()      
      yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@haar.route('/image.jpg')
def image():
    
    height = request.args.get('height')
    width = request.args.get('width')
    downsample = request.args.get('downsample')
    camera = request.args.get('camera')
 
    """Returns a single current image for the webcam"""
    return Response(gen(height,width, downsample, camera), mimetype='multipart/x-mixed-replace; boundary=frame')
