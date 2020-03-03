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
import jetson.inference
import jetson.utils

inception = Blueprint('inception', 'inception' ,url_prefix='/inception')




def gen(height,width, downsample, camera):

    net = jetson.inference.detectNet("ssd-inception-v2", threshold=0.5)
    camera = jetson.utils.gstCamera(1280,720,"/dev/video0")

    while True:
      url = f'http://{camera}:5000/image.jpg?height={height}&width={width}'
      r = requests.get(url) # replace with your ip address
      curr_img = Image.open(BytesIO(r.content))
      img, width, height = camera.CaptureRGBA()
      detections = net.Detect(img,width, height)

      frame = cv2.imencode('.jpg', frame)[1].tobytes()      
      yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@inception.route('/image.jpg')
def image():
    
    height = request.args.get('height')
    width = request.args.get('width')
    downsample = request.args.get('downsample')
    camera = request.args.get('camera')
 
    """Returns a single current image for the webcam"""
    return Response(gen(height,width, downsample, camera), mimetype='multipart/x-mixed-replace; boundary=frame')
