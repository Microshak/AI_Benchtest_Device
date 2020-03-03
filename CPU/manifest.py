from flask_apscheduler import APScheduler
from flask import Blueprint, request, jsonify, session
import requests
import socket
import json
import os
manifest = Blueprint('manifest', 'manifest' ,url_prefix='/manifest')
scheduler = APScheduler()





def set_manifest():
    f = open("manifest_cpu.json", "r")
    manifest = f.read()

    data = json.loads(manifest)
    data['host_name'] =  socket.gethostname()

    gw = os.popen("ip -4 route show default").read().split()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect((gw[2], 0))
    ipaddr = s.getsockname()[0]

    data['ip_address'] = ipaddr
    url = 'https://ai-benchtest.azurewebsites.net/device'
    r = requests.post(url = url, json =data) 
    txt = r.text 
    print(txt)



set_manifest()
scheduler.add_job(id ='Scheduled task', func =set_manifest, trigger = 'interval', minutes = 10)
scheduler.start()    
