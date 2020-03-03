"""
The flask application package.
"""

from flask import Flask
cpu = Flask(__name__)
print(__name__)



from CPU.HAAR import haar
from CPU.Yolo import yolo
from CPU.manifest import manifest
cpu.register_blueprint(haar)
cpu.register_blueprint(yolo)
cpu.register_blueprint(manifest)