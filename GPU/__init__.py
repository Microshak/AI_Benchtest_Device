"""
The flask application package.
"""

from flask import Flask
gpu = Flask(__name__)
print(__name__)



from GPU.HAAR import haar
gpu.register_blueprint(haar)
