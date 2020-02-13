"""
The flask application package.
"""

from flask import Flask
app = Flask(__name__)
print(__name__)



from App.HAAR import haar
from App.manifest import manifest
app.register_blueprint(haar)
app.register_blueprint(manifest)