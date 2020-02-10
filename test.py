import json
import requests

f = open("manifest.json", "r")
manifest = f.read()
data = json.loads(manifest)
print(manifest)
url = 'https://ai-benchtest.azurewebsites.net/device'
r = requests.post(url = url, json =data) 
  
# extracting response text  
txt = r.text 
print(txt)