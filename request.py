# import requests

# url = 'http://localhost:5000/predict_api'
# r = requests.post(url,json={'text':  'bad boy'})

# print(r.json())

import requests
url = 'http://localhost:5000/api'
r = requests.post(url,json={'exp':'devansh',})
print(r.json())