import requests
import json
import cv2
import matplotlib.image as mpimg 


addr = 'http://localhost:8000'
test_url = addr + '/predict'

# prepare headers for http request  
content_type = 'application/json'
headers = {'content-type': content_type}

img = mpimg.imread(r'Plant-disease-detection\healthy2.JPG')
# encode image as jpeg
print(str(img))
data={"data":img.tolist()}
# send http request with image and receive response
response = requests.post(test_url,json=data)
# decode response
print(json.loads(response.text))

#image dimension to be sent is 256x256