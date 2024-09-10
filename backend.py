import torch
import torchvision
from flask import Flask,request,render_template, make_response,Response
from flask_cors import CORS
import json as json
import jsonpickle
import numpy as np

from model import CNN as cnn

app=Flask(__name__)
CORS(app)


@app.route('/')
def home():
    return 'hi'


labels={'Apple___Apple_scab':"scab",
 'Apple___Black_rot':"black rot",
 'Apple___Cedar_apple_rust':"rust",
 'Apple___healthy':"healthy",
 'Blueberry___healthy':"healthy",
 'Cherry_(including_sour)___Powdery_mildew':"powdery mildew",
 'Cherry_(including_sour)___healthy':"healthy",
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot':"gray leaf spot",
 'Corn_(maize)___Common_rust_':"rust",
 'Corn_(maize)___Northern_Leaf_Blight':"blight",
 'Corn_(maize)___healthy':"healthy",
 'Grape___Black_rot':"black rot",
 'Grape___Esca_(Black_Measles)':"black measles",
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)' :"blight",
 'Grape___healthy':"healthy",
 'Orange___Haunglongbing_(Citrus_greening)':"citrus greening",
 'Peach___Bacterial_spot':"bacterial spot",
 'Peach___healthy':"healthy",
 'Pepper,_bell___Bacterial_spot':"bacterial spot",
 'Pepper,_bell___healthy':"healthy",
 'Potato___Early_blight':"early blight",
 'Potato___Late_blight':"late blight",
 'Potato___healthy':"healthy",
 'Raspberry___healthy':"healthy",
 'Soybean___healthy':"healthy",
 'Squash___Powdery_mildew':"powdery mildew",
 'Strawberry___Leaf_scorch': "leaf scorch",
 'Strawberry___healthy':"healthy",
 'Tomato___Bacterial_spot':"bacterial spot",
 'Tomato___Early_blight':"early blight",
 'Tomato___Late_blight':"late blight",
 'Tomato___Leaf_Mold':" leaf mold",
 'Tomato___Septoria_leaf_spot':"fungal spot",
 'Tomato___Spider_mites Two-spotted_spider_mite':"spider mite",
 'Tomato___Target_Spot':"early blight",
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus':"yellow leaf virus",
 'Tomato___Tomato_mosaic_virus':"mosaic virus",
 'Tomato___healthy':"healthy"}


label=['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']


model=cnn(input_shape=3,hidden_units=40,output_shape=len(label))
model.load_state_dict(torch.load(r"Plant-disease-detection\disease_model_state.pth"))
model.to("cuda")

@app.route('/predict',methods=['POST'])
def predict():
    """ #print(request.json)
    sent_data=[float(x) for x in request.json['data']]
    #print('here',sent_data)
    data=np.array(sent_data).reshape(1,-1)
    pred={'prediction':str(model.predict(data)[0])}
    resp = make_response(json.dumps(pred))
    resp.headers["Content-Type"] = "application/json"
 """
    r = request
    # convert string of image data to uint8
    #print(r.get_data())
    data=r.json["data"]
    nparr =np.array(data,dtype=np.float32)
    # decode image
    # do some fancy processing here....
    t1=torchvision.transforms.Resize(256)
    nparr=torch.from_numpy(nparr).permute(2,0,1)
    #print(nparr)
    # build a response dict to send back to client
    t1=torchvision.transforms.ToTensor()
    print(nparr.unsqueeze(0))
    with torch.inference_mode():
          p=model(nparr.to("cuda").unsqueeze(0)).argmax(1)
          print(label[p],p)
          prediction=labels[label[p]] 
    response = {"prediction": prediction
                }
    # encode response using jsonpickle
    print(response)
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="data/json")

if __name__=="__main__":
    app.run(port=8000)