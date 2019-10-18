from keras.models import load_model
from flask import Flask,request,jsonify,render_template
import tensorflow as tf
from PIL import Image
import io
import numpy as np 
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import cv2

import json
model = load_model("model_densenet.h5")
model._make_predict_function()
graph = tf.get_default_graph()

categories = [
    'healthy', 'junk', 'dessert', 'appetizer', 'mains', 'soups', 'carbs', 'protein', 'fats', 'meat'
    ]
MEAN = np.array([51.072815, 51.072815, 51.072815])
STD = np.array([108.75629,  92.98068,  85.61884])

def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)
    # return the processed image
    return image
app = Flask(__name__)
@app.route("/",methods=["GET","POST"])
def predict():
    data = {"success":False}
    if request.method == "POST":
        if request.files.get("image"):
            # read the image in PIL format
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(224, 224))
            data["prediction"]=[]
            global graph
            with graph.as_default():
                prediction = np.round(model.predict(image)[0])
                labels = [categories[idx] for idx, current_prediction in enumerate(prediction) if current_prediction == 1]
                r = {"result":labels}
                data["prediction"].append(r)
                data["success"] = True
    return jsonify(data)
if __name__=="__main__":
    app.run(debug=True,host='0.0.0.0',port='8000')
# [“Soups”, “Mains”, “Appetizer”, “Dessert” ,“Protein”, “Fats”, “Carbs”, “Healthy”, “Junk”, “Meat”].