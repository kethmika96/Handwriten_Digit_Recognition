from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow import keras
import cv2
import base64
import webview


app = Flask(__name__)

# Load  model
model = keras.models.load_model('mnist.h5')


@app.route('/', methods=['GET'])
def drawing():
    return render_template('drawing.html')


@app.route('/', methods=['POST'])
def canvas():

    canvasdata = request.form['canvasimg']
    encoded_data = request.form['canvasimg'].split(',')[1]


    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)


    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    gray_image = cv2.resize(gray_image, (28, 28), interpolation=cv2.INTER_LINEAR)


    img = np.expand_dims(gray_image, axis=0)

    try:
        prediction = np.argmax(model.predict(img))
        print(f"Prediction Result : {str(prediction)}")
        return render_template('drawing.html', response=str(prediction), canvasdata=canvasdata, success=True)
    except Exception as e:
        return render_template('drawing.html', response=str(e), canvasdata=canvasdata)