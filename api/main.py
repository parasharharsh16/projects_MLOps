# from flask import Flask

# app = Flask(__name__)


# @app.route("/")
# def hello_world():
#     return "<p>Hello, World!</p>"


import io
from flask import Flask, render_template, request,jsonify

import numpy as np
from joblib import load
from PIL import Image
import json

app = Flask(__name__)
model = load('models/best_model_C-1_gamma-0.001.pkl')

def preprocess_image(image):
    image = Image.open(image)
    image = image.convert('L')
    image = (image.resize((8, 8)))
    # Convert the image to a NumPy array
    image_array = np.array(image)
    #normalize image to sklearn format
    image_array = (image_array/ 16).astype(float)
    for i in range(8):
        for j in range(8):
            if image_array[i][j] >0:
                image_array[i][j] = 16 - image_array[i][j]
    n_samples = len(image_array )
    image_array = image_array.reshape((n_samples,-1))
    return image_array


@app.route("/")
def hello_world():
    return render_template("template.html")

@app.route('/upload', methods=['POST'])
def imagecompare():
    image1 = request.files["file1"]
    image2 = request.files["file2"]
    #if(str(type(image1))=="<class 'werkzeug.datastructures.file_storage.FileStorage'>"):
    # Preprocess the images and convert them to numpy arrays
    image1 = preprocess_image(image1)
    image2 = preprocess_image(image2)

    # Classify digits
    prediction1 = model.predict(image1.reshape(1,-1))[0]
    prediction2 = model.predict(image2.reshape(1,-1))[0]

    # Compare predictions
    result = (prediction1 == prediction2)
    return "<p>Image comparision returned " + str(result) +"</p>"

@app.route('/predict', methods=['POST'])
def predict():
    image = request.json
    image = json.loads(image)['image']
    img_array = np.array(image)
    try:
        result = model.predict([img_array])
        return jsonify({'result': str(result[0])})
    except:
        return jsonify({'error': 'Error Occured'})

@app.route("/sum/<x>/<y>")
def sum_num(x, y):
    x = int(x)
    y = int(y)
    return "<p>sum is " + str(x + y) + "</p>"

@app.route("/model", methods=["POST"])
def pred_model():
    js = request.get_json()
    x = js["x"]
    y = js["y"]
    x = int(x)
    y = int(y)
    return "<p>sum is " + str(x + y) + "</p>"


if __name__ == "__main__":
    #app.run(debug=True)
    app.run(host="0.0.0.0", port= 8000)
