# from flask import Flask

# app = Flask(__name__)


# @app.route("/")
# def hello_world():
#     return "<p>Hello, World!</p>"


from flask import Flask, render_template, request

import numpy as np
from joblib import load
from PIL import Image

app = Flask(__name__)
model = load('models/best_model_C-1_gamma-0.001.pkl')
def preprocess_image(image):
    image = image.resize((8, 8))
    image = image.convert("L")
    # Convert the image to a NumPy array
    image_array = np.array(image)
    n_samples = len(image_array )
    image_array = image_array.reshape((n_samples,-1))
    return image_array


@app.route("/")
def hello_world():
    return render_template("template.html")

@app.route('/upload', methods=['POST'])
def imagecompare():
    image1 = Image.open(request.files["file1"])
    image2 = Image.open(request.files["file1"])

    # Preprocess the images and convert them to numpy arrays
    image1 = preprocess_image(image1)
    image2 = preprocess_image(image2)


    # Classify digits
    prediction1 = model.predict(image1.reshape(1, -1))[0]
    prediction2 = model.predict(image2.reshape(1, -1))[0]

    # Compare predictions
    result = (prediction1 == prediction2)
    return "<p>Image comparision returned " + str(result) + "</p>"

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
    app.run(host="0.0.0.0", port= 80)

