import warnings
warnings.filterwarnings("ignore")

import numpy as np
import os
from PIL import Image
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import uuid

# Load the model
model_path = r"C:\Users\tinku2196\OneDrive\Desktop\MAJOR-PROJECT\Training\model_inception.h5"  
model = load_model(model_path)

template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'templates')

# Initialize the Flask app with the specified template folder
app = Flask(__name__, template_folder=template_dir)

# Define the route for the home page
@app.route("/")
def home():
    return render_template("index.html")

# Define the route for the about page
@app.route("/about")
def about():
    return render_template("about.html")

# Define the route for the details page
@app.route("/details")
def details():
    return render_template("predict.html")

# Define the route for handling the prediction
@app.route("/result", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # Get the uploaded image file
        f = request.files["image"]
        basepath = os.path.dirname(__file__)

        # Define the upload directory
        upload_dir = os.path.join(basepath, "uploads")

        # Ensure the directory exists
        os.makedirs(upload_dir, exist_ok=True)

        # Create a unique filename to avoid conflicts
        unique_filename = str(uuid.uuid4()) + "_" + f.filename
        filepath = os.path.join(upload_dir, unique_filename)
        f.save(filepath)

        # Open and preprocess the image
        image = Image.open(filepath)
        image = image.resize((224, 224))  # Resize to the expected input shape
        image = np.asarray(image)
        image = image.reshape(-1, 224, 224, 3) / 255.0

        # Predict the class of the image
        pred = np.argmax(model.predict(image))

        # Define the classes
        classes = [
            "Tomato__Bacterial_spot",
            "Tomato__Early_blight",
            "Tomato__Late_blight",
            "Tomato__Leaf_Mold",
            "Tomato__Septoria_leaf_spot",
            "Tomato__Spider_mites Two-spotted_spider_mite",
            "Tomato__Target_Spot",
            "Tomato__Tomato_Yellow_Leaf_Curl_Virus",
            "Tomato__Tomato_mosaic_virus",
            "Tomato__healthy",
        ]

        prediction = classes[pred]
        # Prepare the result message
        result_messages = {
            "Tomato__Bacterial_spot": "to have Bacterial spots.",
            "Tomato__Early_blight": "to have Early Blights.",
            "Tomato__healthy": "to be healthy.",
            "Tomato__Late_blight": "to have Late Blights.",
            "Tomato__Leaf_Mold": "to have Leaf Molds.",
            "Tomato__Septoria_leaf_spot": "to have Septoria Leaf Spot.",
            "Tomato__Spider_mites Two-spotted_spider_mite": "to have Spider Mites.",
            "Tomato__Target_Spot": "to have Target spots.",
            "Tomato__Tomato_mosaic_virus": "to have Tomato Mosaic Virus.",
            "Tomato__Tomato_Yellow_Leaf_Curl_Virus": "to have Tomato Yellow Leaf Curl Virus.",
        }

        # Format the prediction message
        prediction_message = f"Based on the image given as input, the AI model has predicted that it is to be {result_messages[prediction]}"
        return render_template("results.html", prediction_text=prediction_message)

    return render_template("predict.html")

# Run the app
if __name__ == "__main__":
    app.run(debug=True)