from flask import Flask, render_template, request
import jsonify
import requests
import pickle

import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from keras.models import model_from_json
import pandas as pd
from sklearn.preprocessing import LabelEncoder

flask_app = Flask(__name__)

# Load model architecture from JSON file
with open("rnn_model.json", "r") as json_file:
    loaded_model_json = json_file.read()

# Create model from loaded architecture
model = model_from_json(loaded_model_json)

# Load weights into the model
model.load_weights("rnn_model_weights.weights.h5")

# Define a function to make predictions
def predict_class(reshaped_features):
    prediction = model.predict(reshaped_features)
    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(prediction)
    # Map the class index to the corresponding label (assuming you have a list of class labels)
    class_labels = ["Class 0", "Class 1", "Class 2", "Class 3"]  # Replace with your actual class labels
    result = class_labels[predicted_class_index]
    return result

@flask_app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@flask_app.route("/predict", methods=['POST'])
def predict():
    # Extract features from form input
    features = [float(x) for x in request.form.values()]
        
    # Convert features to numpy array
    features_array = np.array(features)
    
    # Reshape features for LSTM input
    reshaped_features = np.reshape(features_array, (1, features_array.shape[0], 1))
    
    # Make prediction
    prediction = predict_class(reshaped_features)
    
    # Format prediction text
    prediction_text = "The Predicted Class is {} Node Status".format(prediction)
    
    return render_template("index.html", prediction_text=prediction_text)

if __name__=="__main__":
    flask_app.run(debug=True)
