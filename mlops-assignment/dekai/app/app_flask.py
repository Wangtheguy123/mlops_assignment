from flask import Flask, render_template, request
import pandas as pd
import pickle
from pycaret.regression import load_model, predict_model
import os

app = Flask(__name__)

# Ensure the correct path to the model file
model_path = os.path.join(os.path.dirname(__file__), "final_melbourne_model")

# Load the trained PyCaret model pipeline
model = load_model(model_path)  # Ensure the filename matches your saved model


@app.route('/')
def home():
    return render_template('dekais_app.html')


@app.route("/predict", methods=["GET", "POST"])
def dekais_predict():
    predicted_price = None  # Placeholder for prediction result

    if request.method == "POST":
        # Collect form inputs
        form_data = {
            "Rooms": int(request.form["Rooms"]),
            "Type": request.form["Type"],  # Dropdown, categorical
            "Distance": float(request.form["Distance"]),
            "Bathroom": int(request.form["Bathroom"]),
            "Car": int(request.form["Car"]),
            "Landsize": float(request.form["Landsize"]),
            "CouncilArea": request.form["CouncilArea"],  # Dropdown, categorical
            "Region": request.form["Region"],  # Dropdown, categorical
            "Age": int(request.form["Age"]),
        }

        # Convert form data to DataFrame for PyCaret
        input_df = pd.DataFrame([form_data])

        # Predict using the loaded model
        prediction = predict_model(model, data=input_df)
        predicted_price = round(prediction["prediction_label"][0], 2)
        print(predicted_price)

    return render_template("dekais_app.html", predicted_price=predicted_price)
