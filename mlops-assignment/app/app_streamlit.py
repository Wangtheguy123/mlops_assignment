import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model
import os

# Get the base directory (one level up)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load models
models = {
    "Wheat Seeds Prediction": load_model(os.path.join(BASE_DIR, "xinrui/models/final_wheat_seeds_model")),
    "Melbourne House Price Prediction": load_model(os.path.join(BASE_DIR, "dekai/models/final_melbourne_model")),
    "Used Car Price Prediction": load_model(os.path.join(BASE_DIR, "jet/models/final_used_car_model"))
}

# Define feature sets
feature_sets = {
    "Wheat Seeds Prediction": ['Area', 'Perimeter', 'Compactness', 'Length', 'Width', 'AsymmetryCoeff', 'Groove'],
    "Melbourne House Price Prediction": ["Rooms", "Type", "Distance", "Bathroom", "Car", "Landsize", "CouncilArea", "Region", "Age"],
    "Used Car Price Prediction": ["Brand_Model", "Location", "Year", "Kilometers_Driven", "Fuel_Type", "Transmission",
                    "Owner_Type", "Mileage", "Engine", "Power", "Seats"]
}

# Title
st.title("Machine Learning Prediction App")

# Model selection
model_type = st.selectbox("Select a model", list(models.keys()))

if model_type:
    st.subheader(f"{model_type} Input Features")
    
    # Input fields for the selected model
    inputs = {}
    for feature in feature_sets[model_type]:
        inputs[feature] = st.text_input(feature, "")
    
    if st.button("Predict"):
        try:
            # Convert input values to the correct data type
            if model_type == "Wheat Seeds Prediction":
                input_values = [float(inputs[feature]) for feature in feature_sets[model_type]]
                input_df = pd.DataFrame([input_values], columns=feature_sets[model_type])
                prediction_df = predict_model(models[model_type], data=input_df)
                prediction = prediction_df['prediction_label'].iloc[0]
                
                wheat_mapping = {1: "Kama", 2: "Rosa", 3: "Canadian"}
                predicted_type = wheat_mapping.get(prediction, "Unknown")
                st.success(f"Predicted Wheat Type: {predicted_type}")
                
            elif model_type == "Melbourne House Price Prediction":
                form_data = {feature: float(inputs[feature]) if feature in ["Distance", "Landsize"] else int(inputs[feature]) if feature in ["Rooms", "Bathroom", "Car", "Age"] else inputs[feature] for feature in feature_sets[model_type]}
                input_df = pd.DataFrame([form_data])
                prediction_df = predict_model(models[model_type], data=input_df)
                predicted_price = round(prediction_df["prediction_label"].iloc[0], 2)
                st.success(f"Predicted House Price: ${predicted_price}")
                
            elif model_type == "Used Car Price Prediction":
                form_data = {feature: float(inputs[feature]) if feature in ["Mileage", "Power"] else int(inputs[feature]) if feature in ["Year", "Kilometers_Driven", "Engine", "Seats"] else inputs[feature] for feature in feature_sets[model_type]}
                input_df = pd.DataFrame([form_data])
                prediction_df = predict_model(models[model_type], data=input_df)
                predicted_price = round(prediction_df["prediction_label"].iloc[0], 2)
                st.success(f"Predicted Car Price: ${predicted_price}")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
