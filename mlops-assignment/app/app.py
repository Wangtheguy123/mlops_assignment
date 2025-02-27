import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model
import os

# Load models
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
models = {
    "predict_xr": load_model(os.path.join(BASE_DIR, "xinrui/models/final_wheat_seeds_model")),
    "predict_dk": load_model(os.path.join(BASE_DIR, "dekai/models/final_melbourne_model"))
}

# Define feature sets
feature_sets = {
    "predict_xr": ['Area', 'Perimeter', 'Compactness', 'Length', 'Width', 'AsymmetryCoeff', 'Groove'],
    "predict_dk": ["Rooms", "Type", "Distance", "Bathroom", "Car", "Landsize", "CouncilArea", "Region", "Age"]
}

# Model mapping
model_mapping = {
    "predict_xr": "Wheat Seeds Prediction",
    "predict_dk": "Melbourne House Price Prediction"
}

# Streamlit app
st.title("Dekais App")

# Model selection
selected_model = st.selectbox("Choose a Model", list(model_mapping.values()))

if selected_model == "Wheat Seeds Prediction":
    st.header("Wheat Seeds Prediction")
    with st.form("wheat_form"):
        area = st.number_input("Area", min_value=0.0)
        perimeter = st.number_input("Perimeter", min_value=0.0)
        compactness = st.number_input("Compactness", min_value=0.0)
        length = st.number_input("Length", min_value=0.0)
        width = st.number_input("Width", min_value=0.0)
        asymmetry_coeff = st.number_input("Asymmetry Coefficient", min_value=0.0)
        groove = st.number_input("Groove", min_value=0.0)
        submitted = st.form_submit_button("Predict")

        if submitted:
            input_data = pd.DataFrame([[area, perimeter, compactness, length, width, asymmetry_coeff, groove]],
                                      columns=feature_sets["predict_xr"])
            prediction = predict_model(models["predict_xr"], data=input_data)
            predicted_type = {1: "Kama", 2: "Rosa", 3: "Canadian"}.get(prediction['prediction_label'].iloc[0], "Unknown")
            st.success(f"Predicted Wheat Type: {predicted_type}")

elif selected_model == "Melbourne House Price Prediction":
    st.header("Melbourne House Price Prediction")
    with st.form("melbourne_form"):
        rooms = st.number_input("Rooms", min_value=1)
        type_ = st.selectbox("Type", ["t", "h", "u"])
        distance = st.number_input("Distance (km)", min_value=0.0)
        bathroom = st.number_input("Bathroom", min_value=1)
        car = st.number_input("Car", min_value=0)
        landsize = st.number_input("Landsize (sqm)", min_value=0.0)
        council_area = st.selectbox("Council Area", ["Moonee Valley", "Port Phillip", "Darebin", "Yarra", "Hobsons Bay", "Stonnington", "Boroondara", "Monash", "Glen Eira", "Whitehorse", "Maribyrnong", "Bayside", "Moreland", "Manningham", "Banyule", "Melbourne", "Kingston", "Brimbank", "Hume", "Knox", "Maroondah", "Casey", "Melton", "Greater Dandenong", "Nillumbik", "Whittlesea", "Frankston", "Macedon Ranges", "Yarra Ranges", "Wyndham", "Moorabool", "Cardinia", "Unavailable"])
        region = st.selectbox("Region", ["Western Metropolitan", "Southern Metropolitan", "Northern Metropolitan", "Eastern Metropolitan", "South-Eastern Metropolitan", "Eastern Victoria", "Northern Victoria", "Western Victoria"])
        age = st.number_input("Age", min_value=0)
        submitted = st.form_submit_button("Predict")

        if submitted:
            input_data = pd.DataFrame([[rooms, type_, distance, bathroom, car, landsize, council_area, region, age]],
                                      columns=feature_sets["predict_dk"])
            prediction = predict_model(models["predict_dk"], data=input_data)
            predicted_price = round(prediction['prediction_label'].iloc[0], 2)
            st.success(f"Predicted Price: ${predicted_price}")