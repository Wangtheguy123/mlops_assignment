import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model
import os

# Set the title of the app
st.title("Melbourne House Price Prediction")

# Ensure the correct path to the model file
model_path = os.path.join(os.path.dirname(__file__), "final_melbourne_model")

# Load the trained PyCaret model pipeline
model = load_model(model_path)  # Ensure the filename matches your saved model

# Input form
with st.form("prediction_form"):
    st.header("Enter House Details")

    # Collect form inputs
    rooms = st.number_input("Rooms", min_value=1, value=2)
    type_ = st.selectbox("Type", ["h", "u", "t"])  # h = house, u = unit, t = townhouse
    distance = st.number_input("Distance (km)", min_value=0.0, value=5.0)
    bathroom = st.number_input("Bathroom", min_value=1, value=1)
    car = st.number_input("Car", min_value=0, value=1)
    landsize = st.number_input("Landsize (sqm)", min_value=0.0, value=300.0)
    council_area = st.selectbox(
        "Council Area",
        [
            "Moonee Valley", "Port Phillip", "Darebin", "Yarra", "Hobsons Bay", "Stonnington",
            "Boroondara", "Monash", "Glen Eira", "Whitehorse", "Maribyrnong", "Bayside",
            "Moreland", "Manningham", "Banyule", "Melbourne", "Kingston", "Brimbank", "Hume",
            "Knox", "Maroondah", "Casey", "Melton", "Greater Dandenong", "Nillumbik",
            "Whittlesea", "Frankston", "Macedon Ranges", "Yarra Ranges", "Wyndham", "Moorabool",
            "Cardinia", "Unavailable"
        ]
    )
    region = st.selectbox(
        "Region",
        [
            "Western Metropolitan", "Southern Metropolitan", "Northern Metropolitan",
            "Eastern Metropolitan", "South-Eastern Metropolitan", "Eastern Victoria",
            "Northern Victoria", "Western Victoria"
        ]
    )
    age = st.number_input("Age", min_value=0, value=10)

    # Submit button
    submitted = st.form_submit_button("Predict")

    if submitted:
        # Prepare the input data as a dictionary
        form_data = {
            "Rooms": rooms,
            "Type": type_,
            "Distance": distance,
            "Bathroom": bathroom,
            "Car": car,
            "Landsize": landsize,
            "CouncilArea": council_area,
            "Region": region,
            "Age": age,
        }

        # Convert form data to DataFrame for PyCaret
        input_df = pd.DataFrame([form_data])

        # Predict using the loaded model
        prediction = predict_model(model, data=input_df)
        predicted_price = round(prediction["prediction_label"][0], 2)

        # Display the prediction
        st.success(f"Predicted Price: ${predicted_price}")