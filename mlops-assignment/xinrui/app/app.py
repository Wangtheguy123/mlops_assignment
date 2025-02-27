from flask import Flask, request, render_template
import pandas as pd
from pycaret.classification import load_model, predict_model
import os

app = Flask(__name__)

# Ensure the correct path to the model file
model_path = os.path.join(os.path.dirname(__file__), "final_wheat_seeds_model.pkl")

# Debug print to verify the path
print(f"Loading model from: {model_path}")

# Check if the model file exists before loading
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}. Ensure training was successful.")

# Load the saved model
model = load_model(model_path.replace(".pkl", ""))

print("Model loaded successfully!")


@app.route('/')
def home():
    return render_template('base.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Define the feature names as used in training
        features = ['Area', 'Perimeter', 'Compactness', 'Length', 'Width', 'AsymmetryCoeff', 'Groove']

        # Extract input values from the form and convert them to float
        input_values = [float(request.form.get(feature)) for feature in features]

        # Create a DataFrame for prediction
        input_df = pd.DataFrame([input_values], columns=features)

        # Generate prediction using the loaded model
        prediction_df = predict_model(model, data=input_df)

        # Extract numeric prediction and score
        numeric_prediction = prediction_df['prediction_label'].iloc[0] if 'prediction_label' in prediction_df.columns else None
        prediction_score = prediction_df['prediction_score'].iloc[0] if 'prediction_score' in prediction_df.columns else "N/A"

        # Map numeric prediction to wheat type
        wheat_mapping = {1: "Kama", 2: "Rosa", 3: "Canadian"}
        predicted_type = wheat_mapping.get(numeric_prediction, "Unknown")

        # Prepare a table (list of dictionaries) to display results
        prediction_table = [
            {"Parameter": "Area", "Value": input_df["Area"].iloc[0]},
            {"Parameter": "Perimeter", "Value": input_df["Perimeter"].iloc[0]},
            {"Parameter": "Compactness", "Value": input_df["Compactness"].iloc[0]},
            {"Parameter": "Length", "Value": input_df["Length"].iloc[0]},
            {"Parameter": "Width", "Value": input_df["Width"].iloc[0]},
            {"Parameter": "AsymmetryCoeff", "Value": input_df["AsymmetryCoeff"].iloc[0]},
            {"Parameter": "Groove", "Value": input_df["Groove"].iloc[0]},
            {"Parameter": "Predicted Wheat Type", "Value": predicted_type},
            {"Parameter": "Prediction Score", "Value": prediction_score},
        ]

        return render_template('base.html', prediction_table=prediction_table, prediction_text=f"Predicted Wheat Type: {predicted_type}")

    except Exception as e:
        # In case of an error, display the error message on the page
        return render_template('base.html', prediction_text=f'Error: {str(e)}')


if __name__ == '__main__':
    app.run(debug=True)