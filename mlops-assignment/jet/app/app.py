from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load("../models/final_used_car_model.pkl")

# Define the home route (renders the HTML form)
@app.route('/')
def home():
    return render_template("index.html")

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Convert data to DataFrame
        input_data = pd.DataFrame([form_data])

        # Ensure numeric values are converted properly
        for col in ["Year", "Kilometers_Driven", "Mileage", "Engine", "Power", "Seats"]:
            input_data[col] = pd.to_numeric(input_data[col])

        # Make prediction
        prediction = model.predict(input_data)

        # Return result
        return jsonify({"predicted_price": round(prediction[0], 2)})
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
