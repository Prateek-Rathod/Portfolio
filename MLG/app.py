from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the model
try:
    with open("customer_churn_model .pkl", "rb") as f:
        model = pickle.load(f)
    loaded_model = model["model"]
    feature_names = model["features_names"]
except FileNotFoundError:
    raise Exception("Model file not found. Ensure 'customer_churn_model .pkl' exists in the same directory.")

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/form')
def form():
    """Render the input form."""
    return render_template('form.html')

@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction requests."""
    try:
        # Dynamically gather input data based on feature names
        input_data = {feature: request.form.get(feature) for feature in feature_names}
        
        # Validate numeric fields
        numeric_fields = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
        for field in numeric_fields:
            if not input_data[field] or not input_data[field].replace('.', '', 1).isdigit():
                return jsonify({"error": f"{field} must be a number."}), 400
            input_data[field] = float(input_data[field])

        # Convert input data into a DataFrame
        input_data_df = pd.DataFrame([input_data])

        # Load encoders and transform categorical data
        try:
            with open("encoders .pkl", "rb") as f:
                encoders = pickle.load(f)
            for column, encoder in encoders.items():
                if column in input_data_df.columns:
                    input_data_df[column] = encoder.transform(input_data_df[column])
        except FileNotFoundError:
            raise Exception("Encoders file not found. Ensure 'encoders .pkl' exists in the same directory.")

        # Make predictions
        prediction = loaded_model.predict(input_data_df)
        probability = loaded_model.predict_proba(input_data_df)[0][prediction[0]] * 100

        # Prepare output
        output = 'Not Satisfied' if prediction[0] == 1 else 'Satisfied'

        logging.info("Prediction successful: %s (%.2f%%)", output, probability)

        # Render result
        return render_template('result.html', prediction_text=f'Prediction: {output} (Probability: {probability:.2f}%)')

    except Exception as e:
        logging.error("Error during prediction: %s", e)
        return jsonify({"error": "An error occurred during prediction. Please check the input data."}), 500

if __name__ == "__main__":
    app.run(debug=True)
