from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model, selector, selected feature names, and all feature names
with open('model_healthcheckpoints.pkl', 'rb') as file:
    model, selector, selected_feature_names, all_feature_names = pickle.load(file)

# Disease labels dictionary
disease_labels = {
    0: "Anemia",
    1: "Leukopenia",
    2: "Thrombocytopenia",
    3: "Leukocytosis",
    4: "Polycythemia",
    5: "Normal",
}

@app.route('/')
def index():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Log all form data for debugging
        print("Form Data Received:", request.form)

        # Initialize input_data dictionary for all features, setting default to 0 if missing
        input_data = {}
        for feature in all_feature_names:
            value = request.form.get(feature, "0")  # Default to "0" if feature is missing
            try:
                input_data[feature] = [float(value)]
            except ValueError:
                return f"Error: Invalid input for feature '{feature}' - expected a numeric value.", 400

        # Log the input data before processing
        print("Processed Input Data:", input_data)

        # Convert input data to DataFrame
        input_data_df = pd.DataFrame(input_data)

        # Transform the input data using the selector
        input_data_selected = selector.transform(input_data_df)

        # Make prediction
        prediction = model.predict(input_data_selected)
        disease = disease_labels.get(prediction[0], "Unknown")

        return render_template('output.html', disease=disease)

    except Exception as e:
        return f"Error: {e}", 400
if __name__ == '__main__':
    app.run(debug=True)
