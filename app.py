from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# Initialize Flask app
app = Flask(__name__)

# Load all required model files
model, feature_columns = joblib.load('best_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
unique_values = joblib.load('unique_values.pkl')
median_values = joblib.load('median_values.pkl')

# Load dataset for median values and column information
df = pd.read_csv('crop_yield.csv')


# Define route for home page (input form)
@app.route('/')
def home():
    try:
        return render_template('index.html',
                               crops=unique_values['crops'],
                               states=unique_values['states'],
                               seasons=unique_values['seasons'])
    except Exception as e:
        return render_template('index.html', error=f"Error loading options: {str(e)}")


# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get required form data
        crop = request.form.get('crop')
        state = request.form.get('state')
        season = request.form.get('season')

        # Get optional form data
        year = request.form.get('year')
        rainfall = request.form.get('rainfall')
        fertilizer = request.form.get('fertilizer')
        pesticide = request.form.get('pesticide')

        if not all([crop, state, season]):
            raise ValueError("Please fill in all required fields.")

        # Validate selections
        # Clean and validate inputs
        crop = crop.strip().title()
        state = state.strip().title()
        season = season.strip()

        if crop not in unique_values['crops']:
            raise ValueError(f"Invalid crop selection: {crop}")
        if state not in unique_values['states']:
            raise ValueError(f"Invalid state selection: {state}")
        if season not in unique_values['seasons']:
            raise ValueError(f"Invalid season selection: {season}")

        # Use optional values if provided, else fallback to median
        input_data = pd.DataFrame({
            'Crop': [crop],
            'State': [state],
            'Season': [season],
            'Crop_Year': [float(year) if year else median_values['Crop_Year']],
            'Annual_Rainfall': [float(rainfall) if rainfall else median_values['Annual_Rainfall']],
            'Fertilizer': [float(fertilizer) if fertilizer else median_values['Fertilizer']],
            'Pesticide': [float(pesticide) if pesticide else median_values['Pesticide']]
        })

        # Apply label encoders
        try:
            for column, encoder in label_encoders.items():
                if column in input_data.columns:
                    input_data[column] = encoder.transform(input_data[column])
        except ValueError:
            raise ValueError("Invalid input values. Please select from the available options.")

        # Ensure correct column order
        input_data = input_data[feature_columns]

        # Make prediction
        prediction = model.predict(input_data)
        pred_value = float(prediction[0]) if isinstance(prediction, (np.ndarray, list)) else prediction

        # Pass prediction to the result template
        return render_template('result.html',
                               crop=crop,
                               state=state,
                               season=season,
                               prediction=f"{pred_value:.2f}")

    except ValueError as e:
        return render_template('index.html',
                               error=str(e),
                               crops=unique_values['crops'],
                               states=unique_values['states'],
                               seasons=unique_values['seasons'])

    except Exception as e:
        print(f"Unexpected error in prediction: {str(e)}")
        return render_template('index.html',
                               error="An unexpected error occurred. Please try again.",
                               crops=unique_values['crops'],
                               states=unique_values['states'],
                               seasons=unique_values['seasons'])


# Run the Flask app
if __name__ == '__main__':
    # Check if required files exist and if static directory exists
    required_files = ['best_model.pkl', 'label_encoders.pkl', 'median_values.pkl', 'unique_values.pkl']
    if not all(os.path.exists(f) for f in required_files):
        print("Error: Required model files are missing. Please train the model first using main.py")
        exit(1)

    # Create static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')

    app.run(debug=True)
