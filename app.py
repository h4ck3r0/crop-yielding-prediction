from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# Initialize Flask app
app = Flask(__name__)

# Load model data, label encoders and unique values
model_data = joblib.load('best_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
try:
    unique_values = joblib.load('unique_values.pkl')
except:
    # Fallback to generating unique values from the dataset if file doesn't exist
    unique_values = {
        'crops': sorted(df['Crop'].unique().tolist()),
        'states': sorted(df['State'].unique().tolist()),
        'seasons': sorted(df['Season'].unique().tolist())
    }

# Load dataset for median values and column information
df = pd.read_csv('crop_yield.csv')

# Get median values for numeric columns
median_values = {
    'Crop_Year': df['Crop_Year'].median(),
    'Annual_Rainfall': df['Annual_Rainfall'].median(),
    'Fertilizer': df['Fertilizer'].median() if 'Fertilizer' in df else 0,
    'Pesticide': df['Pesticide'].median() if 'Pesticide' in df else 0
}


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
        # Get user input from form
        crop = request.form.get('crop')
        state = request.form.get('state')
        season = request.form.get('season')

        if not all([crop, state, season]):
            raise ValueError("Please fill in all fields")
        
        if crop not in df['Crop'].unique():
            raise ValueError(f"Invalid crop selection. Please choose from available options.")
        if state not in df['State'].unique():
            raise ValueError(f"Invalid state selection. Please choose from available options.")
        if season not in df['Season'].unique():
            raise ValueError(f"Invalid season selection. Please choose from available options.")

        # Prepare input data
        input_data = pd.DataFrame({
            'Crop': [crop],
            'State': [state],
            'Season': [season],
            'Crop_Year': [median_values['Crop_Year']],
            'Annual_Rainfall': [median_values['Annual_Rainfall']],
            'Fertilizer': [median_values['Fertilizer']],
            'Pesticide': [median_values['Pesticide']]
        })

        # Apply label encoding
        try:
            for column, encoder in label_encoders.items():
                if column in input_data.columns:
                    input_data[column] = encoder.transform(input_data[column])
        except ValueError:
            raise ValueError(f"Invalid input values. Please select from the available options.")

        # Reorder columns to match training data
        input_data = input_data[model_data['feature_names']]

        # Make prediction
        prediction = model_data['model'].predict(input_data)
        pred_value = prediction[0] if isinstance(prediction, np.ndarray) else prediction

        # Format prediction and handle any potential float conversion issues
        try:
            prediction_text = f"Predicted yield for {crop} in {state} during {season} season: {float(pred_value):.2f} metric ton per hectare"
        except:
            prediction_text = f"Predicted yield for {crop} in {state} during {season} season: {pred_value} metric ton per hectare"

        return render_template('index.html',
                           prediction=prediction_text,
                           crops=unique_values['crops'],
                           states=unique_values['states'],
                           seasons=unique_values['seasons'])

    except ValueError as e:
        # Handle validation errors
        return render_template('index.html',
                           error=str(e),
                           crops=unique_values['crops'],
                           states=unique_values['states'],
                           seasons=unique_values['seasons'])
    except Exception as e:
        # Log unexpected errors
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
        print("Error: Required model files are missing. Please train the model first using old.py")
        exit(1)

    # Create static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')

    app.run(debug=True)
