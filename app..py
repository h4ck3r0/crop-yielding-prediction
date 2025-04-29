from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained model and label encoders
final_model, feature_columns = joblib.load('best_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Load median values
median_values = {
    'Crop_Year': 2020,  # Update if you want to calculate dynamically
    'Annual_Rainfall': 5.5,  # example log1p transformed median
    'Fertilizer': 6.2,       
    'Pesticide': 4.8         
}

# Home route: form input
@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get user input
        crop = request.form.get('crop').title().strip()
        state = request.form.get('state').title().strip()
        season = request.form.get('season').title().strip()

        # Optional fields
        rainfall = request.form.get('rainfall')
        fertilizer = request.form.get('fertilizer')
        pesticide = request.form.get('pesticide')

        try:
            input_dict = {}

            for col in feature_columns:
                if col == 'Crop':
                    input_dict[col] = [label_encoders['Crop'].transform([crop])[0]]
                elif col == 'State':
                    input_dict[col] = [label_encoders['State'].transform([state])[0]]
                elif col == 'Season':
                    input_dict[col] = [label_encoders['Season'].transform([season])[0]]
                elif col == 'Crop_Year':
                    input_dict[col] = [median_values['Crop_Year']]
                elif col == 'Annual_Rainfall':
                    val = float(rainfall) if rainfall else np.expm1(median_values['Annual_Rainfall'])
                    input_dict[col] = [np.log1p(val)]
                elif col == 'Fertilizer':
                    val = float(fertilizer) if fertilizer else np.expm1(median_values['Fertilizer'])
                    input_dict[col] = [np.log1p(val)]
                elif col == 'Pesticide':
                    val = float(pesticide) if pesticide else np.expm1(median_values['Pesticide'])
                    input_dict[col] = [np.log1p(val)]

            # Create input dataframe
            input_data = pd.DataFrame(input_dict)[feature_columns]

            # Make prediction
            prediction = final_model.predict(input_data)[0]

            return render_template('result.html', prediction=prediction, crop=crop, state=state, season=season)

        except Exception as e:
            return f"Error occurred: {e}"

    # GET method: show form
    crops = sorted(label_encoders['Crop'].classes_)
    states = sorted(label_encoders['State'].classes_)
    seasons = sorted(label_encoders['Season'].classes_)
    return render_template('index.html', crops=crops, states=states, seasons=seasons)

# Run app
if __name__ == '__main__':
    app.run(debug=True)
