import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load dataset and model
df = pd.read_csv('crop_yield.csv')
df['Crop'] = df['Crop'].str.title().str.strip()
df['State'] = df['State'].str.title().str.strip()
df['Season'] = df['Season'].str.title().str.strip()
df['Fertilizer'] = np.log1p(df['Fertilizer'])
df['Pesticide'] = np.log1p(df['Pesticide'])
df['Annual_Rainfall'] = np.log1p(df['Annual_Rainfall'])

# Preprocessing
df_copy = df.copy()
df_copy = df_copy.drop(['Area', 'Production'], axis=1)
label_encoders = {}
category_columns = df_copy.select_dtypes(include=['object']).columns

for column in category_columns:
    label_encoders[column] = LabelEncoder()
    df_copy[column] = label_encoders[column].fit_transform(df_copy[column])

# Save the label encoders
joblib.dump(label_encoders, 'label_encoders.pkl')

# Features and target
x = df_copy.drop(['Yield'], axis=1)
y = df_copy['Yield']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train models and choose the best one
models = [LinearRegression(), RandomForestRegressor(n_estimators=100, random_state=42), GradientBoostingRegressor(n_estimators=100, random_state=42)]
best_model = None
best_r2 = -np.inf

for model in models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    if r2 > best_r2:
        best_model = model
        best_r2 = r2

# Save the best model
joblib.dump(best_model, 'best_model.pkl')

# Median values for prediction
median_values = {
    'Crop_Year': df['Crop_Year'].median(),
    'Annual_Rainfall': df['Annual_Rainfall'].median(),
    'Fertilizer': df['Fertilizer'].median() if 'Fertilizer' in df else 0,
    'Pesticide': df['Pesticide'].median() if 'Pesticide' in df else 0
}

# Define prediction function
def predict_yield(model, crop, state, season):
    input_data = pd.DataFrame({
        'Crop': [crop],
        'State': [state],
        'Season': [season],
        'Crop_Year': [median_values['Crop_Year']],
        'Annual_Rainfall': [median_values['Annual_Rainfall']],
        'Fertilizer': [median_values['Fertilizer']],
        'Pesticide': [median_values['Pesticide']]
    })
    return model.predict(input_data)[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    crop = request.form['crop']
    state = request.form['state']
    season = request.form['season']
    
    try:
        # Load the best model
        best_model = joblib.load('best_model.pkl')
        prediction = predict_yield(best_model, crop, state, season)
        return render_template('index.html', prediction=f"Predicted yield for {crop} in {state} during {season} season: {prediction:.2f} metric ton per hectare")
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
