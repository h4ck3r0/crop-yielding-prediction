import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import joblib

# Load the dataset
df = pd.read_csv("crop_yield.csv")

# Clean and preprocess the dataset
df_cleaned = df[(df['Yield'] > 0.5) & (df['Yield'] < 20)].copy()
df_cleaned['Crop'] = df_cleaned['Crop'].str.title().str.strip()
df_cleaned['State'] = df_cleaned['State'].str.title().str.strip()
df_cleaned['Season'] = df_cleaned['Season'].str.title().str.strip()
df_cleaned['Fertilizer'] = np.log1p(df_cleaned['Fertilizer'])
df_cleaned['Pesticide'] = np.log1p(df_cleaned['Pesticide'])
df_cleaned['Annual_Rainfall'] = np.log1p(df_cleaned['Annual_Rainfall'])

# Drop unused columns
df_cleaned = df_cleaned.drop(['Area', 'Production'], axis=1)

# Handle missing values
df_cleaned['Fertilizer'] = df_cleaned['Fertilizer'].fillna(df_cleaned['Fertilizer'].median())
df_cleaned['Pesticide'] = df_cleaned['Pesticide'].fillna(df_cleaned['Pesticide'].median())
df_cleaned['Annual_Rainfall'] = df_cleaned['Annual_Rainfall'].fillna(df_cleaned['Annual_Rainfall'].median())


# Check for missing values
missing_values = df_cleaned.isnull().sum()
if missing_values.sum() > 0:
    print("Missing values found in the dataset:")
    print(missing_values[missing_values > 0])
else:
    print("No missing values found in the dataset.")

# Encode categorical columns
label_encoders = {}
for col in df_cleaned.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_cleaned[col] = le.fit_transform(df_cleaned[col])
    label_encoders[col] = le

# Save label encoders for later use
joblib.dump(label_encoders, 'label_encoders.pkl')

# Prepare features and target
x = df_cleaned.drop('Yield', axis=1)
y = df_cleaned['Yield']
feature_columns = x.columns.tolist()

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train models
models = ['Random Forest', 'Gradient Boosting']
model_list = [RandomForestRegressor(n_estimators=100, random_state=42), 
              GradientBoostingRegressor(n_estimators=100, random_state=42)]

training_scores_r2 = []
testing_scores_r2 = []
training_scores_adj_r2 = []
testing_scores_adj_r2 = []
training_scores_rmse = []
testing_scores_rmse = []

def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

# Evaluate models
for model in model_list:
    model.fit(x_train, y_train)
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    adj_r2_train = adjusted_r2(r2_train, x_train.shape[0], x_train.shape[1])
    adj_r2_test = adjusted_r2(r2_test, x_test.shape[0], x_test.shape[1])
    
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    training_scores_r2.append(r2_train)
    testing_scores_r2.append(r2_test)
    training_scores_adj_r2.append(adj_r2_train)
    testing_scores_adj_r2.append(adj_r2_test)
    training_scores_rmse.append(rmse_train)
    testing_scores_rmse.append(rmse_test)

# Create model performance DataFrame
df_model = pd.DataFrame({
    "Algorithms": models,
    "Training Score R2": training_scores_r2,
    "Training Score Adjusted R2": training_scores_adj_r2,
    "Training Score RMSE": training_scores_rmse,
    "Testing Score R2": testing_scores_r2,
    "Testing Score Adjusted R2": testing_scores_adj_r2,
    "Testing Score RMSE": testing_scores_rmse,
})

# Sort by best testing R2
df_model_sorted = df_model.sort_values(by="Testing Score R2", ascending=False)

# Select best model
best_model_name = df_model_sorted.iloc[0]['Algorithms']
best_model = model_list[models.index(best_model_name)]

# Save best mode
joblib.dump(best_model, 'mbest_model.pkl')

# Show model performance
df_model_sorted, best_model_name, feature_columns, best_model.feature_importances_ if hasattr(best_model, 'feature_importances_') else None

def predict_yield(model, new_data, show_debug=True):
    """
    Makes yield predictions using label encoding instead of one-hot encoding.
    """
    # Load label encoders
    label_encoders = joblib.load('label_encoders.pkl')
    
    # Preprocess the new data
    for col in new_data.select_dtypes(include=['object']).columns:
        new_data[col] = new_data[col].str.title().str.strip()
        
    if show_debug:
        print("\nInput data:")
        print(new_data)
        
    # Apply label encoding to categorical columns
    for column in new_data.columns:
        if column in label_encoders:
            try:
                new_data[column] = label_encoders[column].transform(new_data[column])
            except ValueError as e:
                print(f"Error encoding {column}: {e}")
                print(f"Valid values for {column}: {list(label_encoders[column].classes_)}")
                raise
    
    if show_debug:
        print("\nEncoded data:")
        print(new_data)
    
    # Ensure columns are in the same order as training data
    new_data = new_data[feature_columns]
    
    # Make prediction
    predictions = model.predict(new_data)
    
    return predictions

# Example usage: prediction for different crop combinations


# Get median values for numeric columns
median_values = {
    'Annual_Rainfall': df['Annual_Rainfall'].median(),
    'Fertilizer': df['Fertilizer'].median() if 'Fertilizer' in df else 0,
    'Pesticide': df['Pesticide'].median() if 'Pesticide' in df else 0
}



def interactive_prediction():
    print("\n=== Crop Yield Prediction ===")
    crop = input("Enter crop name: ")
    crop_year = int(input("Enter crop year (YYYY): "))
    state = input("Enter state name: ")
    season = input("Enter season: ")
    
    input_data = pd.DataFrame({
    'Crop': [crop],
    'State': [state],
    'Season': [season],
    'Crop_Year': [crop_year],
    'Annual_Rainfall': [1800],  # Try 1800mm
    'Fertilizer': [500000],     # Lower than 1.2M, but more realistic
    'Pesticide': [3000]
})
    
    try:
        best_model = joblib.load('mbest_model.pkl')  # <- load model here
        prediction = predict_yield(best_model, input_data)
        print(f"\nPredicted yield for {crop} in {state} during {season} season: {prediction[0]:.2f}")
    except Exception as e:
        print(f"Error making prediction: {e}")
        print("Please check your input and try again.")


interactive_prediction()

