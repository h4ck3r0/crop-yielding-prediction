import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Importing the dataset
df = pd.read_csv('crop_yield.csv')

df['Crop'] = df['Crop'].str.title().str.strip()
df['State'] = df['State'].str.title().str.strip()
df['Season'] = df['Season'].str.title().str.strip()

# Print unique values in categorical columns
print("\nUnique values in categorical columns:")
for col in ['Crop', 'State', 'Season']:
    print(f"\n{col}:", df[col].unique())

pd.options.display.float_format = '{:.2f}'.format
df.describe(include=["int64","float64"]).T

df_copy = df.copy()
df_copy = df_copy.drop(['Area', 'Production'], axis=1)

# Initialize label encoders for categorical columns
label_encoders = {}
category_columns = df_copy.select_dtypes(include=['object']).columns

# Apply label encoding to categorical columns
for column in category_columns:
    label_encoders[column] = LabelEncoder()
    df_copy[column] = label_encoders[column].fit_transform(df_copy[column])

# Save label encoders for future use
joblib.dump(label_encoders, 'label_encoders.pkl')

x = df_copy.drop(['Yield'], axis=1)
y = df_copy['Yield']

# Store column order for prediction
feature_columns = x.columns.tolist()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# Initialize lists to store model performance metrics
models = []
training_scores_r2 = []
training_scores_adj_r2 = []
training_scores_rmse = []
testing_scores_r2 = []
testing_scores_adj_r2 = []
testing_scores_rmse = []

def evaluate_model_performance(model, x_train, y_train, x_test, y_test):
    """
    Evaluates R², Adjusted R², and RMSE of a given model on training and testing data.
    
    Parameters:
    - model: The machine learning model to evaluate
    - x_train: Training feature set
    - y_train: Training target values
    - x_test: Testing feature set
    - y_test: Testing target values
    """
    # Add model to the models list
    models.append(model.__class__.__name__)
    
    # Fit the model
    model.fit(x_train, y_train)
    
    # Predictions for training and testing data
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    # Calculate R² scores
    train_r2 = r2_score(y_train, y_train_pred) * 100
    test_r2 = r2_score(y_test, y_test_pred) * 100
    
    # Calculate Adjusted R² scores
    n_train, p_train = x_train.shape
    n_test, p_test = x_test.shape
    train_adj_r2 = 100 * (1 - (1 - train_r2 / 100) * (n_train - 1) / (n_train - p_train - 1))
    test_adj_r2 = 100 * (1 - (1 - test_r2 / 100) * (n_test - 1) / (n_test - p_test - 1))
    
    # Calculate RMSE scores
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Append scores to respective lists
    training_scores_r2.append(train_r2)
    training_scores_adj_r2.append(train_adj_r2)
    training_scores_rmse.append(train_rmse)
    testing_scores_r2.append(test_r2)
    testing_scores_adj_r2.append(test_adj_r2) 
    testing_scores_rmse.append(test_rmse) 
    
    # Display scores
    print(f"{model.__class__.__name__} Performance Metrics:")
    print(f"Training Data: R² = {train_r2:.2f}%, Adjusted R² = {train_adj_r2:.2f}%, RMSE = {train_rmse:.4f}")
    print(f"Testing Data : R² = {test_r2:.2f}%, Adjusted R² = {test_adj_r2:.2f}%, RMSE = {test_rmse:.4f}\n")

# List of models to try
model_list = [
    LinearRegression(),
    RandomForestRegressor(n_estimators=100, random_state=42),
    GradientBoostingRegressor(n_estimators=100, random_state=42)
]

# Train and evaluate each model
for model in model_list:
    evaluate_model_performance(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test
    )

df_model = pd.DataFrame(
        {"Algorithms": models,
         "Training Score R2": training_scores_r2,
         "Training Score Adjusted R2": training_scores_adj_r2,
         "Training Score RMSE": training_scores_rmse,
         "Testing Score R2": testing_scores_r2,
         "Testing Score Adjusted R2": testing_scores_adj_r2,
         "Testing Score RMSE": testing_scores_rmse,
        })
                   
df_model_sort = df_model.sort_values(by="Testing Score R2", ascending=False)
print(df_model_sort)

# Use the best performing model (highest Testing Score R2)
best_model_name = df_model_sort.iloc[0]['Algorithms']
print(f"Best model: {best_model_name}")

# Get the best model instance from the models you evaluated
best_model_idx = models.index(best_model_name)
best_models = [m for i, m in enumerate(model_list) if models[i] == best_model_name]
final_model = best_models[0] if best_models else model_list[0]  # Fallback to first model if not found

ax = df_model_sort.plot(
    x="Algorithms",
    y=["Training Score R2","Testing Score R2"],
    kind="bar",
    figsize=(15, 9),
    colormap="Set3",
    width=0.8
)

# Save the trained model for predictions
final_model.fit(x_train, y_train)

# For tree-based models like RandomForest, show feature importance
if hasattr(final_model, 'feature_importances_'):
    feature_importances = pd.DataFrame({
        'Feature': x.columns,
        'Importance': final_model.feature_importances_
    })
    print("\nFeature Importance:")
    print(feature_importances.sort_values('Importance', ascending=False).head(10))

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
print("\nPrediction Examples:")

# Get median values for numeric columns
median_values = {
    'Crop_Year': df['Crop_Year'].median(),
    'Annual_Rainfall': df['Annual_Rainfall'].median(),
    'Fertilizer': df['Fertilizer'].median() if 'Fertilizer' in df else 0,
    'Pesticide': df['Pesticide'].median() if 'Pesticide' in df else 0
}

# Example 1: Rice in Karnataka during Kharif season
sample1 = pd.DataFrame({
    'Crop': ['Rice'],
    'State': ['Karnataka'],
    'Season': ['Kharif'],
    'Crop_Year': [median_values['Crop_Year']],
    'Annual_Rainfall': [median_values['Annual_Rainfall']],
    'Fertilizer': [median_values['Fertilizer']],
    'Pesticide': [median_values['Pesticide']]
})
pred1 = predict_yield(final_model, sample1)
print(f"Rice, Karnataka, Kharif: {pred1[0]:.2f}")

# Example 2: Wheat in Punjab during Rabi season
sample2 = pd.DataFrame({
    'Crop': ['Wheat'],
    'State': ['Punjab'],
    'Season': ['Rabi'],
    'Crop_Year': [median_values['Crop_Year']],
    'Annual_Rainfall': [median_values['Annual_Rainfall']],
    'Fertilizer': [median_values['Fertilizer']],
    'Pesticide': [median_values['Pesticide']]
})
pred2 = predict_yield(final_model, sample2)
print(f"Wheat, Punjab, Rabi: {pred2[0]:.2f}")

# Example 3: Maize in Gujarat during Summer season
sample3 = pd.DataFrame({
    'Crop': ['Maize'], 
    'State': ['Gujarat'],
    'Season': ['Summer'],
    'Crop_Year': [median_values['Crop_Year']],
    'Annual_Rainfall': [median_values['Annual_Rainfall']],
    'Fertilizer': [median_values['Fertilizer']],
    'Pesticide': [median_values['Pesticide']]
})
pred3 = predict_yield(final_model, sample3)
print(f"Maize, Gujarat, Summer: {pred3[0]:.2f}")

# Function for interactive predictions
def interactive_prediction():
    print("\n=== Crop Yield Prediction ===")
    crop = input("Enter crop name: ")
    state = input("Enter state name: ")
    season = input("Enter season: ")
    
    input_data = pd.DataFrame({
        'Crop': [crop],
        'State': [state],
        'Season': [season],
        'Crop_Year': [median_values['Crop_Year']],
        'Annual_Rainfall': [median_values['Annual_Rainfall']],
        'Fertilizer': [median_values['Fertilizer']],
        'Pesticide': [median_values['Pesticide']]
    })
    
    try:
        prediction = predict_yield(final_model, input_data)
        print(f"\nPredicted yield for {crop} in {state} during {season} season: {prediction[0]:.2f}")
    except Exception as e:
        print(f"Error making prediction: {e}")
        print("Please check your input and try again.")

# Uncomment to use interactive predictions
# interactive_prediction()
