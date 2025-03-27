import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# Importing the dataset
df = pd.read_csv('crop_yield.csv')

df['Crop'] = df['Crop'].str.title()
df['State'] = df['State'].str.title()
df['Season'] = df['Season'].str.title()

pd.options.display.float_format = '{:.2f}'.format
df.describe(include=["int64","float64"]).T


df_copy = df.copy()
df_copy = df_copy.drop(['Area', 'Production'], axis=1)


category_columns = df_copy.select_dtypes(include=['object']).columns # select object type columns

df_copy = pd.get_dummies(df_copy, columns = category_columns, drop_first=True) # one hot encoding

boolean_cols_auto = df_copy.select_dtypes(include=['bool']).columns
df_copy[boolean_cols_auto] = df_copy[boolean_cols_auto].astype(int) # convert boolean to int


x = df_copy.drop(['Yield'], axis = 1)
y = df_copy['Yield']

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


evaluate_model_performance(
    model=LinearRegression(),
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

ax = df_model_sort.plot(
    x="Algorithms",
    y=["Training Score R2","Testing Score R2"],
    kind="bar",
    figsize=(15, 9),
    colormap="Set3",
    width=0.8
)

# Save the trained model for predictions
final_model = LinearRegression()
final_model.fit(x_train, y_train)

def predict_yield(model, new_data):
    """
    Makes yield predictions for new data using the trained model.
    
    Parameters:
    - model: Trained model
    - new_data: DataFrame with same features as training data (before encoding)
    
    Returns:
    - Predicted yield values
    """
    # Preprocess the new data the same way as training data
    # Drop Area and Production if they exist
    if 'Area' in new_data.columns:
        new_data = new_data.drop(['Area'], axis=1)
    if 'Production' in new_data.columns:
        new_data = new_data.drop(['Production'], axis=1)
    
    # Convert string columns to title case to match training data
    for col in new_data.select_dtypes(include=['object']).columns:
        new_data[col] = new_data[col].str.title()
    
    # Perform one-hot encoding on categorical variables
    cat_cols = new_data.select_dtypes(include=['object']).columns
    new_data_encoded = pd.get_dummies(new_data, columns=cat_cols, drop_first=True)
    
    # Make sure columns match training data
    missing_cols = set(x.columns) - set(new_data_encoded.columns)
    for col in missing_cols:
        new_data_encoded[col] = 0
    
    # Ensure the column order matches
    new_data_encoded = new_data_encoded[x.columns]
    
    # Make predictions
    predictions = model.predict(new_data_encoded)
    return predictions

# Example usage:
# Create a sample input for prediction
sample_data = pd.DataFrame({
    'Crop': ['Rice'],
    'State': ['Punjab'],
    'Season': ['Kharif'],
    # Add other required features
})

# Make prediction
predicted_yield = predict_yield(final_model, sample_data)
print(f"Predicted crop yield: {predicted_yield[0]:.2f}")

# First, let's examine what columns the model expects
print("Features the model was trained on:", x.columns.tolist())
print("Number of features:", len(x.columns))

# Create a more complete sample input
# Get a list of the original categorical columns before encoding
original_cat_columns = df.select_dtypes(include=['object']).columns
original_cat_columns = [col for col in original_cat_columns if col not in ['Area', 'Production', 'Yield']]

# Get a list of the numerical columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
numerical_columns = [col for col in numerical_columns if col not in ['Area', 'Production', 'Yield']]

# Fix: Remove .tolist() as these variables are already lists
print("Original categorical columns:", original_cat_columns)
print("Original numerical columns:", numerical_columns)

# Create a more complete sample input with reasonable values
sample_data = pd.DataFrame({
    'Crop': ['Rice'],
    'State': ['Punjab'],
    'Season': ['Kharif'],
})

# Add any missing numeric columns with reasonable values
# Use median values from the training data for numeric columns
for col in numerical_columns:
    if col not in sample_data.columns:
        sample_data[col] = df[col].median()

# Make prediction with improved data
predicted_yield = predict_yield(final_model, sample_data)

# Apply validation to ensure prediction is reasonable
if predicted_yield[0] < 0:
    print(f"Warning: Model predicted negative yield ({predicted_yield[0]:.2f}). Using 0 instead.")
    predicted_yield[0] = 0

print(f"Predicted crop yield: {predicted_yield[0]:.2f}")

# Optional: Save the trained model for future use
import pickle
with open('crop_yield_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)
print("Model saved as 'crop_yield_model.pkl'")