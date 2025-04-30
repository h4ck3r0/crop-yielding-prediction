import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Importing the dataset
df = pd.read_csv('crop_yield.csv')

# Cleaning the data
# Clean the data - strip whitespace and standardize case
for column in ['Crop', 'State']:
    df[column] = df[column].str.strip().str.title()

# Special handling for Season to remove extra whitespace
df['Season'] = df['Season'].str.strip()

# Compute the Yield as Production / Area
df['Yield'] = df['Production'] / df['Area']

# Print unique values in categorical columns
print("\nUnique values in categorical columns:")
for column in ['Crop', 'State', 'Season']:
    print(f"\n{column}:", df[column].unique())


median_values = {
    'Crop_Year': df['Crop_Year'].median(),
    'Annual_Rainfall': df['Annual_Rainfall'].median(),
    'Fertilizer': df['Fertilizer'].median() if 'Fertilizer' in df.columns else 0,
    'Pesticide': df['Pesticide'].median() if 'Pesticide' in df.columns else 0,
}

# Save to file
joblib.dump(median_values, 'median_values.pkl')


# === Create unique values for dropdown options ===
unique_values = {
    'crops': sorted(df['Crop'].dropna().unique().tolist()),
    'states': sorted(df['State'].dropna().unique().tolist()),
    'seasons': sorted(df['Season'].dropna().unique().tolist()),
}

# Save to file
joblib.dump(unique_values, 'unique_values.pkl')

# Define the feature columns and target
feature_columns = ['Crop', 'State', 'Season', 'Crop_Year', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
target_column = 'Yield'

# Create a copy of the dataframe without NaN values in the target column (Yield)
data = df.dropna(subset=[target_column])

# Initialize label encoders for categorical columns
label_encoders = {}
for column in ['Crop', 'State', 'Season']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Save label encoders for future use
joblib.dump(label_encoders, 'label_encoders.pkl')

# Define features and target
x = data[feature_columns]
y = data[target_column]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize lists to store model performance metrics
models = []
training_scores_r2 = []
training_scores_adj_r2 = []
training_scores_rmse = []
testing_scores_r2 = []
testing_scores_adj_r2 = []
testing_scores_rmse = []

def evaluate_model_performance(model, x_train, y_train, x_test, y_test):
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

# Prepare a DataFrame for model performance
df_model = pd.DataFrame(
        {"Algorithms": models,
         "Training Score R2": training_scores_r2,
         "Training Score Adjusted R2": training_scores_adj_r2,
         "Training Score RMSE": training_scores_rmse,
         "Testing Score R2": testing_scores_r2,
         "Testing Score Adjusted R2": testing_scores_adj_r2,
         "Testing Score RMSE": testing_scores_rmse,
        })

# Sort models by Testing R² score
df_model_sort = df_model.sort_values(by="Testing Score R2", ascending=False)
print(df_model_sort)

# Get the best model
best_model_name = df_model_sort.iloc[0]['Algorithms']
print(f"Best model: {best_model_name}")
final_model = next(m for m in model_list if m.__class__.__name__ == best_model_name)

# Save the trained model for future predictions
joblib.dump((final_model, feature_columns), 'best_model.pkl')

# Save the label encoders for future use
joblib.dump(label_encoders, 'label_encoders.pkl')

# Interactive Prediction Function
def interactive_prediction():
    print("\n=== Crop Yield Prediction ===")
    print("Type 'exit' at any prompt to cancel prediction\n")

    # Load label encoders
    label_encoders = joblib.load('label_encoders.pkl')

    # Get available options for categorical columns
    available_crops = label_encoders['Crop'].classes_
    available_states = label_encoders['State'].classes_
    available_seasons = label_encoders['Season'].classes_

    def get_valid_input(prompt, options):
        while True:
            print(f"\nAvailable options: {', '.join(sorted(options))}")
            value = input(prompt).title().strip()
            if value.lower() == 'exit':
                return None
            if value in options:
                return value
            print(f"Invalid input. Please choose from the available options.")

    # Get categorical inputs with validation
    crop = get_valid_input("Enter crop name: ", available_crops)
    if crop is None:
        return

    state = get_valid_input("Enter state name: ", available_states)
    if state is None:
        return

    season = get_valid_input("Enter season: ", available_seasons)
    if season is None:
        return

    # Create a dictionary with features in the correct order
    input_dict = {}
    for col in feature_columns:
        if col == 'Crop':
            input_dict[col] = [label_encoders['Crop'].transform([crop])[0]]
        elif col == 'State':
            input_dict[col] = [label_encoders['State'].transform([state])[0]]
        elif col == 'Season':
            input_dict[col] = [label_encoders['Season'].transform([season])[0]]
        elif col == 'Crop_Year':
            input_dict[col] = [df['Crop_Year'].median()]
        elif col == 'Annual_Rainfall':
            input_dict[col] = [df['Annual_Rainfall'].median()]
        elif col == 'Fertilizer':
            input_dict[col] = [df['Fertilizer'].median() if 'Fertilizer' in df else 0]
        elif col == 'Pesticide':
            input_dict[col] = [df['Pesticide'].median() if 'Pesticide' in df else 0]

    # Create DataFrame with exact column order
    input_data = pd.DataFrame(input_dict)

    # Make prediction
    try:
        prediction = final_model.predict(input_data)
        if prediction is not None:
            print(f"\nPredicted yield for {crop} in {state} during {season} season:")
            print(f"  {prediction[0]:.2f} metric ton per hectare")
    except Exception as e:
        print(f"Error making prediction: {e}")
        print("Please check your input and try again.")

# Call interactive prediction
interactive_prediction()
