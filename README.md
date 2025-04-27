# 🌾 Crop Yield Prediction using Machine Learning

This project uses machine learning models to predict agricultural crop yields based on environmental and input-related features. The aim is to assist farmers and policymakers in understanding potential production outcomes and planning accordingly.

---

## 📊 Dataset Overview

The dataset (`crop_yield.csv`) contains the following columns:

- `State`: Indian state
- `Crop`: Name of the crop
- `Season`: Season (e.g., Kharif, Rabi)
- `Crop_Year`: Year of crop data
- `Annual_Rainfall`: Total rainfall received
- `Fertilizer`: Amount of fertilizer used (transformed using log)
- `Pesticide`: Amount of pesticide used (transformed using log)
- `Area`, `Production`: Used internally for cleaning
- `Yield`: Crop production per unit area (target variable)

---

## 🛠️ Tech Stack

- **Python** (core language)
- **Pandas & NumPy** (data manipulation)
- **Scikit-learn** (ML models, preprocessing, evaluation)
- **Matplotlib** (visualizations)
- **Joblib** (model persistence)

---

## 🤖 Models Implemented

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Each model is trained, evaluated, and compared using R² Score, Adjusted R² Score, and RMSE (Root Mean Squared Error). The best-performing model is automatically selected for prediction.

---

## 📈 Evaluation Metrics

| Metric        | Description                                |
|---------------|--------------------------------------------|
| R² Score      | Measures how well predictions match actual |
| Adjusted R²   | Adjusted for the number of predictors      |
| RMSE          | Root of mean squared error in predictions  |

A bar chart of training vs. testing R² scores is also generated.

---

## 🚀 Usage

### 1. Install Requirements

```bash
pip install numpy pandas scikit-learn matplotlib joblib
```

### 2. Prepare Dataset

Place `crop_yield.csv` in the root directory. Ensure it contains the required columns.

### 3. Run Script

```bash
python crop_yield_prediction.py
```

This will:
- Preprocess the dataset
- Train all models
- Evaluate and select the best model
- Save the encoders and final model
- Launch an interactive CLI for yield prediction

---

## 💬 Prediction Example

```text
=== Crop Yield Prediction ===
Enter crop name: Rice
Enter state name: Tamil Nadu
Enter season: Kharif

Predicted yield for Rice in Tamil Nadu during Kharif season: 2.89 Metric Ton Per Unit Area
```

---

## 🧾 Output Files

- `label_encoders.pkl`: Serialized label encoders for future use
- `best_crop_yield_model.pkl`: Saved best ML model

---

## 👤 Author

**Raj Aryan**  
🎓 B.Tech @ RNSIT  
🔗 [LinkedIn](https://www.linkedin.com/in/h4ck3r0)  
💻 [GitHub](https://github.com/h4ck3r0)

---

## 📄 License

Licensed under the [MIT License](LICENSE). Feel free to use and modify.
