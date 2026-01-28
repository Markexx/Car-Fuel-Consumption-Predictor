# 🚗 Auto MPG Prediction Using Machine Learning

This project focuses on predicting automobile fuel consumption expressed in **Miles Per Gallon (MPG)** using **machine learning regression algorithms**.  
The solution includes **data analysis, model training and evaluation, a REST API**, and a **web-based client application**.

The project was developed as part of the course **Service Computing and Data Analysis**  
Faculty of Electrical Engineering, Computer Science and Information Technology Osijek (FERIT)

---

## 📊 Problem Description

Fuel consumption prediction is a classical **regression problem** in machine learning.  
The goal is to estimate MPG based on technical characteristics of a vehicle without performing real-world testing.

The model predicts MPG using the following features:

- Engine horsepower
- Vehicle weight
- Number of cylinders

---

## 📂 Dataset

The project uses the **Auto MPG dataset** from the **UCI Machine Learning Repository**.

- Samples: 398 (396 after cleaning)
- Time period: 1970–1982
- Format: CSV

### Used Features

| Feature     | Description                  |
|------------|------------------------------|
| mpg        | Fuel consumption (target)    |
| horsepower | Engine power (HP)            |
| weight     | Vehicle weight (lbs)         |
| cylinders  | Number of cylinders          |

Missing values in the `horsepower` column were converted to `NaN` and removed.

---

## 🔎 Data Analysis

### Descriptive Statistics
- Mean, standard deviation, quartiles, and value ranges.

### Correlation Analysis
Strong negative correlations with MPG:

- `weight`: **−0.833**
- `horsepower`: **−0.778**
- `cylinders`: **−0.776**

### Inferential Statistics
Pearson correlation tests with p-values confirmed **statistical significance**:

- p-values < 0.001
- Null hypothesis rejected for all selected features

This validates feature selection for model training.

---

## 🤖 Machine Learning Models

The following regression models were implemented using **scikit-learn**:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- K-Nearest Neighbors (KNN)
- Gradient Boosting Regressor

### Evaluation Metrics

- **MAE** – Mean Absolute Error
- **R²** – Coefficient of Determination

### Model Comparison

| Algorithm             | MAE  | R²    |
|----------------------|------|-------|
| Linear Regression     | 3.53 | 0.623 |
| Decision Tree         | 3.26 | 0.568 |
| Random Forest         | 3.04 | 0.643 |
| KNN                   | 3.17 | 0.650 |
| **Gradient Boosting** | **2.98** | **0.672** |

✅ **Gradient Boosting Regressor** achieved the best performance and was selected as the final model.

---

## ⚙️ Preprocessing

- Missing values removed
- Feature selection based on correlation analysis
- **Standardization (`StandardScaler`) applied for scale-sensitive algorithms (KNN, SVR)**

---

## 🌐 REST API

The trained model is exposed via a **Flask REST API**.

### Endpoints

- `POST /predict` – returns MPG prediction
- `GET /health` – API health check

### Example Request

```json
{
  "data": [[150, 3000, 6]]
}
```

API runs on port 5001 with CORS enabled.

## 🧪 System Testing
| Vehicle Type | HP  | Weight (lbs) | Cylinders | Predicted MPG |
| ------------ | --- | ------------ | --------- | ------------- |
| Economy Car  | 85  | 2500         | 4         | 26.1          |
| Family Car   | 150 | 3000         | 6         | 18.1          |
| SUV          | 200 | 4500         | 8         | 11.5          |
| Sports Car   | 300 | 3500         | 8         | 15.2          |

## 📁 Project Structure

```text
├── venv/
├── train.py
├── app.py
├── simple_app.html
├── auto_mpg_prediction.ipynb
├── environment.yml
├── score.py
├── config.json
├── auto-mpg.csv
├── requirements.txt
├── best_auto_mpg_model.pkl
└── README.md


