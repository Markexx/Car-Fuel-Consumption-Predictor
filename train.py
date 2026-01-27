import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from sklearn.neighbors import KNeighborsRegressor

df = pd.read_csv('auto-mpg.csv')
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df = df.dropna()

features = ['horsepower', 'weight', 'cylinders']
X = df[features]
y = df['mpg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

algorithms = [
    ("Linear Regression", LinearRegression()),
    ("Decision Tree", DecisionTreeRegressor(max_depth=5, random_state=42)),
    ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42)),
    ("K-Nearest Neighbors", KNeighborsRegressor(n_neighbors=5)),
    ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, random_state=42))
]

results = []
for name, model in algorithms:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Algorithm': name,
        'MAE': round(mae, 3),
        'R2': round(r2, 3),
        'Model': model
    })
    
    print(f"{name}: MAE={mae:.2f}, R2={r2:.3f}")

best_model = min(results, key=lambda x: x['MAE'])
print(f"\nNajbolji model: {best_model['Algorithm']} (MAE: {best_model['MAE']})")

joblib.dump(best_model['Model'], 'best_auto_mpg_model.pkl')
print("Model spremljen u best_auto_mpg_model.pkl")