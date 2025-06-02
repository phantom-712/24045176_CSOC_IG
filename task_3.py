import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load and clean data
df = pd.read_csv(r"F:\ML_1\DATA\housing.csv\housing.csv").dropna()

# Define features and target
features = ['median_income', 'housing_median_age', 'total_rooms', 'total_bedrooms',
            'population', 'households', 'latitude', 'longitude']
target = 'median_house_value'

# Prepare data
X = df[features].values
y = df[target].values.reshape(-1, 1)

# Standardize features
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_stdized = (X - X_mean) / X_std

# Train-validation split
split_index = int(0.8 * len(X_stdized))
X_train, X_val = X_stdized[:split_index], X_stdized[split_index:]
y_train, y_val = y[:split_index], y[split_index:]

# Train scikit-learn Linear Regression
start_time = time.time()
model = LinearRegression()
model.fit(X_train, y_train)
end_time = time.time()

elapsed_time = end_time - start_time

# Predictions
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

# Training metrics
mae_train = mean_absolute_error(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_train_pred)

# Validation metrics
mae_val = mean_absolute_error(y_val, y_val_pred)
mse_val = mean_squared_error(y_val, y_val_pred)
rmse_val = np.sqrt(mse_val)
r2_val = r2_score(y_val, y_val_pred)

# Output
print(f"\nTime taken by scikit-learn LinearRegression: {elapsed_time:.4f} seconds")
print("\nTraining Set Metrics (scikit-learn):")
print(f"MAE: {mae_train:.2f}, MSE: {mse_train:.2f}, RMSE: {rmse_train:.2f}, R²: {r2_train:.4f}")
print("\nValidation Set Metrics (scikit-learn):")
print(f"MAE: {mae_val:.2f}, MSE: {mse_val:.2f}, RMSE: {rmse_val:.2f}, R²: {r2_val:.4f}")

# Predict on new data point
new_point = np.array([[3.0, 30.0, 2000.0, 400.0, 1000.0, 500.0, 34.0, -118.0]])
new_point_std = (new_point - X_mean) / X_std
prediction = model.predict(new_point_std)
print(f"\nPredicted Median House Value: ${prediction[0, 0]:.2f}")
