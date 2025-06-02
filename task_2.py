import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# Step 1: Load and preprocess data
df = pd.read_csv(r"F:\ML_1\DATA\housing.csv\housing.csv")
df=df.dropna()

features = ['median_income', 'housing_median_age', 'total_rooms', 'total_bedrooms',
            'population', 'households', 'latitude', 'longitude']
target = 'median_house_value'

X = df[features].values
y = df[[target]].values

# Standardize features
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

# Add bias column
X = np.hstack([np.ones((X.shape[0], 1)), X])

# Split data: 80% train, 20% validation
split_index = int(0.8 * X.shape[0])
X_train, X_val = X[:split_index], X[split_index:]
y_train, y_val = y[:split_index], y[split_index:]

# Step 2: Gradient descent implementation
def compute_cost(X, y, theta):
    m = X.shape[0]
    errors = X @ theta - y
    return (1 / (2 * m)) * np.sum(errors ** 2)

def gradient_descent(X, y, alpha, epochs):
    m, n = X.shape
    theta = np.zeros((n, 1))
    cost_history = []

    for i in range(epochs):
        predictions = X @ theta
        errors = predictions - y
        gradients = (1 / m) * X.T @ errors
        theta -= alpha * gradients
        cost = (errors ** 2).mean()
        cost_history.append(cost)
    
    return theta, cost_history

# Step 3: Train the model
alpha = 0.01
epochs = 10000
start_time = time.time()
theta, cost_history = gradient_descent(X_train, y_train, alpha, epochs)
end_time = time.time()

print(f"\nTime taken for gradient descent to converge: {end_time - start_time:.2f} seconds")

# Step 4: Evaluation metrics
def evaluate(X, y, theta):
    predictions = X @ theta
    errors = predictions - y
    mae = np.mean(np.abs(errors))
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - (np.sum(errors ** 2) / np.sum((y - y.mean()) ** 2))
    return mae, mse, rmse, r2

mae_train, mse_train, rmse_train, r2_train = evaluate(X_train, y_train, theta)
mae_val, mse_val, rmse_val, r2_val = evaluate(X_val, y_val, theta)

print("\nTraining Set Metrics:")
print(f"MAE: {mae_train:.2f}, MSE: {mse_train:.2f}, RMSE: {rmse_train:.2f}, R²: {r2_train:.4f}")

print("\nValidation Set Metrics:")
print(f"MAE: {mae_val:.2f}, MSE: {mse_val:.2f}, RMSE: {rmse_val:.2f}, R²: {r2_val:.4f}")

# Step 5: Prediction for new data
new_point = np.array([[3.0, 30.0, 2000.0, 400.0, 1000.0, 500.0, 34.0, -118.0]])
new_point = (new_point - X_mean) / X_std
new_point = np.hstack([np.ones((1, 1)), new_point])
prediction = new_point @ theta

print(f"\nPredicted Median House Value: ${prediction[0, 0]:.2f}")

# Step 6: Plot cost convergence
plt.figure(figsize=(10, 5))
plt.plot(range(epochs), cost_history, color='blue')
plt.title("Cost Function Convergence Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.grid(True)
plt.tight_layout()
plt.show()
