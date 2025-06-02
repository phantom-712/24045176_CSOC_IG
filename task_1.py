# importing necessary python modules
import pandas as pd
import matplotlib.pyplot as plt
import time

df = pd.read_csv(r"F:\ML_1\DATA\housing.csv\housing.csv")

# dropping the columns with null or missing values
df = df.dropna()


features = ['median_income', 'housing_median_age', 'total_rooms', 'total_bedrooms','population', 'households', 'latitude', 'longitude']
target = 'median_house_value'

X_full = df[features].copy()
X_full = (X_full - X_full.mean()) / X_full.std()
X_full.insert(0, 'bias', 1)

y_full = df[[target]].copy()

split_index = int(0.8 * len(X_full))
X_train = X_full.iloc[:split_index].copy()
X_val = X_full.iloc[split_index:].copy()
y_train = y_full.iloc[:split_index].copy()
y_val = y_full.iloc[split_index:].copy()

X_train_np = X_train.to_numpy()
y_train_np = y_train.to_numpy()
X_val_np = X_val.to_numpy()
y_val_np = y_val.to_numpy()

theta = pd.DataFrame([[0.0] for i in range(X_train.shape[1])])
theta_matrix = theta.to_numpy()

# Gradient Descent with Cost Tracking
def gradient_descent(X, y, theta, alpha, epochs):
    m = len(X)
    cost_history = []

    for i in range(epochs):
        predictions = X @ theta
        errors = predictions - y
        cost = ((errors ** 2).mean())  # MSE
        cost_history.append(cost)
        gradient = (X.T @ errors) / m
        theta = theta - alpha * gradient

    return theta, cost_history

alpha = 0.01
epochs = 10000
start_time = time.time()
theta_matrix, cost_history = gradient_descent(X_train_np, y_train_np, theta_matrix, alpha, epochs)

# printing the time taken for gradient descent to complete
end_time = time.time()
elapsed_time = end_time - start_time

print(f"\nTime taken for gradient descent to converge: {elapsed_time:.2f} seconds")

# passing the parameters in the dataframe for the model as theta
theta = pd.DataFrame(theta_matrix, index=X_train.columns, columns=['theta'])

# error metrics of the model with trained data
predictions_train = X_train_np @ theta_matrix
errors_train = predictions_train - y_train_np
mae_train = abs(errors_train).mean()
rmse_train = (errors_train ** 2).mean() ** 0.5
ss_total_train = ((y_train_np - y_train_np.mean()) ** 2).sum()
ss_residual_train = (errors_train ** 2).sum()
r2_train = 1 - (ss_residual_train / ss_total_train)
mse_train = (errors_train ** 2).mean()

print("\nLearned Parameters (theta):\n", theta)
print(f"\nTraining Set Metrics:")
print(f"Mean Absolute Error (MAE): {mae_train:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_train:.2f}")
print(f"Mean Squared Error (MSE): {mse_train:.2f}")
print(f"R^2 Score: {r2_train:.4f}")

# error metrics of the validation or the test dataset
predictions_val = X_val_np @ theta_matrix
errors_val = predictions_val - y_val_np
mae_val = abs(errors_val).mean()
rmse_val = (errors_val ** 2).mean() ** 0.5
ss_total_val = ((y_val_np - y_val_np.mean()) ** 2).sum()
ss_residual_val = (errors_val ** 2).sum()
r2_val = 1 - (ss_residual_val / ss_total_val)
mse_val = (errors_val ** 2).mean()

print(f"\nValidation Set Metrics:")
print(f"Mean Absolute Error (MAE): {mae_val:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_val:.2f}")
print(f"Mean Squared Error (MSE): {mse_val:.2f}")
print(f"R^2 Score: {r2_val:.4f}")


# plotting the cost function
plt.figure(figsize=(10, 5))
plt.plot(range(epochs), cost_history, label='Cost (MSE)', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Cost Function Convergence Over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()