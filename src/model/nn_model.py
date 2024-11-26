import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Lasso, Ridge
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

# โหลดข้อมูล
df = pd.read_csv('../dataset/real_dataset.csv')

# สร้าง Target Variables (D+1, D+3, D+7)
df['PM2.5(D+1)'] = df['PM2.5'].shift(-1)
df['PM2.5(D+3)'] = df['PM2.5'].shift(-3)
df['PM2.5(D+7)'] = df['PM2.5'].shift(-7)

# ลบข้อมูลที่มีค่า NaN
df = df.dropna(subset=['PM2.5(D+1)', 'PM2.5(D+3)', 'PM2.5(D+7)'])

# ฟังก์ชันเพิ่ม Gaussian Noise
def add_noise(df, columns, noise_level=0.05):
    augmented_data = df.copy()
    for col in columns:
        if col in augmented_data.columns:
            noise = np.random.normal(0, noise_level * augmented_data[col].std(), size=len(augmented_data))
            augmented_data[col] = augmented_data[col] + noise
    return augmented_data

# เพิ่ม noise
columns_to_augment = ['PM2.5', 'PM2.5(D-1)', 'PM2.5(D-3)', 'PM2.5(D-7)', 'PM2.5(D+1)', 'PM2.5(D+3)', 'PM2.5(D+7)']
augmented_df = add_noise(df, columns_to_augment)

# รวมข้อมูลเดิมและข้อมูลที่เพิ่มขึ้น
final_df = pd.concat([df, augmented_df], ignore_index=True)


# เตรียม Features และ Targets
X = final_df[['PM2.5', 'PM2.5(D-1)', 'PM2.5(D-3)', 'PM2.5(D-7)','Season_Summer', 'Season_Rainy', 'Season_Winter']]
y_1 = final_df['PM2.5(D+1)']
y_3 = final_df['PM2.5(D+3)']
y_7 = final_df['PM2.5(D+7)']

split_index = int(len(final_df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train_1, y_test_1 = y_1.iloc[:split_index], y_1.iloc[split_index:]
y_train_3, y_test_3 = y_3.iloc[:split_index], y_3.iloc[split_index:]
y_train_7, y_test_7 = y_7.iloc[:split_index], y_7.iloc[split_index:]

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nn_model_1 = MLPRegressor(hidden_layer_sizes=(64, 128), activation='tanh', max_iter=2000, learning_rate_init=0.01, learning_rate='adaptive')

lasso_model = Lasso(alpha=0.01)
lasso_model.fit(X_train_scaled, y_train_1)

ridge_model = Ridge(alpha=0.01)
ridge_model.fit(X_train_scaled, y_train_1)

nn_model_3 = MLPRegressor(hidden_layer_sizes=(64, 128), activation='tanh', max_iter=2000, learning_rate_init=0.01, learning_rate='adaptive')

lasso_model = Lasso(alpha=0.01)
lasso_model.fit(X_train_scaled, y_train_3)

ridge_model = Ridge(alpha=0.01)
ridge_model.fit(X_train_scaled, y_train_3)

nn_model_7 = MLPRegressor(hidden_layer_sizes=(64, 128), activation='tanh', max_iter=2000, learning_rate_init=0.01, learning_rate='adaptive')

lasso_model = Lasso(alpha=0.01)
lasso_model.fit(X_train_scaled, y_train_7)

ridge_model = Ridge(alpha=0.01)
ridge_model.fit(X_train_scaled, y_train_7)

nn_model_1.fit(X_train_scaled, y_train_1)
y_pred_1 = nn_model_1.predict(X_test_scaled)

nn_model_3.fit(X_train_scaled, y_train_3)
y_pred_3 = nn_model_3.predict(X_test_scaled)

nn_model_7.fit(X_train_scaled, y_train_7)
y_pred_7 = nn_model_7.predict(X_test_scaled)

mae_nn_1 = mean_absolute_error(y_test_1, y_pred_1)
mse_nn_1 = mean_squared_error(y_test_1, y_pred_1)
r2_nn_1 = r2_score(y_test_1, y_pred_1)
print(f"MAE (NN) for D+1: {mae_nn_1:.2f}")
print(f"MSE (NN) for D+1: {mse_nn_1:.2f}")
print(f"R2 (NN) for D+1: {r2_nn_1:.2f}")

mae_nn_3 = mean_absolute_error(y_test_3, y_pred_3)
mse_nn_3 = mean_squared_error(y_test_3, y_pred_3)
r2_nn_3 = r2_score(y_test_3, y_pred_3)
print(f"MAE (NN) for D+3: {mae_nn_3:.2f}")
print(f"MSE (NN) for D+3: {mse_nn_3:.2f}")
print(f"R2 (NN) for D+3: {r2_nn_3:.2f}")

mae_nn_7 = mean_absolute_error(y_test_7, y_pred_7)
mse_nn_7 = mean_squared_error(y_test_7, y_pred_7)
r2_nn_7 = r2_score(y_test_7, y_pred_7)
print(f"MAE (NN) for D+7: {mae_nn_7:.2f}")
print(f"MSE (NN) for D+7: {mse_nn_7:.2f}")
print(f"R2 (NN) for D+7: {r2_nn_7:.2f}")


input_value = [ 20, 13, 19, 13, 1, 0, 0]  

input_scaled = scaler.transform([input_value])

future_prediction_D_1 = max(0, nn_model_1.predict(input_scaled)[0])
print(f"Predicted PM2.5 (NN) for 1 day ahead: {future_prediction_D_1:.2f}")

future_prediction_D_3 = max(0, nn_model_3.predict(input_scaled)[0])
print(f"Predicted PM2.5 (NN) for 3 day ahead: {future_prediction_D_3:.2f}")

future_prediction_D_7 = max(0, nn_model_7.predict(input_scaled)[0])
print(f"Predicted PM2.5 (NN) for 7 day ahead: {future_prediction_D_7:.2f}")

cv_scores = cross_val_score(nn_model_1, X_train_scaled, y_train_1, cv=5, scoring='neg_mean_absolute_error')
print(f"Cross-validation MAE (NN_(D+1)): {-cv_scores.mean():.2f}")


plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.scatter(y_test_1, y_pred_1, color='blue', label="Predicted vs Actual (D+1)")
plt.plot([y_test_1.min(), y_test_1.max()], [y_test_1.min(), y_test_1.max()], color='red', linewidth=2)
plt.xlabel("Actual PM2.5 (D+1)")
plt.ylabel("Predicted PM2.5 (D+1)")
plt.title(f"Predictions vs Actual (D+1)\nR2 = {r2_nn_1:.2f}")
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(y_test_3, y_pred_3, color='green', label="Predicted vs Actual (D+3)")
plt.plot([y_test_3.min(), y_test_3.max()], [y_test_3.min(), y_test_3.max()], color='red', linewidth=2)
plt.xlabel("Actual PM2.5 (D+3)")
plt.ylabel("Predicted PM2.5 (D+3)")
plt.title(f"Predictions vs Actual (D+3)\nR2 = {r2_nn_3:.2f}")
plt.legend()

plt.subplot(1, 3, 3)
plt.scatter(y_test_7, y_pred_7, color='purple', label="Predicted vs Actual (D+7)")
plt.plot([y_test_7.min(), y_test_7.max()], [y_test_7.min(), y_test_7.max()], color='red', linewidth=2)
plt.xlabel("Actual PM2.5 (D+7)")
plt.ylabel("Predicted PM2.5 (D+7)")
plt.title(f"Predictions vs Actual (D+7)\nR2 = {r2_nn_7:.2f}")
plt.legend()

plt.tight_layout()
plt.show()

# Validation Part
validation_df = pd.read_csv('../dataset/validation.csv')

X_validation = validation_df[['PM2.5', 'PM2.5(D-1)', 'PM2.5(D-3)', 'PM2.5(D-7)', 'Season_Summer', 'Season_Rainy', 'Season_Winter']]
y_validation_1 = validation_df['PM2.5(D+1)']
y_validation_3 = validation_df['PM2.5(D+3)']
y_validation_7 = validation_df['PM2.5(D+7)']

print(y_validation_1.head())

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) 
X_validation_scaled = scaler.transform(X_validation) 


y_pred_1 = nn_model_1.predict(X_validation_scaled)
y_pred_3 = nn_model_3.predict(X_validation_scaled)
y_pred_7 = nn_model_7.predict(X_validation_scaled)

mae_nn_1 = mean_absolute_error(y_validation_1, y_pred_1)
mse_nn_1 = mean_squared_error(y_validation_1, y_pred_1)
r2_nn_1 = r2_score(y_validation_1, y_pred_1)

mae_nn_3 = mean_absolute_error(y_validation_3, y_pred_3)
mse_nn_3 = mean_squared_error(y_validation_3, y_pred_3)
r2_nn_3 = r2_score(y_validation_3, y_pred_3)

mae_nn_7 = mean_absolute_error(y_validation_7, y_pred_7)
mse_nn_7 = mean_squared_error(y_validation_7, y_pred_7)
r2_nn_7 = r2_score(y_validation_7, y_pred_7)

print(f"MAE (NN) for D+1: {mae_nn_1:.2f}")
print(f"MSE (NN) for D+1: {mse_nn_1:.2f}")
print(f"R2 (NN) for D+1: {r2_nn_1:.2f}")

print(f"MAE (NN) for D+3: {mae_nn_3:.2f}")
print(f"MSE (NN) for D+3: {mse_nn_3:.2f}")
print(f"R2 (NN) for D+3: {r2_nn_3:.2f}")

print(f"MAE (NN) for D+7: {mae_nn_7:.2f}")
print(f"MSE (NN) for D+7: {mse_nn_7:.2f}")
print(f"R2 (NN) for D+7: {r2_nn_7:.2f}")

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.scatter(y_validation_1, y_pred_1, color='blue', label="Predicted vs Actual (D+1) (Validation)")
plt.plot([y_validation_1.min(), y_validation_1.max()], [y_validation_1.min(), y_validation_1.max()], color='red', linewidth=2)
plt.xlabel("Actual PM2.5 (D+1)")
plt.ylabel("Predicted PM2.5 (D+1)")
plt.title(f"Predictions vs Actual (D+1)\nR2 = {r2_nn_1:.2f}")
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(y_validation_3, y_pred_3, color='green', label="Predicted vs Actual (D+3) (Validation)")
plt.plot([y_validation_3.min(), y_validation_3.max()], [y_validation_3.min(), y_validation_3.max()], color='red', linewidth=2)
plt.xlabel("Actual PM2.5 (D+3)")
plt.ylabel("Predicted PM2.5 (D+3)")
plt.title(f"Predictions vs Actual (D+3)\nR2 = {r2_nn_3:.2f}")
plt.legend()

plt.subplot(1, 3, 3)
plt.scatter(y_validation_7, y_pred_7, color='purple', label="Predicted vs Actual (D+7) (Validation)")
plt.plot([y_validation_7.min(), y_validation_7.max()], [y_validation_7.min(), y_validation_7.max()], color='red', linewidth=2)
plt.xlabel("Actual PM2.5 (D+7)")
plt.ylabel("Predicted PM2.5 (D+7)")
plt.title(f"Predictions vs Actual (D+7)\nR2 = {r2_nn_7:.2f}")
plt.legend()

plt.tight_layout()
plt.show()

# param_grid = {
#     'hidden_layer_sizes': [(32, 64), (64, 128), (64, 64), (128, 128)],
#     'activation': ['relu', 'tanh', 'logistic'],
#     'max_iter': [1000, 1500],
#     'learning_rate_init': [0.001, 0.01, 0.1]
# }

# grid_search = GridSearchCV(MLPRegressor(), param_grid, cv=5, scoring='neg_mean_absolute_error')
# grid_search.fit(X_train_scaled, y_train_1)

# print(f"Best parameters found: {grid_search.best_params_}")