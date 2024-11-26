import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

# แบ่งข้อมูลเป็น Train และ Test
split_index = int(len(final_df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train_1, y_test_1 = y_1.iloc[:split_index], y_1.iloc[split_index:]
y_train_3, y_test_3 = y_3.iloc[:split_index], y_3.iloc[split_index:]
y_train_7, y_test_7 = y_7.iloc[:split_index], y_7.iloc[split_index:]

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model_1 = LinearRegression()
model_1.fit(X_train_scaled, y_train_1)
y_pred_1 = model_1.predict(X_test_scaled)

model_3 = LinearRegression()
model_3.fit(X_train_scaled, y_train_3)
y_pred_3 = model_3.predict(X_test_scaled)

model_7 = LinearRegression()
model_7.fit(X_train_scaled, y_train_7)
y_pred_7 = model_7.predict(X_test_scaled)

# Calculate and print performance metrics
mae_1 = mean_absolute_error(y_test_1, y_pred_1)
mse_1 = mean_squared_error(y_test_1, y_pred_1)
r2_1 = r2_score(y_test_1, y_pred_1)

mae_3 = mean_absolute_error(y_test_3, y_pred_3)
mse_3 = mean_squared_error(y_test_3, y_pred_3)
r2_3 = r2_score(y_test_3, y_pred_3)

mae_7 = mean_absolute_error(y_test_7, y_pred_7)
mse_7 = mean_squared_error(y_test_7, y_pred_7)
r2_7 = r2_score(y_test_7, y_pred_7)

print(f"MAE for D+1: {mae_1}")
print(f"MSE for D+1: {mse_1}")
print(f"R2 for D+1: {r2_1}")

print(f"MAE for D+3: {mae_3}")
print(f"MSE for D+3: {mse_3}")
print(f"R2 for D+3: {r2_3}")

print(f"MAE for D+7: {mae_7}")
print(f"MSE for D+7: {mse_7}")
print(f"R2 for D+7: {r2_7}")


input_value = [49.0, 38.0, 38.0, 20.0, 0, 0, 1] 

future_data = scaler.transform([input_value])

future_prediction_1 = max(0, model_1.predict(future_data)[0])
future_prediction_3 = max(0, model_3.predict(future_data)[0])
future_prediction_7 = max(0, model_7.predict(future_data)[0])

# Print predictions
print(f"Predicted PM2.5 for 1 day ahead: {future_prediction_1:.2f}")
print(f"Predicted PM2.5 for 3 days ahead: {future_prediction_3:.2f}")
print(f"Predicted PM2.5 for 7 days ahead: {future_prediction_7:.2f}")
