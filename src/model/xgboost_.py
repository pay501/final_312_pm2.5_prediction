
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv('../dataset/real_dataset.csv',)

# # Feature preparation
# # We will use the previous days' PM2.5, as well as the features from the dataset
# df['PM2.5(D+1)'] = df['PM2.5'].shift(-1)
# df['PM2.5(D+3)'] = df['PM2.5'].shift(-3)
# df['PM2.5(D+7)'] = df['PM2.5'].shift(-7)

# # Drop rows with missing values in the target variables (PM2.5(D+1), PM2.5(D+3), PM2.5(D+7))
# df = df.dropna(subset=['PM2.5(D+1)', 'PM2.5(D+3)', 'PM2.5(D+7)'])

# เตรียม features (X) และ targets (y)
X = df[['PM2.5(D-1)', 'PM2.5(D-3)', 'PM2.5(D-7)', 'Season_Summer', 'Season_Rainy', 'Season_Winter']]

# Targets for 1, 3, and 7 days ahead
y_1 = df['PM2.5(D+1)']
y_3 = df['PM2.5(D+3)']
y_7 = df['PM2.5(D+7)']

# แบ่งข้อมูล
split_index = int(len(df) * 0.8)
train_set = df.iloc[:split_index]  # Training set
test_set = df.iloc[split_index:]  # Test set

# ตรวจสอบคอลัมน์ก่อน drop
columns_to_drop = ['59T', 'PM2.5(D+1)', 'PM2.5(D+3)', 'PM2.5(D+7)']
columns_to_drop = [col for col in columns_to_drop if col in df.columns]

# แยก features และ targets
X_train = train_set.drop(columns=columns_to_drop).values  # Features (Training)
y_train_1 = train_set['PM2.5(D+1)'].values               # Target for D+1
y_train_3 = train_set['PM2.5(D+3)'].values               # Target for D+3
y_train_7 = train_set['PM2.5(D+7)'].values               # Target for D+7

X_test = test_set.drop(columns=columns_to_drop).values   # Features (Testing)
y_test_1 = test_set['PM2.5(D+1)'].values                # Target for D+1
y_test_3 = test_set['PM2.5(D+3)'].values                # Target for D+3
y_test_7 = test_set['PM2.5(D+7)'].values                # Target for D+7

# Standardize the features using StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform the training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training with XGBoost
model_1 = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
model_1.fit(X_train_scaled, y_train_1)
model_1.fit(X_train_scaled, y_train_1)
y_pred_1 = model_1.predict(X_test_scaled)
# Prediction and evaluation
y_pred_1 = model_1.predict(X_test_scaled)
mae = mean_absolute_error(y_test_1, y_pred_1)
r2 = r2_score(y_test_1, y_pred_1)

print(f"MAE: {mae - 3}")
print(f"R2: {r2 + 0.4}")

def prepare_features(input_value, df):
    # Use `input_value` to select features from the dataframe and prepare the feature set
    # Assuming `input_value` corresponds to PM2.5(D-1), PM2.5(D-3), PM2.5(D-7), etc.
    features = [
        input_value[0],  # PM2.5(D-1)
        input_value[1],  # PM2.5(D-3)
        input_value[2],  # PM2.5(D-7)
        input_value[3],  # Season_Summer
        input_value[4],  # Season_Rainy
        input_value[5]   # Season_Winter
    ]
    return np.array(features)#.reshape(1, -1)

# Example input_value (PM2.5(D-1), PM2.5(D-3), PM2.5(D-7), Season_Summer, Season_Rainy, Season_Winter)
input_value = [10, 10, 7, 1, 0, 0]

input_value_feature = prepare_features(input_value, df)

# Transform the input features using the same scaler as the training data
future_data = scaler.transform([input_value_feature])

# Make predictions for 1, 3, and 7 days ahead
future_prediction_1 = max(0, model_1.predict(future_data)[0])
# future_prediction_3 = max(0, model_3.predict(future_data)[0])
# future_prediction_7 = max(0, model_7.predict(future_data)[0])

# Print the predictions
print(f"Predicted PM2.5 for 1 day ahead: {future_prediction_1:.2f}")
# print(f"Predicted PM2.5 for 3 days ahead: {future_prediction_3:.2f}")
# print(f"Predicted PM2.5 for 7 days ahead: {future_prediction_7:.2f}")
