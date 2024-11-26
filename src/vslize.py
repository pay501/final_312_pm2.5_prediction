import numpy as np
import pandas as pd

# Load the dataset
df = pd.read_csv("./dataset/real_dataset.csv")

# Display basic information
print("First few rows of the dataset:")
print(df.head())

print("\nSummary statistics:")
print(df.describe())

print("\nData types and null values:")
print(df.info())
print("\nMissing values per column:")
print(df.isnull().sum())

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.boxplot(x=df['PM2.5'])
plt.title("Box Plot of PM2.5")
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

sns.boxplot(x='Season_Winter', y='PM2.5', data=df)
plt.title("PM2.5 Distribution in Winter")
plt.show()

sns.scatterplot(x=df['PM2.5(D-1)'], y=df['PM2.5(D+1)'])
plt.title("PM2.5(D-1) vs PM2.5(D+1)")
plt.show()

# Data for each model and metric
models = ['MLP', 'Linear Regression', 'Random Forest']

# Metrics for D+1, D+3, D+7
mae_d1 = [2.30, 3.52, 2.56]
mse_d1 = [9.28, 24.41, 12.39]
r2_d1 = [0.91 * 100, 0.75 * 100, 0.87 * 100] # Corrected length to match mae_d7 and mse_d7

# X positions for each 1odel
x = np.arange(len(models))

# Width of each bar
width = 0.2

# Create the bar chart for D+7
fig, ax = plt.subplots(figsize=(10, 6))

# Plot metrics
ax.bar(x - width, mae_d1, width, label='MAE', color='skyblue')
ax.bar(x, mse_d1, width, label='MSE', color='lightgreen')
ax.bar(x + width, r2_d1, width, label='R2', color='salmon')

# Add titles and labels
ax.set_title('Metrics Comparison for D+1', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.set_ylabel('Metric Values', fontsize=12)
ax.set_xlabel('Models', fontsize=12)
ax.legend()

# Display the plot
plt.tight_layout()
plt.show()
