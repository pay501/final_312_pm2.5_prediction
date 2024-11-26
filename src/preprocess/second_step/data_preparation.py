import pandas as pd

df = pd.read_csv('../dataset/all_year.csv', parse_dates=['Date'])


df['PM2.5(D-1)'] = df['59T'].shift(1)
df['PM2.5(D-3)'] = df['59T'].shift(3)
df['PM2.5(D-7)'] = df['59T'].shift(7)

df['PM2.5(D+1)'] = df['59T'].shift(-1)
df['PM2.5(D+3)'] = df['59T'].shift(-3)
df['PM2.5(D+7)'] = df['59T'].shift(-7)

# df['Last24hrs_mean'] = df['59T'].rolling(window=24, min_periods=1).mean()
# df['Last48hrs_mean'] = df['59T'].rolling(window=48, min_periods=1).mean()
# df['Last72hrs_mean'] = df['59T'].rolling(window=72, min_periods=1).mean()

df['Month'] = df['Date'].dt.month
df['Season_Summer'] = df['Month'].apply(lambda x: 1 if x in [3, 4, 5] else 0)
df['Season_Rainy'] = df['Month'].apply(lambda x: 1 if x in [6, 7, 8, 10] else 0)
df['Season_Winter'] = df['Month'].apply(lambda x: 1 if x in [11, 12, 1, 2] else 0)

df.drop('Month', axis=1, inplace=True)
df.drop('Date', axis=1, inplace=True)

print("before drop colummn", df.isnull().sum())
print(df.shape)
df = df.dropna(subset=['PM2.5(D-1)', 'PM2.5(D-3)', 'PM2.5(D-7)'])
df = df.dropna(subset=['PM2.5(D+1)', 'PM2.5(D+3)', 'PM2.5(D+7)'])
print("after drop colummn", df.isnull().sum())
print(df.shape)

print(df.head(10))

df.to_csv('../dataset/real_dataset.csv', index=False)