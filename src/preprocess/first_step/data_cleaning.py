import pandas as pd

year_list = ["2020","2021", "2022", "2023"]

for year in year_list:
    df = pd.read_csv(f'../csv/day/PM2.5({year}).csv')

    print(df['59T'].isnull().sum())

    df = df.loc[:365, ['Date', '59T']]
    
    df['59T'].fillna(df['59T'].mean() , inplace=True)

    print(df['59T'].isnull().sum())
    
    df.to_csv(f'./dataset/PM2.5({year}).csv', index=False)
    