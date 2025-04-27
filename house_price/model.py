import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('train.csv')
# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.shape)

missing=df.isnull().sum()
missing=missing[missing>0].sort_values(ascending=False)
missing_percentage = (missing / len(df)) * 100

# plt.figure(figsize=(12,6))
# sns.barplot(x=missing_percentage.values,y=missing_percentage.index)
# plt.title("missing values by %")
# plt.show()

df.drop(columns=missing_percentage[missing_percentage>40].index,inplace=True)

catag_col=df.select_dtypes(include='object').columns
numbe_col=df.select_dtypes(exclude='object').columns

for col in catag_col:
    print(col, df[col].nunique())

for col in numbe_col:
    print(col,df[col].nunique())