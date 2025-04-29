import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

def clean_data(df,istrain=True):
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

    # for col in catag_col:
    #     print(col, df[col].nunique())

    # for col in numbe_col:
    #     print(col,df[col].nunique())


    df[catag_col] = df[catag_col].replace(['NA', 'None', 'Missing', '?'], np.nan)


    for col in catag_col:
        df[col] = df[col].fillna(df[col].mode()[0])

    for col in numbe_col:
        df[col] = df[col].fillna(df[col].median())

    # print(df.columns)

    # for col in ['GrLivArea', 'SalePrice', 'LotFrontage', 'GarageArea']:
    #     sns.boxplot(x=df[col])  
    #     plt.title(col)  
    #     plt.show()
    if istrain:
        for col in ['GrLivArea', 'SalePrice', 'LotFrontage', 'GarageArea']:
            Q1=df[col].quantile(.25)
            Q3=df[col].quantile(.75)
            IQR=Q3-Q1
            lower=Q1-1.5*IQR
            upper=Q3+1.5*IQR
            df=df[(df[col]>=lower) & (df[col]<=upper)]

    num_cols = df.select_dtypes(exclude='object').columns
    skewness = df[num_cols].skew()
    for col in skewness[skewness > 0.5].index:
        if (df[col] > 0).all():
            df[col] = np.log1p(df[col])
                # print(col)

    # corr=df[numbe_col].corr()
    # plt.figure(figsize=(15,10))
    # sns.heatmap(corr,cmap='coolwarm',annot=False)
    # plt.title('correlation')
    # plt.show()

    catag_col=df.select_dtypes(include='object').columns
    df=pd.get_dummies(df,columns=catag_col,drop_first=True)

    df = df.drop('Id', axis=1)
    return df

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
test_ids = test_df['Id']


train_clean = clean_data(train_df, istrain=True)
test_clean = clean_data(test_df, istrain=False)


X_train = train_clean.drop('SalePrice', axis=1)
y_train = train_clean['SalePrice']
X_test = test_clean.reindex(columns=X_train.columns, fill_value=0)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = RandomForestRegressor(random_state=42)
model.fit(X_train_scaled, y_train)


preds = model.predict(X_test_scaled)
final_preds = np.expm1(preds)  


pd.DataFrame({'Id': test_ids, 'SalePrice': final_preds}).to_csv("submission.csv", index=False)