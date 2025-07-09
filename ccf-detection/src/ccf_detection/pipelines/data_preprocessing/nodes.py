import pandas as pd
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler


def preprocess_creditcard_data(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Drop duplicates
    df = df.drop_duplicates()

    # 2. Scale 'Amount' column
    #scaler = StandardScaler()
    #df['normAmount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    #df["normTime"] = scaler.fit_transform(df[["Time"]])

    df['scaled_amount'] = RobustScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = RobustScaler().fit_transform(df['Time'].values.reshape(-1, 1))

    #df.drop(['Time', 'Amount'], axis=1, inplace=True)

    # 3. Drop 'Time' column
    #if 'Time' in df.columns:
    #    df = df.drop(columns=['Time'])

    return df
