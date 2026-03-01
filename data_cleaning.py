#data_cleaning.py
import pandas as pd

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remove duplicates
    df = df.drop_duplicates()

    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop(columns=["ehail_fee"])
    df =df[df.isna().sum(axis=1) < 4] # Drop rows with more than 4 missing values, because we can assume that they are not representative of the population and they can (with imputation) introduce noise in the model
    
    return df

def outlier_removal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df[df["passenger_count"] > 0] # Remove passenger counts < 0, because they are not representative of the population

    for col in ["trip_distance", "total_amount"]:
        df = df[df[col] > 0] # Remove rows with non-positive values, because they are not representative of the population and they can introduce noise in the model
        
        Q_range1 = df[[col]].quantile(0.25)
        Q_range2 = df[[col]].quantile(0.75)
        IQR = Q_range2 - Q_range1
        lower_bound = Q_range1  - 3 * IQR
        upper_bound = Q_range2 + 3 * IQR
        df = df[(df[col] >= lower_bound.values[0]) & (df[col] <= upper_bound.values[0])]

    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = basic_cleaning(df)
    df = handle_missing_values(df)
    df = outlier_removal(df)

    return df
