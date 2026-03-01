from data_cleaning import clean_data
import pandas as pd

from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
from sklearn.metrics import root_mean_squared_error as RMSE

import joblib

df = pd.read_parquet("data/green_tripdata_2021-01.parquet")
df_clean = clean_data(df)

train_df, test_df = train_test_split(df_clean, test_size=0.2, random_state=42)

selected_featrues = ["trip_distance", "trip_type"] # Based on EDA in .ipynb file
predictable_feature = "total_amount"

train = train_df[selected_featrues + [predictable_feature]]
test = test_df[selected_featrues + [predictable_feature]]

model = smf.ols(f'{predictable_feature} ~  {" + ".join(selected_featrues)}', train).fit()

preds = model.predict(test[selected_featrues])

print("Root mean squared error:", RMSE(test[predictable_feature], preds))

#joblib.dump(model, "regression_model.pkl")