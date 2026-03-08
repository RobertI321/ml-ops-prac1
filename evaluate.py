from data_cleaning import clean_data, read_and_concat_dfs
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error as RMSE
import joblib
import os

# Load the trained regression model
model = joblib.load("regression_model.pkl")

# Read and clean the data
df = read_and_concat_dfs(["data/" + f for f in os.listdir("data/") if f.endswith(".parquet")])
df_clean = clean_data(df)

selected_features = ["trip_distance", "trip_type"] # Based on EDA in .ipynb file
predictable_feature = "total_amount"

_, test_df = train_test_split(df_clean, test_size=0.2, random_state=42)
test = test_df[selected_features + [predictable_feature]]

# Make predictions and evaluate the model
preds = model.predict(test[selected_features])
RMSE = RMSE(test[predictable_feature], preds)

print("Evaluation for regression model - Root mean squared error:", RMSE)
