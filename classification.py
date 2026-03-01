# Classification
from data_cleaning import clean_data

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import joblib

df = pd.read_parquet("data/green_tripdata_2021-01.parquet")
df_clean = clean_data(df)

selected_features = ["VendorID", "trip_type", "fare_amount"] # Based on EDA in .ipynb file
predictable_feature = "passenger_count"

X = df_clean[selected_features]
y = df_clean[predictable_feature]

X_train, x_test, y_train, y_test = train_test_split(X[selected_features], y, test_size=0.2, random_state=42, stratify=y)


pipeline_classification = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(random_state=42))
])

pipeline_classification.fit(X_train, y_train)
y_pred_pipeline = pipeline_classification.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred_pipeline))
print(classification_report(y_test, y_pred_pipeline))
print(confusion_matrix(y_test, y_pred_pipeline))

#joblib.dump(pipeline_classification, "classification_pipeline.pkl")
