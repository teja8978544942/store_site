import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import os
import joblib

# --- Data Loading and Initial Cleaning ---
file_name = 'Sample - Superstore.csv'
sales_df = pd.read_csv(file_name, encoding='latin1', parse_dates=['Order Date', 'Ship Date'], low_memory=False)

# --- Feature Engineering ---
processed_df = sales_df.copy()
processed_df['year'] = processed_df['Order Date'].dt.year
processed_df['month'] = processed_df['Order Date'].dt.month
processed_df['day'] = processed_df['Order Date'].dt.day
processed_df['dayofweek'] = processed_df['Order Date'].dt.dayofweek
processed_df['weekofyear'] = processed_df['Order Date'].dt.isocalendar().week.astype(int)
processed_df['is_weekend'] = processed_df['dayofweek'].isin([5, 6]).astype(int)
processed_df['shipping_days'] = (processed_df['Ship Date'] - processed_df['Order Date']).dt.days

# --- Data Preparation for Modeling ---
features = processed_df.drop(['Order Date', 'Ship Date', 'Sales', 'Row ID', 'Customer ID', 'Customer Name', 'Product Name'], axis=1)
target = processed_df['Sales']

categorical_cols = features.select_dtypes(include=['object']).columns
numerical_cols = features.select_dtypes(include=np.number).columns

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_features = encoder.fit_transform(features[categorical_cols])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))

features_final = pd.concat([features[numerical_cols].reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

# --- Model Training ---
X_train, X_test, y_train, y_test = train_test_split(features_final, target, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# --- Save Model and Artifacts ---
joblib.dump(model, 'xgboost_model.joblib')
joblib.dump(encoder, 'one_hot_encoder.joblib')
joblib.dump(list(X_train.columns), 'feature_columns.joblib')

print("Model, encoder, and feature columns saved.")
print("Training complete.")