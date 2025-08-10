import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import joblib
from datetime import datetime

# Set the page title
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈"
)

# --- 1. Load Data and Model ---
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('Sample - Superstore.csv', encoding='latin1', parse_dates=['Order Date', 'Ship Date'], low_memory=False)
        return data
    except FileNotFoundError:
        st.error("Data file not found. Please make sure 'Sample - Superstore.csv' is in the same folder as this app.")
        return pd.DataFrame()

@st.cache_resource
def load_model():
    try:
        model = joblib.load('xgboost_model.joblib')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please make sure 'xgboost_model.joblib' is in the same folder as this app.")
        return None

@st.cache_data
def load_artifacts():
    try:
        encoder = joblib.load('one_hot_encoder.joblib')
        feature_columns = joblib.load('feature_columns.joblib')
        return encoder, feature_columns
    except FileNotFoundError:
        st.error("Artifacts not found. Please run the training script first.")
        return None, None

sales_data = load_data()
model = load_model()
encoder, feature_columns = load_artifacts()

# --- 2. Dashboard Title and Sidebar ---
st.title("Dynamic Sales Forecasting Dashboard 📈")
st.sidebar.header("Forecast Settings")
forecast_horizon_option = st.sidebar.selectbox(
    "Choose Forecast Horizon",
    ("Days", "Months")
)

if forecast_horizon_option == "Days":
    forecast_units = 1
    forecast_horizon = st.sidebar.slider("Number of Days to Forecast", 1, 90, 31)
else: # Months
    forecast_units = 30
    forecast_horizon = st.sidebar.slider("Number of Months to Forecast", 1, 24, 12)

total_days_to_forecast = forecast_horizon * forecast_units

# --- 3. Main Dashboard Content ---
st.subheader("Historical Sales Data")

if not sales_data.empty:
    sales_over_time = sales_data.groupby('Order Date')['Sales'].sum().reset_index()
    st.line_chart(sales_over_time, x="Order Date", y="Sales")

st.subheader(f"Sales Forecast for the Next {total_days_to_forecast} Days")

if st.button("Generate Forecast"):
    if not sales_data.empty and model is not None and encoder is not None:
        st.success(f"Generating forecast for the next {total_days_to_forecast} days...")
        
        # --- Create future dates and features for forecasting ---
        last_date = sales_data['Order Date'].max()
        future_dates = pd.date_range(start=last_date, periods=total_days_to_forecast + 1, freq='D')[1:]
        
        future_features = pd.DataFrame({'Order Date': future_dates})
        future_features['year'] = future_features['Order Date'].dt.year
        future_features['month'] = future_features['Order Date'].dt.month
        future_features['day'] = future_features['Order Date'].dt.day
        future_features['dayofweek'] = future_features['Order Date'].dt.dayofweek
        future_features['weekofyear'] = future_features['Order Date'].dt.isocalendar().week.astype(int)
        future_features['is_weekend'] = future_features['dayofweek'].isin([5, 6]).astype(int)
        
        # Create a dummy dataframe with the same columns as the training data
        dummy_df = pd.DataFrame(0, index=future_features.index, columns=feature_columns)
        for col in dummy_df.columns:
            if col in future_features.columns:
                dummy_df[col] = future_features[col]
            else:
                pass
        
        # Make predictions and create a forecast dataframe
        future_predictions = model.predict(dummy_df)
        
        forecast_df = pd.DataFrame({
            'Order Date': future_dates,
            'Sales': future_predictions
        })
        
        # Combine historical and forecasted data for a single chart
        combined_data = pd.concat([sales_over_time, forecast_df], ignore_index=True)
        
        st.success("Forecast generated!")
        st.line_chart(combined_data, x="Order Date", y="Sales")
    else:
        st.warning("Model or data not loaded. Please ensure all files are in the correct location.")
        
st.info("This is an interactive dashboard to visualize sales forecasts.")