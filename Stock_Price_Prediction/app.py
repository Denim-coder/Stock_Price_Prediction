import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load('model.pkl')

# Title of the web app
st.title('Stock Price Prediction')

# Instructions for the user
st.write("""
    This app predicts whether the stock price will go up or down based on the following features:
    - Difference between Open and Close prices
    - Difference between Low and High prices
    - Whether the date is a quarter-end
""")

# User input section
st.sidebar.header('User Input Features')

# Get user input for 'open-close' (difference between Open and Close)
open_price = st.sidebar.number_input('Open Price', min_value=0.0, value=100.0)
close_price = st.sidebar.number_input('Close Price', min_value=0.0, value=100.0)
open_close = open_price - close_price

# Get user input for 'low-high' (difference between Low and High)
low_price = st.sidebar.number_input('Low Price', min_value=0.0, value=90.0)
high_price = st.sidebar.number_input('High Price', min_value=0.0, value=110.0)
low_high = low_price - high_price

# Get user input for 'is_quarter_end'
is_quarter_end = st.sidebar.selectbox('Is Quarter End?', ('Yes', 'No'))
is_quarter_end = 1 if is_quarter_end == 'Yes' else 0

# Create feature array (scaled)
features = np.array([[open_close, low_high, is_quarter_end]])

# Apply StandardScaler (same scaling applied during training)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Predict the target using the model
prediction = model.predict(scaled_features)[0]

# Show prediction result
if prediction == 1:
    result = "📈 Stock Price Will Go Up"
else:
    result = "📉 Stock Price Will Go Down"

# Display the prediction result
st.write(f"### Prediction: {result}")

