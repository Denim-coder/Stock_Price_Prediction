import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("📈 Stock Price Movement Prediction App")

st.markdown("Enter the following values to predict whether the stock price will go UP or DOWN:")

# Input fields
open_price = st.number_input("Open Price", value=100.0)
high_price = st.number_input("High Price", value=110.0)
low_price = st.number_input("Low Price", value=95.0)
close_price = st.number_input("Close Price", value=105.0)
is_quarter_end = st.selectbox("Is it quarter end?", options=[0, 1])

# Calculate features
open_close = open_price - close_price  # open - close
low_high = low_price - high_price      # low - high

# Feature array to be passed to the model
features = np.array([[open_close, low_high, is_quarter_end]])

# Scale the features
scaled_features = scaler.transform(features)

# Predict when the button is clicked
if st.button("Predict"):
    prediction = model.predict(scaled_features)  # Make the prediction

    # Show the result
    if prediction == 1:
        st.success("🔼 The stock price is likely to go **UP** tomorrow.")
    else:
        st.error("🔻 The stock price is likely to go **DOWN** tomorrow.")

