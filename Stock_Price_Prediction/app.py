import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load('model.pkl')

# Title
st.title("📈 Stock Price Prediction")

# Inputs
open_price = st.number_input("Open Price")
high_price = st.number_input("High Price")
low_price = st.number_input("Low Price")
close_price = st.number_input("Close Price")
volume = st.number_input("Volume")

# Button and prediction
if st.button("Predict"):
    input_data = np.array([[open_price, high_price, low_price, close_price, volume]])
    prediction = model.predict(input_data)[0]
    result = "📈 Price Will Go Up" if prediction == 1 else "📉 Price Will Go Down"
    st.success(result)
