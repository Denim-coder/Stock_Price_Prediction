import streamlit as st
import numpy as np
import joblib

# Title and initial message
st.title("📈 Stock Price Movement Prediction")

# Try to load the model and show error if it fails
try:
    model = joblib.load('model.pkl')
    st.success("✅ Model loaded successfully.")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Dummy input for testing
st.markdown("### 📥 Enter Sample Inputs:")
open_price = st.number_input("Open Price", value=100.0)
high_price = st.number_input("High Price", value=110.0)
low_price = st.number_input("Low Price", value=95.0)



# Button and prediction
if st.button("Predict"):
    try:
        input_data = np.array([[open_price, high_price, low_price]])
        prediction = model.predict(input_data)[0]
        result = "📈 Price Will Go Up" if prediction == 1 else "📉 Price Will Go Down"
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
