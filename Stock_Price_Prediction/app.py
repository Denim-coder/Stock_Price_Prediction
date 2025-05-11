import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app layout
st.title("Stock Price Prediction")

# Input fields for user data
st.sidebar.header('User Input Features')

def user_input_features():
    open_price = st.sidebar.number_input('Open Price', min_value=0.0, value=100.0)
    close_price = st.sidebar.number_input('Close Price', min_value=0.0, value=95.0)
    low_price = st.sidebar.number_input('Low Price', min_value=0.0, value=90.0)
    high_price = st.sidebar.number_input('High Price', min_value=0.0, value=110.0)
    is_quarter_end = st.sidebar.selectbox('Is Quarter End?', [0, 1])  # 0 for No, 1 for Yes

    data = {'open-close': open_price - close_price,
            'low-high': low_price - high_price,
            'is_quarter_end': is_quarter_end}
    
    features = np.array([list(data.values())])
    return features

# Get user input features
features = user_input_features()

# Scale the input data using the same scaler used during training
scaled_features = scaler.transform(features)

# Make predictions
prediction = model.predict(scaled_features)[0]

# Display prediction result
st.header('Prediction')
if prediction == 1:
    st.write("### Stock Price will go up.")
else:
    st.write("### Stock Price will go down.")

# Add the model's prediction probabilities
probs = model.predict_proba(scaled_features)
st.write(f"Prediction Probabilities: {probs}")

