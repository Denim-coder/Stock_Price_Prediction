import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('model.pkl')  
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get input data from the request
        data = request.get_json() 

        # Create a DataFrame from the input data
        input_df = pd.DataFrame([data])  

        # Select the relevant features
        relevant_features = input_df[['open-close', 'low-high', 'is_quarter_end']] 

        # Transform the features using the scaler
        input_data = scaler.transform(relevant_features)  

        # Make prediction
        prediction = model.predict(input_data)[0]  

        # Return the prediction as JSON response
        return jsonify({'prediction': int(prediction)})  
    else:
        return "Welcome to Tesla Stock Price prediction!"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
