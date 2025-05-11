import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('model.pkl')  # Replace 'model.pkl' with your model file name
scaler = joblib.load('scaler.pkl') # Replace 'scaler.pkl' with your scaler file name

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json()
        open_close = data['open_close']
        low_high = data['low_high']
        is_quarter_end = data['is_quarter_end']

        # Preprocess the input data
        input_data = scaler.transform([[open_close, low_high, is_quarter_end]])

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Return the prediction as JSON response
        return jsonify({'prediction': int(prediction)})  # Convert prediction to int

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
