import joblib
import numpy as np
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
        open_close = data['open-close']
        low_high = data['low-high']
        is_quarter_end = data['is_quarter_end']
        
        # Preprocess the input data
        input_data = scaler.transform([[open_close, low_high, is_quarter_end]])

        # Make prediction
        prediction = model.predict(input_data)[0] 

        # Return the prediction as JSON response
        return jsonify({'prediction': int(prediction)}) 
    else:
        return "Welcome to Tesla Stock Price prediction!"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
