from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load saved models and scaler
logistic_model = pickle.load(open('logistic_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get all 30 features from the form
    features = [float(request.form[f'feature{i}']) for i in range(1, 31)]
    
    # Preprocess input data
    input_data = scaler.transform([features])  # Scale input features
    
    # Predict using the logistic regression model
    prediction = logistic_model.predict(input_data)[0]
    result = 'Fraudulent' if prediction == 1 else 'Legitimate'
    
    return render_template('index.html', prediction_text=f'Transaction is {result}')

if __name__ == '__main__':
    app.run(debug=True)
