from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load('best_rf_model.pkl')  # Replace with 'best_xgb_model.pkl' if using XGBoost

# Initialize Flask app
app = Flask(__name__)

# Define a prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get data from the request
    features = np.array(data['features']).reshape(1, -1)  # Convert to 2D array for prediction
    prediction = model.predict(features)[0]
    prediction_proba = model.predict_proba(features)[0][1]
    
    return jsonify({'prediction': int(prediction), 'churn_probability': float(prediction_proba)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Use a different port, e.g., 5001
