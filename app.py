from flask import Flask, request, jsonify, render_template
from joblib import load  # Import joblib for loading the model
import numpy as np

# Load the pre-trained model using joblib
model_path = 'salary_model.pkl'
model = load(model_path)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input from form data
        years_experience = float(request.form['YearsExperience'])
        input_features = np.array([[years_experience]])

        # Make prediction
        predicted_salary = model.predict(input_features)[0]

        # Return prediction
        return render_template('index.html', prediction_text=f'Predicted Salary: ${round(predicted_salary, 2)}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
