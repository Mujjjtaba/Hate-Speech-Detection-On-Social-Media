# backend/app.py

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Load models and vectorizer
logreg_model = joblib.load('backend/model_logreg.pkl')
xgb_model = joblib.load('backend/model_xgb.pkl')
vectorizer = joblib.load('backend/vectorizer.pkl')

# Load model accuracies
with open('backend/model_accuracy.txt', 'r') as f:
    acc_lines = f.readlines()
logreg_acc = float(acc_lines[0].split(':')[-1].strip())
xgb_acc = float(acc_lines[1].split(':')[-1].strip())

# Set up Flask app
app = Flask(__name__, template_folder='../frontend/templates', static_folder='../frontend/static')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form.get('text')
    if not input_text:
        return jsonify({'error': 'No input text provided'}), 400

    # Vectorize input
    vectorized_text = vectorizer.transform([input_text])

    # Predict
    logreg_pred = logreg_model.predict(vectorized_text)[0]
    xgb_pred = xgb_model.predict(vectorized_text)[0]

    result = {
        'input_text': input_text,
        'logistic_regression': {
            'prediction': 'Hate Speech' if logreg_pred == 1 else 'Not Hate Speech',
            'accuracy': f"{logreg_acc:.2%}"
        },
        'xgboost': {
            'prediction': 'Hate Speech' if xgb_pred == 1 else 'Not Hate Speech',
            'accuracy': f"{xgb_acc:.2%}"
        }
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
