from flask import Flask, request, render_template
import pandas as pd
from preprocess import preprocess_input
from predictor import predict_disease

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = {
        'age': int(request.form['age']),
        'bmi': float(request.form['bmi']),
        'blood_pressure': int(request.form['bp']),
        'glucose_level': int(request.form['glucose']),
        'cholesterol': int(request.form['cholesterol']),
        'gender': request.form['gender'],
        'smoking_status': request.form['smoking'],
        'physical_activity': request.form['activity'],
        'diabetes_history': int(request.form['diabetes']),
        'heart_disease': int(request.form['heart']),
    }
    
    df = pd.DataFrame([input_data])
    processed = preprocess_input(df)
    result, prob = predict_disease(processed)
    
    return render_template('index.html', prediction=result, probability=round(prob * 100, 2))

if __name__ == '__main__':
    app.run(debug=True)
