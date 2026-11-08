"""
Diabetes Disease Prediction for the Age Group Between 20 to 60
Flask Web Application

Adapted from the public Multiple-Disease-Predictor-ML-Flask-WebApp project:
https://github.com/sidroy9/Multiple-Disease-Predictor-ML-Flask-WebApp
Refactored into a focused, highly accurate diabetes risk screening tool.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
APP_ENV = os.environ.get('APP_ENV', 'development').strip().lower()
secret_key = os.environ.get('SECRET_KEY')

if not secret_key:
    if APP_ENV == 'production':
        raise RuntimeError('APP_ENV=production requires SECRET_KEY to be set in the environment.')
    if APP_ENV != 'development':
        raise RuntimeError('SECRET_KEY must be set unless APP_ENV=development.')
    secret_key = 'development-only-secret-do-not-use-in-production'

app.config.update(APP_ENV=APP_ENV, SECRET_KEY=secret_key)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'diabetes_model.pkl')
METRICS_PATH = os.path.join(BASE_DIR, 'model_metrics.json')

# Load trained pipeline model
diabetes_pipeline = None
if os.path.exists(MODEL_PATH):
    try:
        diabetes_pipeline = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading {MODEL_PATH}: {e}")

# Load model evaluation metrics
model_metrics = {}
if os.path.exists(METRICS_PATH):
    try:
        with open(METRICS_PATH, 'r', encoding='utf-8') as f:
            model_metrics = json.load(f)
    except Exception as e:
        print(f"Error loading {METRICS_PATH}: {e}")


def validate_inputs(form_data):
    """
    Validates form inputs for the 5 selected clinical features:
    Glucose, BMI, BloodPressure, Insulin, Age.

    Returns:
        tuple: (dict_of_clean_values, error_message_or_None)
    """
    errors = []
    clean_values = {}

    # Feature definitions: (display_name, min_val, max_val, allow_zero)
    feature_specs = {
        'Glucose': ('Glucose', 40.0, 400.0, False),
        'BMI': ('BMI (Body Mass Index)', 10.0, 80.0, False),
        'BloodPressure': ('Blood Pressure', 30.0, 250.0, False),
        'Insulin': ('Insulin', 0.0, 900.0, True),
        'Age': ('Age', 20.0, 60.0, False)
    }

    # Verify presence of all keys
    for field, (label, min_val, max_val, allow_zero) in feature_specs.items():
        raw_val = form_data.get(field, '').strip() if field in form_data else form_data.get(field.lower(), '').strip()
        
        # Check alias keys field1..field5 if coming from older forms
        if not raw_val:
            alias_map = {'Glucose': 'field1', 'BMI': 'field2', 'BloodPressure': 'field3', 'Insulin': 'field4', 'Age': 'field5'}
            raw_val = form_data.get(alias_map.get(field, ''), '').strip()

        if not raw_val:
            errors.append(f"{label} is required.")
            continue

        try:
            val = float(raw_val)
        except ValueError:
            errors.append(f"{label} must be a valid numerical value.")
            continue

        # Biological validation
        if not allow_zero and val <= 0:
            errors.append(f"{label} cannot be zero or negative. A valid clinical reading is required.")
            continue
        elif allow_zero and val < 0:
            errors.append(f"{label} cannot be negative.")
            continue

        # Strict Age requirement (20 to 60 inclusive)
        if field == 'Age':
            if val < 20 or val > 60:
                errors.append(f"Age must be between 20 and 60 years inclusive for this targeted clinical model (received {val:g}).")
                continue
        else:
            if val < min_val or val > max_val:
                errors.append(f"{label} reading of {val:g} is outside realistic biological bounds ({min_val:g}–{max_val:g}).")
                continue

        clean_values[field] = val

    if errors:
        return None, " | ".join(errors)

    return clean_values, None


@app.route('/')
def home():
    """Homepage route displaying project overview."""
    return render_template(
        'home.html',
        metrics=model_metrics,
        page_title="Home"
    )


@app.route('/about')
def about():
    """About page with collapsible educational and technical details."""
    return render_template(
        'about.html',
        metrics=model_metrics,
        page_title="About"
    )


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Clean alias for prediction page."""
    return diabetes()


@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    """Diabetes risk assessment page route (handles GET form display and POST inference)."""
    prediction_result = None
    prediction_code = None
    explanation = None
    error_message = None
    form_values = {}

    if request.method == 'POST':
        form_values = {
            'Glucose': request.form.get('Glucose', request.form.get('glucose', '')),
            'BMI': request.form.get('BMI', request.form.get('bmi', '')),
            'BloodPressure': request.form.get('BloodPressure', request.form.get('bloodpressure', '')),
            'Insulin': request.form.get('Insulin', request.form.get('insulin', '')),
            'Age': request.form.get('Age', request.form.get('age', ''))
        }

        clean_data, error_message = validate_inputs(request.form)

        if error_message:
            flash(error_message, 'danger')
        elif diabetes_pipeline is None:
            error_message = "Machine learning model file (diabetes_model.pkl) is not loaded. Please train the model first."
            flash(error_message, 'danger')
        else:
            # Build DataFrame with the exact five features in order
            input_df = pd.DataFrame([{
                'Glucose': clean_data['Glucose'],
                'BMI': clean_data['BMI'],
                'BloodPressure': clean_data['BloodPressure'],
                'Insulin': clean_data['Insulin'] if clean_data['Insulin'] > 0 else np.nan,
                'Age': clean_data['Age']
            }])

            try:
                pred = int(diabetes_pipeline.predict(input_df)[0])
                prediction_code = pred
                if pred == 1:
                    prediction_result = "Higher Predicted Diabetes Risk Pattern"
                    explanation = "The Machine Learning model classified your input as a higher-risk pattern."
                else:
                    prediction_result = "Lower Predicted Diabetes Risk Pattern"
                    explanation = "The Machine Learning model classified your input as a lower-risk pattern."
            except Exception as e:
                error_message = f"Error during model inference: {str(e)}"
                flash(error_message, 'danger')

    return render_template(
        'diabetes.html',
        metrics=model_metrics,
        prediction_result=prediction_result,
        prediction_code=prediction_code,
        explanation=explanation,
        error_message=error_message,
        form_values=form_values,
        page_title="Predict"
    )


@app.route('/predictdiabetes/', methods=['POST'])
def predictdiabetes():
    """POST endpoint for form submission compatibility."""
    return diabetes()


# Redirect legacy disease routes cleanly to the diabetes assessment page
@app.route('/heartdisease/')
@app.route('/parkinsons/')
def legacy_redirect():
    return redirect(url_for('diabetes'))


if __name__ == '__main__':
    # Run development server
    app.run(host='127.0.0.1', port=5000, debug=True)
