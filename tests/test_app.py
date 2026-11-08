"""
Automated Test Suite for Diabetes Risk Assessment Flask Application
Validates simplified UI, model inference, age constraints (20-60), routes, and expandable sections.
"""

import os
import sys
import json
import joblib
import pytest
import pandas as pd
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app, MODEL_PATH, METRICS_PATH, validate_inputs


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_model_and_metrics_files_exist():
    """Verify saved model and metrics files exist on disk."""
    assert os.path.exists(MODEL_PATH), "diabetes_model.pkl should exist"
    assert os.path.exists(METRICS_PATH), "model_metrics.json should exist"


def test_metrics_content():
    """Verify metrics JSON contains real values and 5 features."""
    with open(METRICS_PATH, 'r', encoding='utf-8') as f:
        metrics = json.load(f)

    assert metrics['selected_model'] == "Support Vector Machine (SVM)"
    assert metrics['feature_names'] == ['Glucose', 'BMI', 'BloodPressure', 'Insulin', 'Age']
    assert metrics['dataset']['filtered_records'] == 741
    assert metrics['dataset']['age_20_records_found'] == 0
    assert metrics['test_metrics']['accuracy'] > 0.70
    assert metrics['test_metrics']['precision'] > 0.65
    assert metrics['test_metrics']['f1_score'] > 0.55


def test_pipeline_exact_five_features_inference():
    """Test that the loaded model pipeline requires and processes exactly 5 features."""
    pipeline = joblib.load(MODEL_PATH)

    # Higher risk case
    high_risk_sample = pd.DataFrame([{
        'Glucose': 185.0,
        'BMI': 38.2,
        'BloodPressure': 92.0,
        'Insulin': 240.0,
        'Age': 52.0
    }])
    pred_high = pipeline.predict(high_risk_sample)
    assert pred_high[0] == 1, "Expected high risk prediction for elevated metabolic markers"

    # Lower risk case
    low_risk_sample = pd.DataFrame([{
        'Glucose': 88.0,
        'BMI': 21.5,
        'BloodPressure': 68.0,
        'Insulin': 65.0,
        'Age': 25.0
    }])
    pred_low = pipeline.predict(low_risk_sample)
    assert pred_low[0] == 0, "Expected low risk prediction for baseline healthy markers"


def test_input_validation_logic():
    """Test input validation helper function."""
    valid_data = {
        'Glucose': '120',
        'BMI': '24.5',
        'BloodPressure': '80',
        'Insulin': '85',
        'Age': '35'
    }
    clean, err = validate_inputs(valid_data)
    assert err is None
    assert clean['Age'] == 35.0
    assert clean['Glucose'] == 120.0

    # Age below 20 (e.g. 19)
    underage_data = valid_data.copy()
    underage_data['Age'] = '19'
    clean, err = validate_inputs(underage_data)
    assert clean is None
    assert "Age must be between 20 and 60" in err

    # Age above 60 (e.g. 61)
    overage_data = valid_data.copy()
    overage_data['Age'] = '61'
    clean, err = validate_inputs(overage_data)
    assert clean is None
    assert "Age must be between 20 and 60" in err

    # Valid boundary: Age = 20
    b_min_data = valid_data.copy()
    b_min_data['Age'] = '20'
    clean, err = validate_inputs(b_min_data)
    assert err is None
    assert clean['Age'] == 20.0

    # Valid boundary: Age = 60
    b_max_data = valid_data.copy()
    b_max_data['Age'] = '60'
    clean, err = validate_inputs(b_max_data)
    assert err is None
    assert clean['Age'] == 60.0

    # Invalid non-numeric input
    invalid_num = valid_data.copy()
    invalid_num['Glucose'] = 'abc'
    clean, err = validate_inputs(invalid_num)
    assert clean is None
    assert "must be a valid numerical value" in err

    # Invalid zero for glucose
    zero_glucose = valid_data.copy()
    zero_glucose['Glucose'] = '0'
    clean, err = validate_inputs(zero_glucose)
    assert clean is None
    assert "cannot be zero" in err


def test_homepage_route(client):
    """Test simplified homepage renders properly with status 200 and simple content."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"Diabetes Risk Assessment" in response.data
    assert b"Check Your Risk" in response.data
    assert b"20" in response.data and b"60" in response.data


def test_diabetes_get_route(client):
    """Test simplified prediction form page renders with all five input fields and Clear button."""
    response = client.get('/diabetes')
    assert response.status_code == 200
    assert b'name="Glucose"' in response.data
    assert b'name="BMI"' in response.data
    assert b'name="BloodPressure"' in response.data
    assert b'name="Insulin"' in response.data
    assert b'name="Age"' in response.data
    assert b"Check Result" in response.data
    assert b"Clear" in response.data


def test_diabetes_post_valid_lower_risk(client):
    """Test POST prediction with lower risk inputs."""
    data = {
        'Glucose': '85',
        'BMI': '21.0',
        'BloodPressure': '70',
        'Insulin': '50',
        'Age': '24'
    }
    response = client.post('/diabetes', data=data)
    assert response.status_code == 200
    assert b"Your readings show a lower-risk pattern" in response.data
    assert b"Based on the five health values you entered, the machine learning model identified a lower-risk pattern." in response.data
    assert b"What does this mean?" in response.data
    assert b"The values you provided were classified into the lower-risk category by this model. This does not guarantee that you are free from diabetes." in response.data
    assert b"This is an educational machine learning prediction, not a medical diagnosis. Please consult a qualified healthcare professional for medical advice." in response.data
    assert b"Medical Notice:" in response.data
    assert b"Check Another Patient Reading" in response.data


def test_diabetes_post_valid_higher_risk(client):
    """Test POST prediction with higher risk inputs."""
    data = {
        'Glucose': '185',
        'BMI': '38.0',
        'BloodPressure': '92',
        'Insulin': '220',
        'Age': '52'
    }
    response = client.post('/diabetes', data=data)
    assert response.status_code == 200
    assert b"Your readings show a higher-risk pattern" in response.data
    assert b"Based on the five health values you entered, the machine learning model identified a higher-risk pattern." in response.data
    assert b"What does this mean?" in response.data
    assert b"Your entered values were classified into the higher-risk category by this model. This does not mean that you definitely have diabetes." in response.data
    assert b"Consider consulting a healthcare professional and getting appropriate medical tests to understand your health status." in response.data
    assert b"This is an educational machine learning prediction, not a medical diagnosis. Please consult a qualified healthcare professional for medical advice." in response.data
    assert b"Medical Notice:" in response.data
    assert b"Check Another Patient Reading" in response.data


def test_diabetes_post_age_boundary_below_20(client):
    """Test POST prediction with Age = 19 (below 20) triggers error message."""
    data = {
        'Glucose': '110',
        'BMI': '25.0',
        'BloodPressure': '75',
        'Insulin': '90',
        'Age': '19'
    }
    response = client.post('/diabetes', data=data)
    assert response.status_code == 200
    assert b"Age must be between 20 and 60" in response.data


def test_diabetes_post_age_boundary_above_60(client):
    """Test POST prediction with Age = 61 (above 60) triggers error message."""
    data = {
        'Glucose': '110',
        'BMI': '25.0',
        'BloodPressure': '75',
        'Insulin': '90',
        'Age': '61'
    }
    response = client.post('/diabetes', data=data)
    assert response.status_code == 200
    assert b"Age must be between 20 and 60" in response.data


def test_diabetes_post_age_boundaries_accepted(client):
    """Test POST prediction accepts Age = 20 and Age = 60."""
    # Age 20
    data_20 = {
        'Glucose': '100',
        'BMI': '22.0',
        'BloodPressure': '70',
        'Insulin': '60',
        'Age': '20'
    }
    res_20 = client.post('/diabetes', data=data_20)
    assert res_20.status_code == 200
    assert (
        b"Your readings show a lower-risk pattern" in res_20.data
        or b"Your readings show a higher-risk pattern" in res_20.data
    )

    # Age 60
    data_60 = {
        'Glucose': '105',
        'BMI': '23.0',
        'BloodPressure': '72',
        'Insulin': '65',
        'Age': '60'
    }
    res_60 = client.post('/diabetes', data=data_60)
    assert res_60.status_code == 200
    assert (
        b"Your readings show a lower-risk pattern" in res_60.data
        or b"Your readings show a higher-risk pattern" in res_60.data
    )


def test_about_page_and_collapsible_sections(client):
    """Test About page renders with all five collapsible information sections."""
    response = client.get('/about')
    assert response.status_code == 200
    assert b"A. How Does This System Work?" in response.data
    assert b"B. What Do These Five Parameters Mean?" in response.data
    assert b"C. Model Performance" in response.data
    assert b"D. About This Project" in response.data
    assert b"E. Medical Disclaimer" in response.data


def test_legacy_routes_redirect(client):
    """Test legacy Parkinson's and Heart Disease routes redirect to diabetes page."""
    res_heart = client.get('/heartdisease/')
    assert res_heart.status_code == 302
    assert '/diabetes' in res_heart.location

    res_park = client.get('/parkinsons/')
    assert res_park.status_code == 302
    assert '/diabetes' in res_park.location
