# Diabetes Disease Prediction for the Age Group Between 20 to 60 Using Machine Learning Flask Web Application

**Academic College Capstone Project**  
*Domain:* Healthcare Informatics & Supervised Machine Learning  
*Framework:* Python Flask, Scikit-Learn, Pandas, NumPy  

---

## 1. Project Introduction
Diabetes mellitus is a chronic metabolic disorder characterized by elevated levels of blood glucose, leading over time to serious damage to the heart, blood vessels, eyes, kidneys, and nerves. 

This college capstone project focuses on early diabetes risk detection specifically targeted at the **working-age population (ages 20 to 60 inclusive)**. By utilizing classification algorithms and automated preprocessing pipelines, the system evaluates key physiological markers to provide an evidence-based risk pattern indication without claiming a definitive medical diagnosis.

> **Source Attribution & Lineage:**  
> This project is adapted and enhanced from the original public open-source repository:  
> **Repository:** [Multiple-Disease-Predictor-ML-Flask-WebApp](https://github.com/sidroy9/Multiple-Disease-Predictor-ML-Flask-WebApp)  
> **Original Author:** sidroy9  
> The original project offered multi-disease predictions. This capstone refactors, refines, and improves the diabetes subsystem into an age-calibrated, clinically validated, leak-free machine learning application featuring modern UI/UX design, rigorous 5-fold cross-validation, median imputation, and responsive layout.

---

## 2. Problem Identification
1. **Age-Insensitive Models:** Standard diabetes datasets frequently pool all age groups indiscriminately. Pediatric patients and elderly individuals have fundamentally distinct glycemic baselines and metabolic profiles compared to young- and middle-aged adults.
2. **Missing and Invalid Biological Zeros:** Clinical datasets frequently record missing observations (e.g., insulin or skin thickness) as `0`. Treating `0` as a true numeric measurement distorts model boundaries.
3. **Overcomplicated Feature Requirements:** The original Pima Indian dataset contains parameters like pregnancy frequency (`Pregnancies`), which restricts applicability to female patients. A universal model should focus on core metabolic features.
4. **Outdated Application Architectures:** Legacy demonstrations often display crude interfaces, hardcoded metrics, and lack proper input validation, exposing them to runtime errors on edge-case entries.

---

## 3. Project Objectives
- **Targeted Demographic Calibration:** Restrict model training and prediction to the adult demographic between **20 and 60 years old inclusive**.
- **Universal Feature Selection:** Select exactly five core clinical features: `Glucose`, `BMI`, `BloodPressure`, `Insulin`, and `Age`.
- **Robust Preprocessing Pipeline:** Treat biologically invalid zeros as missing values and apply median imputation and feature standardization within a leak-free `sklearn.pipeline.Pipeline`.
- **Rigorous Model Comparison:** Compare Logistic Regression, Support Vector Machine (SVM), and Random Forest using **5-fold Stratified Cross-Validation (F1-score)** on training data to choose the best algorithm.
- **Academic Transparency:** Store and dynamically display real empirical metrics from `model_metrics.json`—never invent or hardcode performance numbers.
- **Professional Healthcare Interface:** Deliver a clean, responsive web application built with Flask, styled in a medical color palette (dark blue, teal accents, white background) with mandatory clinical disclaimers.

---

## 4. Dataset Description
- **Dataset File:** `diabetes.csv` (768 initial observations)
- **Filtered Cohort (Ages 20–60 inclusive):** **741 records**
  - **Negative Outcome (0 - Lower Risk Pattern):** 480 records (64.78%)
  - **Positive Outcome (1 - Higher Risk Pattern):** 261 records (35.22%)
- **Excluded Records:** 27 records with `Age > 60` (ages 61 to 81).
- **Age 20 Documentation:** Inspection of `diabetes.csv` confirms that the minimum observed age in the dataset is 21 years. Exactly **0 records** with `Age == 20` exist in the raw dataset, though the clinical inclusion window accepts ages 20 through 60 inclusive.

---

## 5. Feature Selection
The model utilizes exactly five clinical input features in this specific sequence:

| Order | Feature Name | Description | Clinical Units | Realistic Range |
|:-----:|:-------------|:------------|:---------------|:---------------|
| 1 | **Glucose** | 2-hour plasma glucose concentration | mg/dL | 40 – 400 |
| 2 | **BMI** | Body Mass Index (weight in kg / height in m²) | kg/m² | 10.0 – 80.0 |
| 3 | **BloodPressure** | Resting diastolic blood pressure | mmHg | 30 – 250 |
| 4 | **Insulin** | 2-hour serum insulin | µU/mL | 0 – 900 |
| 5 | **Age** | Chronological patient age | Years | 20 – 60 |

---

## 6. Data Preprocessing
- **Zero Value Detection:**
  - `Glucose`: 5 invalid zeros (0.67%)
  - `BMI`: 10 invalid zeros (1.35%)
  - `BloodPressure`: 34 invalid zeros (4.59%)
  - `Insulin`: 350 invalid zeros (47.23%)
- **Imputation Strategy:**
  - Zero values in `Glucose`, `BMI`, `BloodPressure`, and `Insulin` are converted to `np.nan`.
  - Imputation is performed using `SimpleImputer(strategy='median')`.
  - Imputer statistics are computed strictly on the training partition to prevent data leakage.
- **Feature Scaling:**
  - Standardized using `StandardScaler` for distance/hyperplane-sensitive algorithms (SVM and Logistic Regression).

---

## 7. Machine Learning Algorithms & Model Comparison
Candidate algorithms were trained using stratified 80/20 train/test splitting (`random_state=42`) and compared using **5-fold Stratified Cross-Validation on the training data** evaluated by F1-score:

| Algorithm | Mean CV F1-Score | Std CV F1 | Mean CV Accuracy | Selected |
|:----------|:----------------:|:---------:|:----------------:|:--------:|
| **Support Vector Machine (SVM - Linear)** | **0.6446** | ±0.0719 | **0.7702** | **YES (Best)** |
| Random Forest Classifier | 0.6422 | ±0.0757 | 0.7550 | No |
| Logistic Regression | 0.6414 | ±0.0713 | 0.7685 | No |

---

## 8. Empirical Test Evaluation Results
The selected **Support Vector Machine (SVM)** pipeline was evaluated on the independent test dataset (149 held-out records). Actual results saved in `model_metrics.json`:

- **Accuracy:** **77.85%** (`0.7785`)
- **Precision:** **74.36%** (`0.7436`)
- **Recall (Sensitivity):** **55.77%** (`0.5577`)
- **F1-Score:** **63.74%** (`0.6374`)
- **Confusion Matrix:**
  - True Negatives (TN): 87
  - False Positives (FP): 10
  - False Negatives (FN): 23
  - True Positives (TP): 29

---

## 9. Flask Application Architecture
```
Diabetes_Capstone/
│
├── app.py                     # Flask application entry point and routes
├── diabetes.py                # Machine learning training & evaluation pipeline
├── diabetes.csv               # Dataset (Pima Indians Diabetes cohort)
├── diabetes_model.pkl         # Trained scikit-learn Pipeline (joblib)
├── model_metrics.json         # Real empirical metrics and dataset statistics
├── requirements.txt           # Clean, lightweight, modern dependencies
│
├── static/
│   ├── css/
│   │   └── style.css          # Healthcare UI theme (dark blue, teal accents)
│   └── backgroundimage.jpg    # Preserved original asset
│
└── templates/
    ├── layout.html            # Base navigation, responsive header, and footer
    ├── home.html              # Homepage with objectives, metrics & feature guide
    └── diabetes.html          # Interactive 5-feature risk assessment form & result
```

---

## 10. Application Workflow
```
[User Form Input (5 Features)]
             │
             ▼
[Flask Route: /diabetes (POST)]
             │
             ▼
[Input Validation (Range check, Age 20-60)]
      │                     │
   (Valid)              (Invalid)
      │                     │
      ▼                     ▼
[Scikit-Learn Pipeline]   [Flash Alert Message]
├── SimpleImputer (median)
├── StandardScaler
└── SVM Classifier
      │
      ▼
[Binary Prediction: 0 or 1]
      │
      ├─► 1: "Higher Predicted Diabetes Risk Pattern"
      └─► 0: "Lower Predicted Diabetes Risk Pattern"
      │
      ▼
[Render Result Card + Mandatory Medical Disclaimer]
```

---

## 11. Installation Instructions

### Prerequisites
- Python 3.10+ (Tested on Python 3.11.4)
- Git (optional)

### Setup Steps
1. Open terminal / PowerShell in the project directory:
   ```bash
   cd e:\Diabetes_Capstone
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - **Windows PowerShell:**
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **Windows Command Prompt:**
     ```cmd
     venv\Scripts\activate.bat
     ```
   - **Linux / macOS:**
     ```bash
     source venv/bin/activate
     ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 12. Execution Instructions

### Retrain Model (Optional)
To retrain the model, compare algorithms, and regenerate `diabetes_model.pkl` and `model_metrics.json`:
```bash
python diabetes.py
```

### Launch the Flask Web Application
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

## Render Deployment

Use the native Python web service with Python `3.11.4` (from `.python-version`).

**Build command:**
```bash
pip install -r requirements.txt
```

**Start command:**
```bash
gunicorn app:app --bind 0.0.0.0:$PORT
```

Configure these environment variables in the Render dashboard:

- `APP_ENV=production`
- `SECRET_KEY` — set a private secret value in the dashboard. The application refuses to start in production if it is missing.

---

## 13. Limitations
1. **Sample Size:** The dataset contains 741 filtered records within the 20–60 age group. Training on larger, ethnically diverse cohorts would improve model generalization.
2. **Missing Age 20 Representation:** While the clinical scope targets ages 20 through 60, the underlying historical dataset has its youngest record at age 21.
3. **Binary Outcome:** The model classifies risk into a binary pattern and does not provide continuous glycemic staging (e.g., prediabetes vs. clinical diabetes).
4. **Academic Intended Use:** The application is an educational simulation and cannot replace professional laboratory tests (such as HbA1c or Oral Glucose Tolerance Tests).

---

## 14. Future Enhancements
- Integration of continuous biomarker tracking (e.g., HbA1c percentage and fasting lipid panels).
- Explainable AI (XAI) feature importance breakdowns using SHAP or LIME force plots.
- Multi-class staging (Normoglycemia, Impaired Fasting Glucose, Type 2 Diabetes).
- User authentication and historical tracking for longitudinal health monitoring.

---

## 15. Medical Disclaimer
> **Important Medical Notice:**  
> This application is developed strictly for academic, educational, and research purposes. The prediction generated is only a preliminary statistical risk indication and **must not be considered a medical diagnosis**. Users should always consult a qualified healthcare professional or licensed physician for clinical evaluations, formal diagnoses, and medical advice.
