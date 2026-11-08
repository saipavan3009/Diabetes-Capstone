"""
Diabetes Disease Prediction for the Age Group Between 20 to 60
Machine Learning Model Training and Evaluation Pipeline

This script:
1. Loads diabetes.csv and filters records for age group 20 to 60 inclusive.
2. Selects exactly 5 features in order: Glucose, BMI, BloodPressure, Insulin, Age.
3. Treats biologically invalid zero values as missing and imputes with median.
4. Uses stratified train-test split (80/20, random_state=42).
5. Compares Logistic Regression, Support Vector Machine (SVM), and Random Forest
   using 5-fold Stratified Cross-Validation (F1-score on training set).
6. Selects the best performing model based on CV F1-score.
7. Evaluates the chosen model on the independent test dataset.
8. Saves the final pipeline to diabetes_model.pkl using joblib.
9. Exports all metrics and dataset statistics to model_metrics.json.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ---------------------------------------------------------
# 1. Dataset Loading and Inspection
# ---------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), 'diabetes.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'diabetes_model.pkl')
METRICS_PATH = os.path.join(os.path.dirname(__file__), 'model_metrics.json')

print("=" * 70)
print("DIABETES DISEASE PREDICTION PIPELINE (AGE 20 - 60)")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
original_total = len(df)
print(f"[Dataset] Total original records loaded: {original_total}")

# Verify age 20 presence in dataset
age_20_count = int((df['Age'] == 20).sum())
print(f"[Dataset] Number of records with Age == 20: {age_20_count}")
if age_20_count == 0:
    print("  -> Confirmed: The dataset contains no records of age exactly 20.")
    print(f"  -> Observed age range in raw dataset: {df['Age'].min()} to {df['Age'].max()} years.")

# Filter age 20 to 60 inclusive
df_filtered = df[(df['Age'] >= 20) & (df['Age'] <= 60)].copy()
filtered_total = len(df_filtered)
excluded_total = original_total - filtered_total
print(f"[Dataset] Records within age range 20-60 inclusive: {filtered_total}")
print(f"[Dataset] Records excluded (Age < 20 or Age > 60): {excluded_total}")

# ---------------------------------------------------------
# 2. Feature Selection & Missing / Zero Value Handling
# ---------------------------------------------------------
# Selected 5 features in exact specified order
FEATURES = ['Glucose', 'BMI', 'BloodPressure', 'Insulin', 'Age']
TARGET = 'Outcome'

print(f"\n[Features] Selected features ({len(FEATURES)}): {FEATURES}")

# Zero value inspection in clinical variables where zero is invalid
ZERO_INVALID_COLS = ['Glucose', 'BMI', 'BloodPressure', 'Insulin']
zero_counts = {}
for col in ZERO_INVALID_COLS:
    cnt = int((df_filtered[col] == 0).sum())
    pct = (cnt / filtered_total) * 100
    zero_counts[col] = {"zero_count": cnt, "percentage": round(pct, 2)}
    print(f"  - {col}: {cnt} invalid zeros ({pct:.2f}%)")

# Replace invalid zero values with NaN for median imputation in pipeline
df_filtered[ZERO_INVALID_COLS] = df_filtered[ZERO_INVALID_COLS].replace(0, np.nan)

X = df_filtered[FEATURES]
y = df_filtered[TARGET]

# ---------------------------------------------------------
# 3. Train-Test Split (Stratified, random_state=42)
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\n[Split] Train set size: {len(X_train)} records")
print(f"[Split] Test set size:  {len(X_test)} records")
print(f"[Split] Target distribution in train: Negative={int((y_train == 0).sum())}, Positive={int((y_train == 1).sum())}")
print(f"[Split] Target distribution in test:  Negative={int((y_test == 0).sum())}, Positive={int((y_test == 1).sum())}")

# ---------------------------------------------------------
# 4. Pipeline Definitions and Cross-Validation
# ---------------------------------------------------------
# 5-fold Stratified Cross-Validation on training data
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

candidate_pipelines = {
    'Logistic Regression': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ]),
    'Support Vector Machine (SVM)': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', SVC(kernel='linear', random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
    ])
}

print("\n" + "=" * 70)
print("5-FOLD STRATIFIED CROSS-VALIDATION (Training Set)")
print("=" * 70)

cv_results = {}
best_model_name = None
best_cv_f1 = -1.0

for name, pipeline in candidate_pipelines.items():
    f1_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1')
    acc_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
    mean_f1 = float(np.mean(f1_scores))
    std_f1 = float(np.std(f1_scores))
    mean_acc = float(np.mean(acc_scores))
    std_acc = float(np.std(acc_scores))

    cv_results[name] = {
        "mean_f1": round(mean_f1, 4),
        "std_f1": round(std_f1, 4),
        "mean_accuracy": round(mean_acc, 4),
        "std_accuracy": round(std_acc, 4),
        "fold_f1_scores": [round(float(s), 4) for s in f1_scores]
    }

    print(f"{name:30} | Mean F1: {mean_f1:.4f} (+/- {std_f1:.4f}) | Mean Acc: {mean_acc:.4f}")

    if mean_f1 > best_cv_f1:
        best_cv_f1 = mean_f1
        best_model_name = name

print("-" * 70)
print(f"Selected Best Model based on CV F1-Score: {best_model_name} (F1 = {best_cv_f1:.4f})")

# ---------------------------------------------------------
# 5. Final Model Training & Evaluation on Test Set
# ---------------------------------------------------------
best_pipeline = candidate_pipelines[best_model_name]
best_pipeline.fit(X_train, y_train)

y_pred = best_pipeline.predict(X_test)

test_acc = float(accuracy_score(y_test, y_pred))
test_prec = float(precision_score(y_test, y_pred, zero_division=0))
test_rec = float(recall_score(y_test, y_pred, zero_division=0))
test_f1 = float(f1_score(y_test, y_pred, zero_division=0))
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = [int(v) for v in cm.ravel()]

print("\n" + "=" * 70)
print(f"FINAL TEST SET EVALUATION ({best_model_name})")
print("=" * 70)
print(f"  Accuracy:  {test_acc:.4f} ({test_acc * 100:.2f}%)")
print(f"  Precision: {test_prec:.4f} ({test_prec * 100:.2f}%)")
print(f"  Recall:    {test_rec:.4f} ({test_rec * 100:.2f}%)")
print(f"  F1-Score:  {test_f1:.4f} ({test_f1 * 100:.2f}%)")
print("  Confusion Matrix:")
print(f"    [[TN={tn}, FP={fp}],")
print(f"     [FN={fn}, TP={tp}]]")

# ---------------------------------------------------------
# 6. Model Persistence and Metrics Export
# ---------------------------------------------------------
# Save pipeline using joblib as diabetes_model.pkl
joblib.dump(best_pipeline, MODEL_PATH)
print(f"\n[Artifact] Saved model pipeline to: {MODEL_PATH}")

# Also keep diabetes.pkl for compatibility
joblib.dump(best_pipeline, os.path.join(os.path.dirname(__file__), 'diabetes.pkl'))

metrics_data = {
    "project_title": "Diabetes Disease Prediction for the Age Group Between 20 to 60 Using Machine Learning Flask Web Application",
    "selected_model": best_model_name,
    "feature_names": FEATURES,
    "dataset": {
        "dataset_file": "diabetes.csv",
        "total_original_records": original_total,
        "age_filter": "20 to 60 inclusive",
        "filtered_records": filtered_total,
        "excluded_records": excluded_total,
        "age_20_records_found": age_20_count,
        "observed_min_age": int(df['Age'].min()),
        "observed_max_age": int(df['Age'].max()),
        "age_documentation": "Inspection of diabetes.csv confirms min age is 21; exactly 0 records exist with Age == 20.",
        "training_records": len(X_train),
        "testing_records": len(X_test),
        "zero_value_analysis": zero_counts
    },
    "cross_validation": {
        "k_folds": 5,
        "strategy": "StratifiedKFold",
        "selection_metric": "F1-Score",
        "random_state": 42,
        "results": cv_results
    },
    "test_metrics": {
        "accuracy": round(test_acc, 4),
        "precision": round(test_prec, 4),
        "recall": round(test_rec, 4),
        "f1_score": round(test_f1, 4),
        "accuracy_percent": round(test_acc * 100, 2),
        "precision_percent": round(test_prec * 100, 2),
        "recall_percent": round(test_rec * 100, 2),
        "f1_score_percent": round(test_f1 * 100, 2),
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp
        }
    }
}

with open(METRICS_PATH, 'w', encoding='utf-8') as f:
    json.dump(metrics_data, f, indent=4)

print(f"[Artifact] Saved model metrics to: {METRICS_PATH}")
print("=" * 70)
print("TRAINING PIPELINE COMPLETED SUCCESSFULLY.")
print("=" * 70)
