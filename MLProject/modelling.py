# =========================================================
# MODELLING MACHINE LEARNING
# =========================================================

# =========================================================
# IMPORT LIBRARY
# =========================================================

import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns

import joblib

# =========================================================
# SET MLFLOW TRACKING URI
# =========================================================

mlflow.set_tracking_uri(
    "file:./mlruns"
)

# =========================================================
# CREATE EXPERIMENT
# =========================================================

mlflow.set_experiment(
    "Customer Churn Modelling"
)

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv(
    'Churn_Modelling_preprocessing.csv'
)

print("=" * 50)
print("DATASET PREVIEW")
print("=" * 50)

print(df.head())

# =========================================================
# SPLIT FEATURE DAN TARGET
# =========================================================

X = df.drop(
    'Exited',
    axis=1
)

y = df['Exited']

# =========================================================
# SPLIT TRAIN TEST
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 50)
print("DATA SPLITTING")
print("=" * 50)

print(f"X_train shape : {X_train.shape}")
print(f"X_test shape  : {X_test.shape}")
print(f"y_train shape : {y_train.shape}")
print(f"y_test shape  : {y_test.shape}")

# =========================================================
# MODEL
# =========================================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

# =========================================================
# TRAINING
# =========================================================

model.fit(
    X_train,
    y_train
)

# =========================================================
# PREDICTION
# =========================================================

y_pred = model.predict(
    X_test
)

# =========================================================
# EVALUATION
# =========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

# =========================================================
# LOG PARAMETERS
# =========================================================

mlflow.log_param(
    'n_estimators',
    200
)

mlflow.log_param(
    'max_depth',
    10
)

mlflow.log_param(
    'random_state',
    42
)

# =========================================================
# LOG METRICS
# =========================================================

mlflow.log_metric(
    'accuracy',
    accuracy
)

mlflow.log_metric(
    'precision',
    precision
)

mlflow.log_metric(
    'recall',
    recall
)

mlflow.log_metric(
    'f1_score',
    f1
)

# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.savefig(
    'confusion_matrix.png'
)

plt.close()

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

report = classification_report(
    y_test,
    y_pred
)

print("\n" + "=" * 50)
print("CLASSIFICATION REPORT")
print("=" * 50)

print(report)

with open(
    'classification_report.txt',
    'w'
) as f:

    f.write(report)

# =========================================================
# SAVE MODEL
# =========================================================

joblib.dump(
    model,
    'best_model.pkl'
)

# =========================================================
# LOG ARTIFACT
# =========================================================

mlflow.log_artifact(
    'confusion_matrix.png'
)

mlflow.log_artifact(
    'classification_report.txt'
)

mlflow.log_artifact(
    'best_model.pkl'
)

# =========================================================
# LOG MODEL
# =========================================================

mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path='model'
)

# =========================================================
# FINISH
# =========================================================

print("\n" + "=" * 50)
print("TRAINING SELESAI")
print("=" * 50)

print(f"Accuracy : {accuracy:.4f}")