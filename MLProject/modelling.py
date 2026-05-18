# =========================================================
# MODELLING MACHINE LEARNING
# =========================================================

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# SET EXPERIMENT
# =========================================================

mlflow.set_experiment(
    "Customer Churn Modelling"
)

# =========================================================
# ENABLE AUTOLOG
# =========================================================

mlflow.autolog()

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv(
    "Churn_Modelling_preprocessing.csv"
)

print("=" * 50)
print("DATASET PREVIEW")
print("=" * 50)

print(df.head())

# =========================================================
# FEATURES & TARGET
# =========================================================

X = df.drop(
    "Exited",
    axis=1
)

y = df["Exited"]

# =========================================================
# DATA SPLITTING
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

# =========================================================
# START MLFLOW RUN
# =========================================================

with mlflow.start_run():

    # =====================================================
    # BUILD MODEL
    # =====================================================

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    # =====================================================
    # TRAIN MODEL
    # =====================================================

    model.fit(
        X_train,
        y_train
    )

    # =====================================================
    # PREDICTION
    # =====================================================

    y_pred = model.predict(
        X_test
    )

    # =====================================================
    # EVALUATION
    # =====================================================

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
    print(f"F1 Score  : {f1:.4f}")

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

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

    plt.title("Training Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        "training_confusion_matrix.png"
    )

    plt.close()

# =========================================================
# FINISH
# =========================================================

print("\n" + "=" * 50)
print("MODELLING SELESAI")
print("=" * 50)

print(f"Accuracy : {accuracy:.4f}")