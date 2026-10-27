# xgboost_model.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from xgboost import XGBClassifier


# ============================================================
# 1. SETTINGS
# ============================================================

DATA_PATH = "data/combined_dataset.csv"

RANDOM_STATE = 42


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("XGBOOST - VACCINE COLD-CHAIN RISK PREDICTION")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

# ============================================================
# 3. FEATURES AND TARGET
# ============================================================

FEATURES = [
    "Temperature_C",
    "Humidity_percent",
    "Storage_Duration_hours",
    "Transport_Duration_hours",
    "Num_Excursions",
    "Excursion_Duration_hours",
    "Temperature_Deviation_C",
    "Cumulative_Exposure"
]

TARGET = "Risk_Level"

X = df[FEATURES]
y = df[TARGET]

print("\nFeatures used:")
for feature in FEATURES:
    print("-", feature)

print("\nTarget:")
print("-", TARGET)


# ============================================================
# 4. ENCODE TARGET LABELS
# ============================================================

# XGBoost works with integer class labels.
# This keeps the original Risk_Level categories unchanged
# while converting them internally to 0, 1, 2.

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nClass mapping:")

for encoded, original in enumerate(label_encoder.classes_):
    print(f"{original} -> {encoded}")


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 6. XGBOOST MODEL
# ============================================================

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=len(label_encoder.classes_),
    eval_metric="mlogloss",
    random_state=RANDOM_STATE,
    n_jobs=-1
)


# ============================================================
# 7. TRAIN MODEL
# ============================================================

print("\nTraining XGBoost...")

xgb_model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# 8. PREDICTION
# ============================================================

y_pred = xgb_model.predict(X_test)


# ============================================================
# 9. EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


print("\n" + "=" * 60)
print("XGBOOST RESULTS")
print("=" * 60)

print(f"Accuracy          : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision         : {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall            : {recall:.4f} ({recall * 100:.2f}%)")
print(f"Weighted F1-score : {f1:.4f} ({f1 * 100:.2f}%)")


# ============================================================
# 10. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

os.makedirs("figures", exist_ok=True)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)

disp.plot()

plt.title("XGBoost Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "figures/xgboost_confusion_matrix.png",
    dpi=300
)

plt.show()


# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================

importance = xgb_model.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": importance
}).sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")

print(
    feature_importance_df.to_string(
        index=False
    )
)


# ============================================================
# 13. FEATURE IMPORTANCE GRAPH
# ============================================================

plt.figure(figsize=(9, 5))

plt.bar(
    feature_importance_df["Feature"],
    feature_importance_df["Importance"]
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("XGBoost Feature Importance")

plt.tight_layout()

plt.savefig(
    "figures/xgboost_feature_importance.png",
    dpi=300
)

plt.show()


# ============================================================
# 14. SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "Weighted F1-score"
    ],
    "XGBoost": [
        accuracy,
        precision,
        recall,
        f1
    ]
})

results.to_csv(
    "figures/xgboost_results.csv",
    index=False
)

print("\nResults saved to:")
print("figures/xgboost_results.csv")

print("\nXGBoost experiment completed!")