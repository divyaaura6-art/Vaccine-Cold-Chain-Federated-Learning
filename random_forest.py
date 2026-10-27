# random_forest.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

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
print("RANDOM FOREST - VACCINE COLD-CHAIN RISK PREDICTION")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

# ============================================================
# 3. SELECT FEATURES AND TARGET
# ============================================================

# Same 8 numerical features used by the centralized MLP
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
# 4. TRAIN-TEST SPLIT
# ============================================================

# Same 80/20 stratified split idea as the centralized MLP
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))

# ============================================================
# 5. RANDOM FOREST MODEL
# ============================================================

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight=None
)

# ============================================================
# 6. TRAIN MODEL
# ============================================================

print("\nTraining Random Forest...")

rf_model.fit(X_train, y_train)

print("Training completed.")

# ============================================================
# 7. PREDICTION
# ============================================================

y_pred = rf_model.predict(X_test)

# ============================================================
# 8. EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

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
print("RANDOM FOREST RESULTS")
print("=" * 60)

print(f"Accuracy          : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision         : {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall            : {recall:.4f} ({recall * 100:.2f}%)")
print(f"Weighted F1-score : {f1:.4f} ({f1 * 100:.2f}%)")

# ============================================================
# 9. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Create figures directory if it doesn't exist
os.makedirs("figures", exist_ok=True)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=rf_model.classes_
)

disp.plot()
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "figures/random_forest_confusion_matrix.png",
    dpi=300
)

plt.show()

# ============================================================
# 11. FEATURE IMPORTANCE
# ============================================================

importance = rf_model.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": importance
}).sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance_df.to_string(index=False))

# Plot feature importance
plt.figure(figsize=(9, 5))

plt.bar(
    feature_importance_df["Feature"],
    feature_importance_df["Importance"]
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.ylabel("Importance")
plt.xlabel("Feature")
plt.title("Random Forest Feature Importance")

plt.tight_layout()

plt.savefig(
    "figures/random_forest_feature_importance.png",
    dpi=300
)

plt.show()

# ============================================================
# 12. SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "Weighted F1-score"
    ],
    "Random Forest": [
        accuracy,
        precision,
        recall,
        f1
    ]
})

results.to_csv(
    "figures/random_forest_results.csv",
    index=False
)

print("\nResults saved to:")
print("figures/random_forest_results.csv")

print("\nDone!")