import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "data_v2/combined_dataset.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 2. SELECT FEATURES AND TARGET
# ============================================================

features = [
    "Temperature_C",
    "Humidity_percent",
    "Storage_Duration_hours",
    "Transport_Duration_hours",
    "Num_Excursions",
    "Excursion_Duration_hours",
    "Temperature_Deviation_C",
    "Cumulative_Exposure"
]

target = "Risk_Level"

X = df[features]
y = df[target]


# ============================================================
# 3. CHECK TARGET DISTRIBUTION
# ============================================================

print("\nOverall Risk Distribution:")
print(
    y.value_counts().sort_index()
)


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 5. FEATURE STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 6. BUILD MLP MODEL
# ============================================================

model = Sequential([

    Dense(
        32,
        activation="relu",
        input_shape=(len(features),)
    ),

    Dropout(0.2),

    Dense(
        16,
        activation="relu"
    ),

    Dropout(0.2),

    Dense(
        3,
        activation="softmax"
    )
])


# ============================================================
# 7. COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 8. DISPLAY MODEL
# ============================================================

print("\nModel Architecture:")
model.summary()


# ============================================================
# 9. EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)


# ============================================================
# 10. TRAIN MODEL
# ============================================================

print("\nStarting centralized training...")

history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.20,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)


# ============================================================
# 11. PREDICTION
# ============================================================

y_probability = model.predict(
    X_test_scaled,
    verbose=0
)

y_pred = np.argmax(
    y_probability,
    axis=1
)


# ============================================================
# 12. EVALUATION
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


print("\n========================================")
print("CENTRALIZED MLP RESULTS")
print("========================================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")


# ============================================================
# 13. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Low",
            "Moderate",
            "High"
        ],
        zero_division=0
    )
)


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 15. SAVE MODEL
# ============================================================

model.save(
    "centralized_mlp.keras"
)

print("\nModel saved as:")
print("centralized_mlp.keras")

print("\n========================================")
print("CENTRALIZED EXPERIMENT COMPLETED")
print("========================================")