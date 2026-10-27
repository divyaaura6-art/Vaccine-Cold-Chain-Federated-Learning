import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# ============================================================
# SETTINGS
# ============================================================

PHCS = ["PHC1", "PHC2", "PHC3", "PHC4", "PHC5"]

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

NUM_ROUNDS = 20
LOCAL_EPOCHS = 3
BATCH_SIZE = 32

np.random.seed(42)
tf.random.set_seed(42)

os.makedirs("figures", exist_ok=True)


# ============================================================
# MODEL
# ============================================================

def create_model():

    model = Sequential([
        Dense(
            32,
            activation="relu",
            input_shape=(len(FEATURES),)
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

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# LOAD PHC DATA
# ============================================================

phc_data = {}

print("\nLoading PHC datasets...")

for phc in PHCS:

    path = f"data/{phc}.csv"

    df = pd.read_csv(path)

    phc_data[phc] = df

    print(
        f"{phc}: {len(df)} records loaded"
    )


# ============================================================
# CREATE GLOBAL TEST SET
# ============================================================

# Take 20% from each PHC for testing.
# The remaining 80% stays local to that PHC.

local_train_data = {}
test_parts = []

for phc in PHCS:

    df = phc_data[phc]

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df[TARGET]
    )

    local_train_data[phc] = train_df

    test_parts.append(test_df)


# Combine ONLY the held-out test portions.
# No test observations are used during local training.

global_test_df = pd.concat(
    test_parts,
    ignore_index=True
)


# ============================================================
# FIT GLOBAL SCALER
# ============================================================

# The scaler is fitted using training observations only.

all_training_data = pd.concat(
    list(local_train_data.values()),
    ignore_index=True
)

scaler = StandardScaler()

scaler.fit(
    all_training_data[FEATURES]
)


# ============================================================
# PREPARE GLOBAL TEST DATA
# ============================================================

X_test = scaler.transform(
    global_test_df[FEATURES]
)

y_test = global_test_df[TARGET].values


# ============================================================
# PREPARE LOCAL DATA
# ============================================================

local_X = {}
local_y = {}

for phc in PHCS:

    df = local_train_data[phc]

    local_X[phc] = scaler.transform(
        df[FEATURES]
    )

    local_y[phc] = df[TARGET].values


# ============================================================
# INITIALIZE GLOBAL MODEL
# ============================================================

global_model = create_model()


# ============================================================
# FEDAVG FUNCTION
# ============================================================

def fedavg(local_weights, local_sizes):

    total_samples = sum(local_sizes)

    new_weights = []

    for layer_weights in zip(*local_weights):

        weighted_layer = np.zeros_like(
            layer_weights[0]
        )

        for weights, size in zip(
            layer_weights,
            local_sizes
        ):

            weighted_layer += (
                weights * size / total_samples
            )

        new_weights.append(
            weighted_layer
        )

    return new_weights


# ============================================================
# TRAINING HISTORY
# ============================================================

history = []


# ============================================================
# FEDERATED TRAINING
# ============================================================

print("\n========================================")
print("STARTING FEDAVG TRAINING")
print("========================================")

for round_number in range(1, NUM_ROUNDS + 1):

    print(
        f"\nCommunication Round "
        f"{round_number}/{NUM_ROUNDS}"
    )

    local_weights = []
    local_sizes = []

    # --------------------------------------------------------
    # LOCAL TRAINING
    # --------------------------------------------------------

    for phc in PHCS:

        print(
            f"  Training local model at {phc}..."
        )

        local_model = create_model()

        # Start from current global model
        local_model.set_weights(
            global_model.get_weights()
        )

        local_model.fit(
            local_X[phc],
            local_y[phc],
            epochs=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0
        )

        local_weights.append(
            local_model.get_weights()
        )

        local_sizes.append(
            len(local_y[phc])
        )

    # --------------------------------------------------------
    # SERVER-SIDE FEDAVG
    # --------------------------------------------------------

    new_global_weights = fedavg(
        local_weights,
        local_sizes
    )

    global_model.set_weights(
        new_global_weights
    )

    # --------------------------------------------------------
    # GLOBAL EVALUATION
    # --------------------------------------------------------

    probabilities = global_model.predict(
        X_test,
        verbose=0
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    history.append({
        "Round": round_number,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    })

    print(
        f"  Accuracy: {accuracy:.4f}"
    )

    print(
        f"  F1-score: {f1:.4f}"
    )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(history)

history_df.to_csv(
    "fedavg_training_history.csv",
    index=False
)


# ============================================================
# FINAL PREDICTIONS
# ============================================================

final_probabilities = global_model.predict(
    X_test,
    verbose=0
)

final_predictions = np.argmax(
    final_probabilities,
    axis=1
)


# ============================================================
# FINAL METRICS
# ============================================================

final_accuracy = accuracy_score(
    y_test,
    final_predictions
)

final_precision = precision_score(
    y_test,
    final_predictions,
    average="weighted",
    zero_division=0
)

final_recall = recall_score(
    y_test,
    final_predictions,
    average="weighted",
    zero_division=0
)

final_f1 = f1_score(
    y_test,
    final_predictions,
    average="weighted",
    zero_division=0
)


# ============================================================
# DISPLAY FINAL RESULTS
# ============================================================

print("\n========================================")
print("FINAL FEDAVG RESULTS")
print("========================================")

print(
    f"Accuracy : {final_accuracy:.4f}"
)

print(
    f"Precision: {final_precision:.4f}"
)

print(
    f"Recall   : {final_recall:.4f}"
)

print(
    f"F1-score : {final_f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        final_predictions,
        target_names=[
            "Low",
            "Moderate",
            "High"
        ],
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    final_predictions
)

print("\nConfusion Matrix:")

print(cm)


# ============================================================
# CONVERGENCE GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    history_df["Round"],
    history_df["Accuracy"],
    marker="o",
    label="Accuracy"
)

plt.plot(
    history_df["Round"],
    history_df["F1_Score"],
    marker="s",
    label="F1-score"
)

plt.xlabel(
    "Communication Round"
)

plt.ylabel(
    "Performance"
)

plt.title(
    "FedAvg Convergence Across Communication Rounds"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "figures/fedavg_convergence.png",
    dpi=300
)

plt.close()


# ============================================================
# SAVE MODEL
# ============================================================

global_model.save(
    "fedavg_global_model.keras"
)


print("\n========================================")
print("FEDAVG TRAINING COMPLETED")
print("========================================")

print(
    "Training history saved as:"
)

print(
    "fedavg_training_history.csv"
)

print(
    "\nGlobal model saved as:"
)

print(
    "fedavg_global_model.keras"
)

print(
    "\nConvergence graph saved as:"
)

print(
    "figures/fedavg_convergence.png"
)

print("========================================")