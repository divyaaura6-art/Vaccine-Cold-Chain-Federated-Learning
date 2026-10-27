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


# -----------------------------
# Basic settings
# -----------------------------

phcs = ["PHC1", "PHC2", "PHC3", "PHC4", "PHC5"]

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

rounds = 20
local_epochs = 3
batch_size = 32

# FedProx parameter
mu = 0.01

np.random.seed(42)
tf.random.set_seed(42)

os.makedirs("figures", exist_ok=True)


# -----------------------------
# Create the MLP model
# -----------------------------

def create_model():

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

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# -----------------------------
# Load data from each PHC
# -----------------------------

phc_data = {}

print("\nLoading PHC datasets...")

for phc in phcs:

    df = pd.read_csv(f"data/{phc}.csv")

    phc_data[phc] = df

    print(phc, ":", len(df), "records")


# -----------------------------
# Split each PHC into
# training and testing data
# -----------------------------

local_train = {}
test_data = []

for phc in phcs:

    train_df, test_df = train_test_split(
        phc_data[phc],
        test_size=0.20,
        random_state=42,
        stratify=phc_data[phc][target]
    )

    local_train[phc] = train_df

    test_data.append(test_df)


# Use the test part from all PHCs
# as one common test dataset

global_test = pd.concat(
    test_data,
    ignore_index=True
)


# -----------------------------
# Scale the input features
# -----------------------------

all_train = pd.concat(
    list(local_train.values()),
    ignore_index=True
)

scaler = StandardScaler()

scaler.fit(
    all_train[features]
)


# -----------------------------
# Prepare test data
# -----------------------------

X_test = scaler.transform(
    global_test[features]
)

y_test = global_test[target].values


# -----------------------------
# Prepare local PHC data
# -----------------------------

local_X = {}
local_y = {}

for phc in phcs:

    local_X[phc] = scaler.transform(
        local_train[phc][features]
    )

    local_y[phc] = local_train[phc][target].values


# -----------------------------
# Start with the global model
# -----------------------------

global_model = create_model()


# -----------------------------
# FedAvg aggregation
# -----------------------------

def fedavg(local_weights, local_sizes):

    total_samples = sum(local_sizes)

    new_weights = []

    for layer in zip(*local_weights):

        average_layer = np.zeros_like(layer[0])

        for weights, size in zip(
            layer,
            local_sizes
        ):

            average_layer += (
                weights * size / total_samples
            )

        new_weights.append(
            average_layer
        )

    return new_weights


# -----------------------------
# Store results
# -----------------------------

results = []


# -----------------------------
# Federated training
# -----------------------------

print("\n================================")
print("STARTING FEDPROX TRAINING")
print("================================")


for current_round in range(1, rounds + 1):

    print(
        "\nCommunication Round",
        current_round,
        "/",
        rounds
    )

    local_weights = []
    local_sizes = []

    # -------------------------
    # Train each PHC separately
    # -------------------------

    for phc in phcs:

        print(" Training", phc)

        local_model = create_model()

        # Start from the current global model
        local_model.set_weights(
            global_model.get_weights()
        )

        # Save global weights
        global_weights = global_model.get_weights()

        # -------------------------
        # Local training
        # -------------------------

        local_model.fit(
            local_X[phc],
            local_y[phc],
            epochs=local_epochs,
            batch_size=batch_size,
            verbose=0
        )

        # -------------------------
        # FedProx correction
        # -------------------------

        new_weights = local_model.get_weights()

        corrected_weights = []

        for new_w, global_w in zip(
            new_weights,
            global_weights
        ):

            corrected_w = (
                new_w
                - mu * (new_w - global_w)
            )

            corrected_weights.append(
                corrected_w
            )

        # Store corrected local weights
        local_weights.append(
            corrected_weights
        )

        # Store number of local samples
        local_sizes.append(
            len(local_y[phc])
        )

    # -----------------------------
    # Server aggregation
    # -----------------------------

    global_weights = fedavg(
        local_weights,
        local_sizes
    )

    global_model.set_weights(
        global_weights
    )


    # -----------------------------
    # Evaluate global model
    # -----------------------------

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

    results.append({
        "Round": current_round,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    })

    print(
        " Accuracy:",
        round(accuracy, 4)
    )

    print(
        " F1-score:",
        round(f1, 4)
    )


# -----------------------------
# Save round-wise results
# -----------------------------

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    "fedprox_training_history.csv",
    index=False
)


# -----------------------------
# Final evaluation
# -----------------------------

final_probabilities = global_model.predict(
    X_test,
    verbose=0
)

final_predictions = np.argmax(
    final_probabilities,
    axis=1
)


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


# -----------------------------
# Print final results
# -----------------------------

print("\n================================")
print("FINAL FEDPROX RESULTS")
print("================================")

print(
    "Accuracy :",
    round(final_accuracy, 4)
)

print(
    "Precision:",
    round(final_precision, 4)
)

print(
    "Recall   :",
    round(final_recall, 4)
)

print(
    "F1-score :",
    round(final_f1, 4)
)


# -----------------------------
# Classification report
# -----------------------------

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


# -----------------------------
# Confusion matrix
# -----------------------------

cm = confusion_matrix(
    y_test,
    final_predictions
)

print("\nConfusion Matrix:")

print(cm)


# -----------------------------
# Convergence graph
# -----------------------------

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    results_df["Round"],
    results_df["Accuracy"],
    marker="o",
    label="Accuracy"
)

plt.plot(
    results_df["Round"],
    results_df["F1_Score"],
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
    "FedProx Convergence Across Communication Rounds"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "figures/fedprox_convergence.png",
    dpi=300
)

plt.close()


# -----------------------------
# Save the final model
# -----------------------------

global_model.save(
    "fedprox_global_model.keras"
)


print("\n================================")
print("FEDPROX TRAINING COMPLETED")
print("================================")

print(
    "Results saved as: fedprox_training_history.csv"
)

print(
    "Model saved as: fedprox_global_model.keras"
)

print(
    "Graph saved as: figures/fedprox_convergence.png"
)

print("================================")