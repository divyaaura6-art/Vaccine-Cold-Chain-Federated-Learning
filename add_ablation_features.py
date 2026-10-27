import os
import numpy as np
import pandas as pd

np.random.seed(42)

INPUT_DIR = "data"
OUTPUT_DIR = "data_ablation"

PHCS = ["PHC1", "PHC2", "PHC3", "PHC4", "PHC5"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def add_features(df):
    df = df.copy()

    # 1. Power outage duration
    power_outage = (
        0.15
        + 0.12 * df["Num_Excursions"]
        + np.random.normal(0, 0.20, len(df))
    )

    df["Power_Outage_Duration_hours"] = np.clip(
        power_outage, 0, 6
    )

    # 2. Door opening frequency
    door_openings = (
        2
        + 1.5 * df["Num_Excursions"]
        + np.random.normal(0, 1.5, len(df))
    )

    df["Door_Opening_Frequency"] = np.clip(
        np.round(door_openings), 0, 30
    ).astype(int)

    # 3. Cooling recovery time
    recovery_time = (
        0.25
        + 0.35 * df["Excursion_Duration_hours"]
        + 0.20 * df["Temperature_Deviation_C"]
        + np.random.normal(0, 0.15, len(df))
    )

    df["Cooling_Recovery_Time_hours"] = np.clip(
        recovery_time, 0, 8
    )

    ambient_temperature = np.random.normal(28.0, 3.5, len(df))

    df["Ambient_Temperature_C"] = np.clip(
        ambient_temperature, 20, 38
    )

    return df


all_data = []

for phc in PHCS:

    df = pd.read_csv(f"{INPUT_DIR}/{phc}.csv")

    augmented_df = add_features(df)

    augmented_df.to_csv(
        f"{OUTPUT_DIR}/{phc}.csv",
        index=False
    )

    all_data.append(augmented_df)


combined_df = pd.concat(
    all_data,
    ignore_index=True
)

combined_df.to_csv(
    f"{OUTPUT_DIR}/combined_dataset.csv",
    index=False
)


print("\n=== DATASET CREATED ===")
print("Shape:", combined_df.shape)

print("\nColumns:")
print(combined_df.columns.tolist())

print("\nNew feature statistics:")
print(
    combined_df[
        [
            "Power_Outage_Duration_hours",
            "Door_Opening_Frequency",
            "Cooling_Recovery_Time_hours",
            "Ambient_Temperature_C"
        ]
    ].describe().round(2)
)

print("\nRisk distribution:")
print(combined_df["Risk_Level"].value_counts())

print("\nNaN count:")
print(combined_df.isna().sum().sum())