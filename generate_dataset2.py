import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# SYNTHETIC VACCINE COLD-CHAIN DATASET - VERSION 2
# ============================================================

np.random.seed(42)

N = 1000

MIN_TEMP = 2.0
MAX_TEMP = 8.0

# Save V2 separately so the original dataset is not overwritten
os.makedirs("data_v2", exist_ok=True)
os.makedirs("figures_v2", exist_ok=True)


# ============================================================
# TEMPERATURE DEVIATION
# ============================================================

def calculate_temperature_deviation(temperature):

    if temperature < MIN_TEMP:
        return MIN_TEMP - temperature

    elif temperature > MAX_TEMP:
        return temperature - MAX_TEMP

    else:
        return 0.0


# ============================================================
# GENERATE DATA FOR ONE PHC
# ============================================================

def generate_phc_data(phc_id):

    # --------------------------------------------------------
    # PHC1 - mostly stable conditions
    # --------------------------------------------------------

    if phc_id == "PHC1":

        temperature = np.random.normal(
            5.0, 0.8, N
        )

        humidity = np.random.normal(
            50, 6, N
        )

        storage_duration = np.random.normal(
            12, 2, N
        )

        transport_duration = np.random.normal(
            3, 0.8, N
        )

        num_excursions = np.random.poisson(
            0.5, N
        )

        excursion_duration = np.random.exponential(
            0.4, N
        )


    # --------------------------------------------------------
    # PHC2 - more temperature excursions
    # --------------------------------------------------------

    elif phc_id == "PHC2":

        temperature = np.random.normal(
            5.3, 1.4, N
        )

        humidity = np.random.normal(
            55, 7, N
        )

        storage_duration = np.random.normal(
            12, 2.5, N
        )

        transport_duration = np.random.normal(
            4, 1.0, N
        )

        num_excursions = np.random.poisson(
            1.2, N
        )

        excursion_duration = np.random.exponential(
            0.5, N
        )


    # --------------------------------------------------------
    # PHC3 - higher humidity
    # --------------------------------------------------------

    elif phc_id == "PHC3":

        temperature = np.random.normal(
            5.5, 1.1, N
        )

        humidity = np.random.normal(
            70, 8, N
        )

        storage_duration = np.random.normal(
            13, 2.5, N
        )

        transport_duration = np.random.normal(
            3.5, 1.0, N
        )

        num_excursions = np.random.poisson(
            1.0, N
        )

        excursion_duration = np.random.exponential(
            0.6, N
        )


    # --------------------------------------------------------
    # PHC4 - longer transportation
    # --------------------------------------------------------

    elif phc_id == "PHC4":

        temperature = np.random.normal(
            5.7, 1.4, N
        )

        humidity = np.random.normal(
            62, 8, N
        )

        storage_duration = np.random.normal(
            14, 3, N
        )

        transport_duration = np.random.normal(
            7, 1.8, N
        )

        num_excursions = np.random.poisson(
            1.4, N
        )

        excursion_duration = np.random.exponential(
            0.8, N
        )


    # --------------------------------------------------------
    # PHC5 - highest temperature variation
    # --------------------------------------------------------

    elif phc_id == "PHC5":

        temperature = np.random.normal(
            5.9, 2.2, N
        )

        humidity = np.random.normal(
            66, 9, N
        )

        storage_duration = np.random.normal(
            11, 3, N
        )

        transport_duration = np.random.normal(
            6, 1.8, N
        )

        num_excursions = np.random.poisson(
            2.0, N
        )

        excursion_duration = np.random.exponential(
            1.0, N
        )


    else:

        raise ValueError("Unknown PHC ID")


    # ========================================================
    # CLEAN VALUES
    # ========================================================

    humidity = np.clip(
        humidity,
        20,
        100
    )

    storage_duration = np.clip(
        storage_duration,
        1,
        None
    )

    transport_duration = np.clip(
        transport_duration,
        0.5,
        None
    )

    num_excursions = np.clip(
        num_excursions,
        0,
        None
    )

    excursion_duration = np.clip(
        excursion_duration,
        0,
        24
    )


    # ========================================================
    # TEMPERATURE DEVIATION
    # ========================================================

    temperature_deviation = np.array([
        calculate_temperature_deviation(t)
        for t in temperature
    ])


    # ========================================================
    # CUMULATIVE EXPOSURE
    # ========================================================

    cumulative_exposure = (
        temperature_deviation * excursion_duration
        + 0.15 * num_excursions
        + 0.01 * np.maximum(humidity - 60, 0)
        + 0.02 * transport_duration
        + 0.01 * storage_duration
    )


    # ========================================================
    # SYNTHETIC RISK SCORE
    # ========================================================

    risk_score = (
        10 * temperature_deviation
        + 4 * excursion_duration
        + 2 * num_excursions
        + 0.15 * np.maximum(humidity - 60, 0)
        + 0.5 * transport_duration
        + 0.2 * storage_duration
    )

    # Add noise so the relationship is not perfectly fixed
    risk_score += np.random.normal(
        0,
        2,
        N
    )


    # ========================================================
    # DATAFRAME
    # ========================================================

    df = pd.DataFrame({

        "PHC_ID": phc_id,

        "Temperature_C": np.round(
            temperature,
            2
        ),

        "Humidity_percent": np.round(
            humidity,
            2
        ),

        "Storage_Duration_hours": np.round(
            storage_duration,
            2
        ),

        "Transport_Duration_hours": np.round(
            transport_duration,
            2
        ),

        "Num_Excursions": num_excursions.astype(int),

        "Excursion_Duration_hours": np.round(
            excursion_duration,
            2
        ),

        "Temperature_Deviation_C": np.round(
            temperature_deviation,
            2
        ),

        "Cumulative_Exposure": np.round(
            cumulative_exposure,
            2
        ),

        "Risk_Score": np.round(
            risk_score,
            2
        )
    })

    return df


# ============================================================
# GENERATE ALL PHCs
# ============================================================

phcs = [
    "PHC1",
    "PHC2",
    "PHC3",
    "PHC4",
    "PHC5"
]

all_data = []

print("\n========================================")
print("GENERATING DATASET V2")
print("========================================")

for phc in phcs:

    df = generate_phc_data(phc)

    all_data.append(df)

    print(
        phc,
        ":",
        len(df),
        "records generated"
    )


# ============================================================
# COMBINE DATA
# ============================================================

combined_df = pd.concat(
    all_data,
    ignore_index=True
)


# ============================================================
# CREATE RISK LEVELS
# ============================================================

# Use the overall risk-score distribution to create
# three broad risk groups.
#
# This keeps the total dataset reasonably balanced
# while still allowing each PHC to have a different
# risk distribution.

low_threshold = combined_df["Risk_Score"].quantile(
    0.333
)

high_threshold = combined_df["Risk_Score"].quantile(
    0.667
)


combined_df["Risk_Level"] = np.where(

    combined_df["Risk_Score"] < low_threshold,

    0,

    np.where(
        combined_df["Risk_Score"] < high_threshold,
        1,
        2
    )
)


# ============================================================
# SAVE INDIVIDUAL PHC FILES
# ============================================================

for phc in phcs:

    phc_df = combined_df[
        combined_df["PHC_ID"] == phc
    ]

    phc_df.to_csv(
        f"data_v2/{phc}.csv",
        index=False
    )


# ============================================================
# SAVE COMBINED DATASET
# ============================================================

combined_df.to_csv(
    "data_v2/combined_dataset.csv",
    index=False
)


# ============================================================
# DISPLAY RISK DISTRIBUTION
# ============================================================

risk_distribution = pd.crosstab(
    combined_df["PHC_ID"],
    combined_df["Risk_Level"]
)

risk_distribution.columns = [
    "Low",
    "Moderate",
    "High"
]

print("\nRisk distribution by PHC:")

print(
    risk_distribution
)


print("\nRisk distribution (%):")

print(
    risk_distribution
    .div(risk_distribution.sum(axis=1), axis=0)
    .mul(100)
    .round(2)
)


print("\nRisk score thresholds:")

print(
    "Low / Moderate threshold:",
    round(low_threshold, 2)
)

print(
    "Moderate / High threshold:",
    round(high_threshold, 2)
)


# ============================================================
# TEMPERATURE PLOT
# ============================================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=combined_df,
    x="PHC_ID",
    y="Temperature_C"
)

plt.axhspan(
    MIN_TEMP,
    MAX_TEMP,
    alpha=0.15
)

plt.title(
    "Temperature Distribution Across PHCs - Dataset V2"
)

plt.xlabel("Healthcare Centre")

plt.ylabel("Temperature (°C)")

plt.tight_layout()

plt.savefig(
    "figures_v2/temperature_distribution_v2.png",
    dpi=300
)

plt.close()


# ============================================================
# HUMIDITY PLOT
# ============================================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=combined_df,
    x="PHC_ID",
    y="Humidity_percent"
)

plt.title(
    "Humidity Distribution Across PHCs - Dataset V2"
)

plt.xlabel("Healthcare Centre")

plt.ylabel("Humidity (%)")

plt.tight_layout()

plt.savefig(
    "figures_v2/humidity_distribution_v2.png",
    dpi=300
)

plt.close()


# ============================================================
# RISK DISTRIBUTION PLOT
# ============================================================

risk_counts = pd.crosstab(
    combined_df["PHC_ID"],
    combined_df["Risk_Level"]
)

risk_counts.columns = [
    "Low",
    "Moderate",
    "High"
]

risk_counts.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title(
    "Risk-Level Distribution Across PHCs - Dataset V2"
)

plt.xlabel("Healthcare Centre")

plt.ylabel("Number of Observations")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "figures_v2/risk_distribution_v2.png",
    dpi=300
)

plt.close()


# ============================================================
# PHC FEATURE COMPARISON
# ============================================================

summary = combined_df.groupby("PHC_ID")[
    [
        "Temperature_C",
        "Humidity_percent",
        "Storage_Duration_hours",
        "Transport_Duration_hours",
        "Num_Excursions",
        "Excursion_Duration_hours",
        "Temperature_Deviation_C"
    ]
].mean()


summary.plot(
    kind="bar",
    figsize=(12, 7)
)

plt.title(
    "Average Cold-Chain Characteristics Across PHCs - Dataset V2"
)

plt.xlabel("Healthcare Centre")

plt.ylabel("Average Value")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "figures_v2/phc_comparison_v2.png",
    dpi=300
)

plt.close()


# ============================================================
# FINISHED
# ============================================================

print("\n========================================")
print("DATASET V2 GENERATION COMPLETED")
print("========================================")

print(
    "Dataset files saved in: data_v2/"
)

print(
    "Figures saved in: figures_v2/"
)

print("========================================")