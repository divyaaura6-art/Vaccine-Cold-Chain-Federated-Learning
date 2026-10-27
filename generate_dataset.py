import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# SYNTHETIC VACCINE COLD-CHAIN DATASET
# ============================================================

# Reproducibility
np.random.seed(42)

# Number of observations per PHC
N = 1000

# Reference temperature range for this simulation
MIN_TEMP = 2.0
MAX_TEMP = 8.0

# Create output folders
os.makedirs("data", exist_ok=True)
os.makedirs("figures", exist_ok=True)


# ============================================================
# HELPER FUNCTION
# ============================================================

def calculate_temperature_deviation(temperature):
    """
    Calculates deviation from the acceptable 2-8°C range.
    Deviation is 0 when temperature is within the range.
    """
    if temperature < MIN_TEMP:
        return MIN_TEMP - temperature
    elif temperature > MAX_TEMP:
        return temperature - MAX_TEMP
    else:
        return 0.0


def generate_phc_data(phc_id):
    """
    Generate synthetic cold-chain data for one PHC.
    Each PHC has a different distribution to create
    heterogeneous / non-IID data.
    """

    # --------------------------------------------------------
    # PHC 1: Mostly stable temperature
    # --------------------------------------------------------
    if phc_id == "PHC1":

        temperature = np.random.normal(
            loc=5.0,
            scale=0.8,
            size=N
        )

        humidity = np.random.normal(
            loc=50,
            scale=6,
            size=N
        )

        storage_duration = np.random.normal(
            loc=12,
            scale=2,
            size=N
        )

        transport_duration = np.random.normal(
            loc=3,
            scale=0.8,
            size=N
        )

        num_excursions = np.random.poisson(
            lam=0.3,
            size=N
        )

        excursion_duration = np.random.exponential(
            scale=0.3,
            size=N
        )

    # --------------------------------------------------------
    # PHC 2: Frequent short temperature excursions
    # --------------------------------------------------------
    elif phc_id == "PHC2":

        temperature = np.random.normal(
            loc=5.5,
            scale=2.0,
            size=N
        )

        humidity = np.random.normal(
            loc=55,
            scale=7,
            size=N
        )

        storage_duration = np.random.normal(
            loc=11,
            scale=2.5,
            size=N
        )

        transport_duration = np.random.normal(
            loc=4,
            scale=1.0,
            size=N
        )

        num_excursions = np.random.poisson(
            lam=2.5,
            size=N
        )

        excursion_duration = np.random.exponential(
            scale=0.7,
            size=N
        )

    # --------------------------------------------------------
    # PHC 3: Higher humidity
    # --------------------------------------------------------
    elif phc_id == "PHC3":

        temperature = np.random.normal(
            loc=5.2,
            scale=1.0,
            size=N
        )

        humidity = np.random.normal(
            loc=78,
            scale=8,
            size=N
        )

        storage_duration = np.random.normal(
            loc=13,
            scale=2.5,
            size=N
        )

        transport_duration = np.random.normal(
            loc=3.5,
            scale=1.0,
            size=N
        )

        num_excursions = np.random.poisson(
            lam=0.7,
            size=N
        )

        excursion_duration = np.random.exponential(
            scale=0.5,
            size=N
        )

    # --------------------------------------------------------
    # PHC 4: Longer transportation
    # --------------------------------------------------------
    elif phc_id == "PHC4":

        temperature = np.random.normal(
            loc=5.8,
            scale=1.5,
            size=N
        )

        humidity = np.random.normal(
            loc=62,
            scale=8,
            size=N
        )

        storage_duration = np.random.normal(
            loc=15,
            scale=3,
            size=N
        )

        transport_duration = np.random.normal(
            loc=10,
            scale=2.5,
            size=N
        )

        num_excursions = np.random.poisson(
            lam=1.5,
            size=N
        )

        excursion_duration = np.random.exponential(
            scale=1.2,
            size=N
        )

    # --------------------------------------------------------
    # PHC 5: Extreme temperature variation
    # --------------------------------------------------------
    elif phc_id == "PHC5":

        temperature = np.random.normal(
            loc=6.0,
            scale=3.2,
            size=N
        )

        humidity = np.random.normal(
            loc=68,
            scale=10,
            size=N
        )

        storage_duration = np.random.normal(
            loc=10,
            scale=3,
            size=N
        )

        transport_duration = np.random.normal(
            loc=7,
            scale=2.5,
            size=N
        )

        num_excursions = np.random.poisson(
            lam=3.5,
            size=N
        )

        excursion_duration = np.random.exponential(
            scale=2.0,
            size=N

        )

    else:
        raise ValueError("Unknown PHC ID")

    # --------------------------------------------------------
    # Prevent impossible negative values
    # --------------------------------------------------------

    humidity = np.clip(humidity, 20, 100)

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

    # --------------------------------------------------------
    # Temperature deviation
    # --------------------------------------------------------

    temperature_deviation = np.array([
        calculate_temperature_deviation(t)
        for t in temperature
    ])

    # --------------------------------------------------------
    # Synthetic cumulative exposure
    # --------------------------------------------------------

    cumulative_exposure = (
        temperature_deviation * excursion_duration
        + 0.15 * num_excursions
        + 0.01 * np.maximum(humidity - 60, 0)
        + 0.02 * transport_duration
        + 0.01 * storage_duration
    )

    # --------------------------------------------------------
    # Synthetic risk score
    # --------------------------------------------------------

    risk_score = (
        10 * temperature_deviation
        + 4 * excursion_duration
        + 2 * num_excursions
        + 0.15 * np.maximum(humidity - 60, 0)
        + 0.5 * transport_duration
        + 0.2 * storage_duration
    )

    # Add small random noise so labels are not perfectly deterministic
    risk_score += np.random.normal(
        loc=0,
        scale=2,
        size=N
    )

    # --------------------------------------------------------
    # Convert risk score into categories
    # --------------------------------------------------------

    risk_level = np.where(
        risk_score < 15,
        0,
        np.where(
            risk_score < 30,
            1,
            2
        )
    )

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame({
        "PHC_ID": phc_id,
        "Temperature_C": np.round(temperature, 2),
        "Humidity_percent": np.round(humidity, 2),
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
        ),
        "Risk_Level": risk_level
    })

    return df


# ============================================================
# GENERATE DATA FOR ALL 5 PHCs
# ============================================================

phcs = [
    "PHC1",
    "PHC2",
    "PHC3",
    "PHC4",
    "PHC5"
]

all_data = []

for phc in phcs:

    df = generate_phc_data(phc)

    # Save individual PHC dataset
    file_path = f"data/{phc}.csv"
    df.to_csv(file_path, index=False)

    all_data.append(df)

    print(
        f"{phc}: {len(df)} records generated"
    )


# ============================================================
# COMBINE ALL PHCs
# ============================================================

combined_df = pd.concat(
    all_data,
    ignore_index=True
)

combined_df.to_csv(
    "data/combined_dataset.csv",
    index=False
)

print("\nTotal records:", len(combined_df))


# ============================================================
# DISPLAY DATASET SUMMARY
# ============================================================

print("\nDataset summary:")
print(
    combined_df.groupby("PHC_ID")[
        [
            "Temperature_C",
            "Humidity_percent",
            "Storage_Duration_hours",
            "Transport_Duration_hours",
            "Num_Excursions",
            "Excursion_Duration_hours",
            "Temperature_Deviation_C",
            "Cumulative_Exposure"
        ]
    ].mean().round(2)
)


# ============================================================
# RISK DISTRIBUTION
# ============================================================

risk_distribution = pd.crosstab(
    combined_df["PHC_ID"],
    combined_df["Risk_Level"],
    normalize="index"
) * 100

risk_distribution.columns = [
    "Low (%)",
    "Moderate (%)",
    "High (%)"
]

print("\nRisk distribution by PHC:")
print(risk_distribution.round(2))


# ============================================================
# FIGURE 1 — TEMPERATURE DISTRIBUTION
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
    "Temperature Distribution Across PHCs"
)

plt.xlabel("Healthcare Centre")
plt.ylabel("Temperature (°C)")

plt.tight_layout()

plt.savefig(
    "figures/temperature_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# FIGURE 2 — HUMIDITY DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=combined_df,
    x="PHC_ID",
    y="Humidity_percent"
)

plt.title(
    "Humidity Distribution Across PHCs"
)

plt.xlabel("Healthcare Centre")
plt.ylabel("Humidity (%)")

plt.tight_layout()

plt.savefig(
    "figures/humidity_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# FIGURE 3 — RISK DISTRIBUTION
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
    "Risk-Level Distribution Across PHCs"
)

plt.xlabel("Healthcare Centre")
plt.ylabel("Number of Observations")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "figures/risk_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# FIGURE 4 — PHC FEATURE COMPARISON
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
    "Average Cold-Chain Characteristics Across PHCs"
)

plt.xlabel("Healthcare Centre")
plt.ylabel("Average Value")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "figures/phc_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# FINISHED
# ============================================================

print("\n========================================")
print("DATASET GENERATION COMPLETED")
print("========================================")
print("Files saved in the data/ folder.")
print("Figures saved in the figures/ folder.")