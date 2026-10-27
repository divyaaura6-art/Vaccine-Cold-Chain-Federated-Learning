
import pandas as pd
from sklearn.model_selection import train_test_split

PHCS = ["PHC1", "PHC2", "PHC3", "PHC4", "PHC5"]
TARGET = "Risk_Level"

for phc in PHCS:
    original = pd.read_csv(f"data/{phc}.csv")
    ablation = pd.read_csv(f"data_ablation/{phc}.csv")

    _, original_test = train_test_split(
        original,
        test_size=0.20,
        random_state=42,
        stratify=original[TARGET]
    )

    _, ablation_test = train_test_split(
        ablation,
        test_size=0.20,
        random_state=42,
        stratify=ablation[TARGET]
    )

    original_test = original_test.reset_index(drop=True)
    ablation_test = ablation_test.reset_index(drop=True)

    print(
        phc,
        "| Test labels identical:",
        original_test[TARGET].equals(ablation_test[TARGET]),
        "| Test rows:", len(original_test), len(ablation_test)
    )
