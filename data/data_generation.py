# data_generation.py
import pandas as pd
import numpy as np
from datetime import datetime
import os

def generate_data(n=1000, output_csv="data/yarra_assets.csv"):
    # Ensure output folder exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    folder = os.path.dirname(output_csv)
    current_year = datetime.now().year
    np.random.seed(42)
    if folder:
        os.makedirs(folder, exist_ok=True)

    materials = ["Steel", "Plastic", "Copper", "Cast Iron", "Concrete"]
    statuses = ["Active", "Inactive", "Decommissioned"]
    asset_types = ["Pipe", "Pump", "Valve", "Sewer"]
    network_types = ["Water", "Sewer", "Recycled Water"]
    zones = ["North", "East", "South", "West", "Central"]

    df = pd.DataFrame({
        "asset_id": [f"A-{i:05d}" for i in range(n)],
        "asset_name": [f"Asset_{i}" for i in range(n)],
        "asset_type": np.random.choice(asset_types, n),
        "material": np.random.choice(materials, n),
        "diameter": np.random.uniform(50, 500, n).round(1),  # mm
        "installation_year": np.random.randint(1950, current_year, n),
        "status": np.random.choice(statuses, n, p=[0.8, 0.15, 0.05]),
        "network_type": np.random.choice(network_types, n),
        "length": np.random.uniform(5, 500, n).round(1),
        "depth": np.random.uniform(0.5, 5.0, n).round(2),
        "zone": np.random.choice(zones, n),
        "pressure_rating": np.random.choice([100, 150, 200, 250, 300], n),
        "location": np.random.choice(
            ["Melbourne CBD", "Suburb A", "Suburb B", "Suburb C", "Rural Area"], n
        ),
        "geometry": [
            f"{np.random.uniform(-37.90, -37.70):.6f},{np.random.uniform(144.80, 145.20):.6f}"
            for _ in range(n)
        ]
    })

    # -------------------------------
    # Remaining Useful Life Simulation
    # -------------------------------
    def simulate_rul(row):
        expected_lifespan = 60  # base lifespan
        age = current_year - row["installation_year"]
        remaining = expected_lifespan - age

        # Adjust for risky materials and high pressure
        if row["material"] in ["Steel", "Cast Iron"]:
            remaining -= 10
        if row["pressure_rating"] > 200:
            remaining -= 5

        remaining += np.random.normal(0, 3)  # noise
        return max(0, remaining)

    df["Remaining_Years"] = df.apply(simulate_rul, axis=1)

    # -------------------------------
    # Failure Simulation
    # -------------------------------
    def simulate_failure(row):
        risk = 0
        age = current_year - row["installation_year"]

        if row["material"] in ["Steel", "Cast Iron"]:
            risk += 0.3
        if age > 40:
            risk += 0.4
        if row["pressure_rating"] > 200:
            risk += 0.2
        if row["status"] == "Inactive":
            risk += 0.1

        # Add random noise
        risk += np.random.normal(0, 0.05)
        return 1 if risk > 0.6 else 0

    df["Failure"] = df.apply(simulate_failure, axis=1)

    # Save dataset
    df.to_csv(output_csv, index=False)
    print(f"Generated dataset with {n} rows at {output_csv}")
    print(df.head())

if __name__ == "__main__":
    generate_data()