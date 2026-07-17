"""
Generate a synthetic photocatalyst dataset for CatalystML.

This dataset is intended for demonstrating machine learning workflows.
It is inspired by experimentally relevant photocatalyst descriptors but
does not represent actual experimental measurements.
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

n = 1000

materials = ["Cu2O", "TiO2", "CeO2", "Fe2O3"]

df = pd.DataFrame({
    "Material": np.random.choice(materials, n),
    "Particle_Size_nm": np.random.uniform(20, 120, n),
    "Bandgap_eV": np.random.uniform(1.8, 3.2, n),
    "Surface_Area_m2_g": np.random.uniform(10, 120, n),
    "Oxygen_Vacancy": np.random.uniform(0, 1, n),
    "Light_Intensity_mW_cm2": np.random.uniform(50, 150, n)
})

# Chemistry-inspired activity equation
activity = (
    80
    + 0.30 * df["Surface_Area_m2_g"]
    - 0.35 * df["Particle_Size_nm"]
    + 18 * df["Oxygen_Vacancy"]
    + 0.10 * df["Light_Intensity_mW_cm2"]
    - 12 * (df["Bandgap_eV"] - 2.2) ** 2
    + np.random.normal(0, 5, n)
)

df["Activity"] = activity.round(2)

os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/catalyst_dataset.csv", index=False)

print(df.head())
print("\nDataset saved to data/processed/catalyst_dataset.csv")