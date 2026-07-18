import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

CU2O_DENSITY_G_CM3 = 6.0

# Reference operating conditions held constant while sweeping particle size,
# so the recommendation isolates the effect of the variable actually
# controlled at synthesis time (size/shape) rather than conflating it with
# arbitrary experimental conditions.
REFERENCE_OXYGEN_VACANCY = 0.5
REFERENCE_LIGHT_INTENSITY = 100.0
REFERENCE_BANDGAP = 2.10

rf_pipeline = joblib.load("models/random_forest.pkl")
xgb_pipeline = joblib.load("models/xgboost.pkl")

sizes = np.arange(30, 501, 5)
shapes = ["Sphere", "Cube"]

rows = []
for shape in shapes:
    for size in sizes:
        surface_area = 6000.0 / (CU2O_DENSITY_G_CM3 * size)
        rows.append({
            "Shape": shape,
            "Particle_Size_nm": size,
            "Surface_Area_m2_g": surface_area,
            "Bandgap_eV": REFERENCE_BANDGAP,
            "Oxygen_Vacancy": REFERENCE_OXYGEN_VACANCY,
            "Light_Intensity_mW_cm2": REFERENCE_LIGHT_INTENSITY,
        })

candidates = pd.DataFrame(rows)

candidates["RF_Predicted_Activity"] = rf_pipeline.predict(candidates)
candidates["XGB_Predicted_Activity"] = xgb_pipeline.predict(candidates)
candidates["Mean_Predicted_Activity"] = candidates[
    ["RF_Predicted_Activity", "XGB_Predicted_Activity"]
].mean(axis=1)

os.makedirs("results", exist_ok=True)
candidates.to_csv("results/catalyst_recommendations.csv", index=False)

print("Top 5 recommended candidates per shape (by mean predicted activity):\n")
for shape in shapes:
    top5 = (
        candidates[candidates["Shape"] == shape]
        .sort_values("Mean_Predicted_Activity", ascending=False)
        .head(5)
    )
    print(f"--- {shape} ---")
    print(top5[["Particle_Size_nm", "RF_Predicted_Activity", "XGB_Predicted_Activity", "Mean_Predicted_Activity"]]
          .to_string(index=False))
    best_size = top5.iloc[0]["Particle_Size_nm"]
    print(f"Best predicted size: {best_size:.0f} nm\n")

os.makedirs("figures", exist_ok=True)
plt.figure(figsize=(8, 5))
for shape in shapes:
    subset = candidates[candidates["Shape"] == shape]
    plt.plot(subset["Particle_Size_nm"], subset["Mean_Predicted_Activity"], label=shape)

plt.xlabel("Particle Size (nm)")
plt.ylabel("Predicted Activity (rate constant, hr$^{-1}$)")
plt.title("Model-Predicted Catalyst Activity vs Particle Size")
plt.legend()
plt.tight_layout()
plt.savefig("figures/catalyst_recommendation_curve.png", dpi=150)

print("Recommendation curve saved to figures/catalyst_recommendation_curve.png")
print("Full candidate grid saved to results/catalyst_recommendations.csv")