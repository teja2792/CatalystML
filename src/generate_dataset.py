"""
Generate a synthetic Cu2O photocatalyst dataset for CatalystML.

Scope: restricted to Cu2O only. The particle-size dependence of
photocatalytic activity is calibrated against real, published experimental
data for Cu2O nanospheres and nanocubes:

    Tirumala, R. T. A. et al. "Structure-Property-Performance Relationships
    of Cuprous Oxide Nanostructures for Dielectric Mie Resonance-Enhanced
    Photocatalysis." ACS Catalysis 2022, 12, 7975-7985.
    https://doi.org/10.1021/acscatal.2c00977

Measured first-order rate constants for MB degradation (Table S1, SI):
    Spheres,  42 nm : k = 0.034 hr^-1
    Spheres, 145 nm : k = 0.327 hr^-1
    Cubes,    92 nm : k = 0.113 hr^-1
    Cubes,   286 nm : k = 0.141 hr^-1
    Cubes,   456 nm : k = 0.045 hr^-1

These five points show a volcano-type (non-monotonic) relationship between
particle size and rate, driven by dielectric Mie resonance rather than
surface area (145-nm spheres outperform 42-nm spheres by ~9.6x despite
having only ~0.29x the surface area). A Gaussian resonance curve is fit
per shape to these real measurements and used as the dominant term below.

No equivalent measured rate data exists for CeO2, Fe2O3, or TiO2 in this
paper (only simulated optical constants, SI Tables S3-S5) -- so this
dataset excludes them rather than inventing unsupported relationships.

This remains a SYNTHETIC dataset: the resonance curve is real-data-
calibrated, but the oxygen-vacancy and bandgap-shift modulation terms below
are illustrative assumptions, not literature-derived values. Noise is
added on top of the fitted curve to build a workable dataset for ML
demonstration purposes.
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

np.random.seed(42)

n = 1000

# ---------------------------------------------------------------------------
# Real experimental anchor points (Table S1, Tirumala et al., ACS Catal. 2022)
# ---------------------------------------------------------------------------
SPHERE_SIZES = np.array([42.0, 145.0])
SPHERE_RATES = np.array([0.034, 0.327])

CUBE_SIZES = np.array([92.0, 286.0, 456.0])
CUBE_RATES = np.array([0.113, 0.141, 0.045])


def gaussian(size, peak_size, width, amplitude):
    return amplitude * np.exp(-((size - peak_size) ** 2) / (2 * width ** 2))


# Spheres: only 2 measured points. Peak fixed at 145 nm (the largest
# measured sphere size, reported as exhibiting the Mie resonance); width
# solved from the single remaining data point (42 nm).
sphere_peak = 145.0
sphere_amplitude = SPHERE_RATES[1]
ratio = SPHERE_RATES[0] / sphere_amplitude
sphere_width = np.sqrt(-((SPHERE_SIZES[0] - sphere_peak) ** 2) / (2 * np.log(ratio)))

# Cubes: 3 measured points -- enough to fit all three Gaussian parameters.
cube_params, _ = curve_fit(
    gaussian, CUBE_SIZES, CUBE_RATES,
    p0=[300.0, 150.0, 0.15],
    maxfev=10000,
)
cube_peak, cube_width, cube_amplitude = cube_params

print(f"Fitted sphere resonance: peak={sphere_peak:.1f} nm, width={sphere_width:.1f} nm, amplitude={sphere_amplitude:.3f}")
print(f"Fitted cube resonance:   peak={cube_peak:.1f} nm, width={cube_width:.1f} nm, amplitude={cube_amplitude:.3f}")

# ---------------------------------------------------------------------------
# Generate synthetic samples
# ---------------------------------------------------------------------------
CU2O_DENSITY_G_CM3 = 6.0  # bulk Cu2O density, used for SSA below

shapes = np.random.choice(["Sphere", "Cube"], n)
particle_size = np.random.uniform(30, 500, n)

# Specific surface area derived geometrically from particle size (not
# sampled independently). SSA = 6000 / (density * size). This reproduces
# the paper's own stated ratio: (6000/(6*145)) / (6000/(6*42)) = 42/145
# = 0.29, matching their reported "0.29x the surface area" for 145 vs 42 nm.
surface_area = 6000.0 / (CU2O_DENSITY_G_CM3 * particle_size)
surface_area *= np.random.normal(1.0, 0.05, n)  # measurement noise

oxygen_vacancy = np.random.uniform(0, 1, n)
light_intensity = np.random.uniform(50, 150, n)
bandgap = np.random.normal(2.10, 0.04, n)  # Cu2O bandgap ~2.1 eV +/- batch variation

resonance = np.where(
    shapes == "Sphere",
    gaussian(particle_size, sphere_peak, sphere_width, sphere_amplitude),
    gaussian(particle_size, cube_peak, cube_width, cube_amplitude),
)

# Secondary modulation terms below are illustrative, not literature-derived.
light_term = light_intensity / 100.0
vacancy_term = 1 + 0.20 * oxygen_vacancy
bandgap_term = 1 - 0.15 * ((bandgap - 2.10) / 0.10) ** 2
noise = np.random.normal(1.0, 0.10, n)

rate_constant = resonance * light_term * vacancy_term * bandgap_term * noise
rate_constant = np.clip(rate_constant, 0.001, None)

df = pd.DataFrame({
    "Shape": shapes,
    "Particle_Size_nm": particle_size.round(1),
    "Surface_Area_m2_g": surface_area.round(2),
    "Bandgap_eV": bandgap.round(3),
    "Oxygen_Vacancy": oxygen_vacancy.round(3),
    "Light_Intensity_mW_cm2": light_intensity.round(1),
    "Activity": rate_constant.round(4),
})

os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/catalyst_dataset.csv", index=False)

print(df.head())
print("\nDataset saved to data/processed/catalyst_dataset.csv")