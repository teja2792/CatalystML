import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

CATEGORICAL_FEATURES = ["Shape"]
NUMERICAL_FEATURES = [
    "Particle_Size_nm",
    "Bandgap_eV",
    "Surface_Area_m2_g",
    "Oxygen_Vacancy",
    "Light_Intensity_mW_cm2",
]
TARGET = "Activity"


def load_dataset(path="data/processed/catalyst_dataset.csv"):
    return pd.read_csv(path)


def build_preprocessor():
    return ColumnTransformer([
        ("cat", OneHotEncoder(), CATEGORICAL_FEATURES),
        ("num", "passthrough", NUMERICAL_FEATURES),
    ])


def split_features_target(df, target=TARGET):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def train_test_split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)