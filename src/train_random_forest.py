import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

df = pd.read_csv("data/processed/catalyst_dataset.csv")

X = df.drop(columns=["Activity"])
y = df["Activity"]

categorical = ["Material"]

numerical = [
    "Particle_Size_nm",
    "Bandgap_eV",
    "Surface_Area_m2_g",
    "Oxygen_Vacancy",
    "Light_Intensity_mW_cm2"
]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(), categorical),
    ("num", "passthrough", numerical)
])

model = RandomForestRegressor(n_estimators=200, random_state=42)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)

print("R2 =", round(r2_score(y_test, predictions), 3))
print("RMSE =", round(mean_squared_error(y_test, predictions) ** 0.5, 2))

joblib.dump(pipeline, "models/random_forest.pkl")

print("Model saved.")