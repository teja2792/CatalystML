import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("data/processed/catalyst_dataset.csv")

X = df.drop(columns=["Activity"])
y = df["Activity"]

# Same split params used in both training scripts, so this reproduces
# the identical held-out test set for a fair comparison.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf_pipeline = joblib.load("models/random_forest.pkl")
xgb_pipeline = joblib.load("models/xgboost.pkl")

rf_predictions = rf_pipeline.predict(X_test)
xgb_predictions = xgb_pipeline.predict(X_test)

rf_r2 = r2_score(y_test, rf_predictions)
rf_rmse = mean_squared_error(y_test, rf_predictions) ** 0.5

xgb_r2 = r2_score(y_test, xgb_predictions)
xgb_rmse = mean_squared_error(y_test, xgb_predictions) ** 0.5

print("Model Comparison (same held-out test set)")
print("-------------------------------------------")
print(f"{'Model':<15}{'R2':>10}{'RMSE':>10}")
print(f"{'Random Forest':<15}{rf_r2:>10.3f}{rf_rmse:>10.2f}")
print(f"{'XGBoost':<15}{xgb_r2:>10.3f}{xgb_rmse:>10.2f}")

# SHAP explainability on the XGBoost model
preprocessor = xgb_pipeline.named_steps["preprocessor"]
model = xgb_pipeline.named_steps["model"]

X_transformed = preprocessor.transform(X)
feature_names = preprocessor.get_feature_names_out()

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_transformed)

shap.summary_plot(
    shap_values,
    X_transformed,
    feature_names=feature_names,
    show=False
)
plt.tight_layout()
plt.savefig("figures/shap_summary.png", dpi=150)
print("\nSHAP summary plot saved to figures/shap_summary.png")