import joblib
import shap
import matplotlib.pyplot as plt

from preprocessing import load_dataset, split_features_target, train_test_split_data
from evaluation import evaluate

df = load_dataset()
X, y = split_features_target(df)
X_train, X_test, y_train, y_test = train_test_split_data(X, y)

rf_pipeline = joblib.load("models/random_forest.pkl")
xgb_pipeline = joblib.load("models/xgboost.pkl")

rf_metrics = evaluate(y_test, rf_pipeline.predict(X_test))
xgb_metrics = evaluate(y_test, xgb_pipeline.predict(X_test))

print("Model Comparison (same held-out test set)")
print("-------------------------------------------")
print(f"{'Model':<15}{'R2':>10}{'RMSE':>10}")
print(f"{'Random Forest':<15}{rf_metrics['r2']:>10}{rf_metrics['rmse']:>10}")
print(f"{'XGBoost':<15}{xgb_metrics['r2']:>10}{xgb_metrics['rmse']:>10}")

preprocessor = xgb_pipeline.named_steps["preprocessor"]
model = xgb_pipeline.named_steps["model"]

X_transformed = preprocessor.transform(X)
feature_names = preprocessor.get_feature_names_out()

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_transformed)

shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig("figures/shap_summary.png", dpi=150)
print("\nSHAP summary plot saved to figures/shap_summary.png")