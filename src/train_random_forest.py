import joblib

from preprocessing import load_dataset, split_features_target, train_test_split_data
from model import build_random_forest_pipeline
from evaluation import evaluate, print_metrics

df = load_dataset()
X, y = split_features_target(df)
X_train, X_test, y_train, y_test = train_test_split_data(X, y)

pipeline = build_random_forest_pipeline()
pipeline.fit(X_train, y_train)

metrics = evaluate(y_test, pipeline.predict(X_test))
print_metrics("Random Forest", metrics)

joblib.dump(pipeline, "models/random_forest.pkl")
print("Model saved.")