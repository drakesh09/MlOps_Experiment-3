import json
import os
import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

with open("params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)

df = pd.read_csv("data/iris.csv")
target_column = "class" if "class" in df.columns else df.columns[-1]
X = df.drop(columns=[target_column])
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=params["train"]["test_size"],
    random_state=params["train"]["random_state"],
    stratify=y if params["train"]["stratify_split"] else None
)

os.makedirs("pipeline_models", exist_ok=True)

for version in ["v1", "v2"]:
    model_params = params["models"][version]

    model = RandomForestClassifier(
        n_estimators=model_params["n_estimators"],
        max_depth=model_params["max_depth"],
        random_state=model_params["random_state"]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, predictions), 5),
        "precision": round(precision_score(y_test, predictions, average="weighted", zero_division=0), 5),
        "recall": round(recall_score(y_test, predictions, average="weighted", zero_division=0), 5),
        "f1_score": round(f1_score(y_test, predictions, average="weighted", zero_division=0), 5)
    }

    model_path = f"pipeline_models/random_forest_{version}.pkl"
    metrics_path = f"metrics_{version}.json"

    joblib.dump(model, model_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"Model {version}: {model_path}")
    print(f"Metrics {version}: {metrics}")

print("Pipeline training completed successfully.")
