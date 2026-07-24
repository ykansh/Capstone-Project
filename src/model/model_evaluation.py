import json
import pickle

import dagshub
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.logger import logging


# -------------------- MLflow Setup -------------------- #
mlflow.set_tracking_uri(
    "https://dagshub.com/iamanshchourasiya/Capstone-Project.mlflow"
)

dagshub.init(
    repo_owner="iamanshchourasiya",
    repo_name="Capstone-Project",
    mlflow=True,
)


# -------------------- Helper Functions -------------------- #

def load_model(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def load_data(file_path):
    return pd.read_csv(file_path)


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),
    }

    return metrics


def save_metrics(metrics):

    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)


def save_model_info(run_id, model_uri):

    info = {
        "run_id": run_id,
        "model_uri": model_uri,
    }

    with open("reports/experiment_info.json", "w") as f:
        json.dump(info, f, indent=4)


# -------------------- Main -------------------- #

def main():

    mlflow.set_experiment("my-dvc-pipeline")

    with mlflow.start_run() as run:

        clf = load_model("models/model.pkl")

        test = load_data("data/processed/test_bow.csv")

        X_test = test.iloc[:, :-1].values
        y_test = test.iloc[:, -1].values

        metrics = evaluate_model(clf, X_test, y_test)

        save_metrics(metrics)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log parameters
        if hasattr(clf, "get_params"):
            mlflow.log_params(clf.get_params())

        # Log sklearn model
        model_info = mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
        )

        print("=" * 60)
        print("Run ID :", run.info.run_id)
        print("Model URI :", model_info.model_uri)
        print("=" * 60)

        # Verify artifacts
        client = mlflow.tracking.MlflowClient()

        print("Artifacts:")
        for artifact in client.list_artifacts(run.info.run_id):
            print("-", artifact.path)

        # Save model info for registration stage
        save_model_info(
            run.info.run_id,
            model_info.model_uri,
        )

        # Log metrics file
        mlflow.log_artifact("reports/metrics.json")

        print("\nModel evaluation completed successfully!")


if __name__ == "__main__":
    main()