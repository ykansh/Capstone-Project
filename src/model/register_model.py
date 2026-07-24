import json
import mlflow
import dagshub

from mlflow import MlflowClient
from src.logger import logging


# ------------------------------------------------------------
# MLflow + DagsHub
# ------------------------------------------------------------
mlflow.set_tracking_uri(
    "https://dagshub.com/iamanshchourasiya/Capstone-Project.mlflow"
)

dagshub.init(
    repo_owner="iamanshchourasiya",
    repo_name="Capstone-Project",
    mlflow=True,
)


# ------------------------------------------------------------
# Load experiment info
# ------------------------------------------------------------
def load_model_info(path):

    with open(path, "r") as f:
        return json.load(f)


# ------------------------------------------------------------
# Register Model
# ------------------------------------------------------------
def register_model():

    info = load_model_info("reports/experiment_info.json")

    model_uri = info["model_uri"]

    client = MlflowClient()

    model_name = "my_model"

    # Create registered model if it doesn't exist
    try:
        client.get_registered_model(model_name)
        print(f"Registered model '{model_name}' already exists.")
    except Exception:
        client.create_registered_model(model_name)
        print(f"Created registered model '{model_name}'.")

    # Register a new version
    version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )

    print("=" * 60)
    print("Registered Successfully")
    print("Model Name :", model_name)
    print("Version    :", version.version)
    print("=" * 60)

    # Move latest version to Staging
    client.transition_model_version_stage(
        name=model_name,
        version=version.version,
        stage="Staging",
        archive_existing_versions=False,
    )

    print(f"Model Version {version.version} moved to STAGING")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():

    try:
        register_model()

    except Exception as e:
        logging.error(e)
        print(e)


if __name__ == "__main__":
    main()