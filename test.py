import mlflow
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/iamanshchourasiya/Capstone-Project.mlflow")
dagshub.init(repo_owner="iamanshchourasiya",
             repo_name="Capstone-Project",
             mlflow=True)

with mlflow.start_run() as run:
    print("Run:", run.info.run_id)

    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

    print(info)