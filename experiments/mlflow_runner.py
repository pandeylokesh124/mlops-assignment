import mlflow
import mlflow.sklearn
import mlflow.xgboost
from models.classification_models import ClassificationModel

class MLFlowRunner:
    def __init__(self, tracking_uri: str = "http://127.0.0.1:5000"):
        self.uri = tracking_uri
        print("Initializing experimentation ...")

    def experiment(self):
        mlflow.set_tracking_uri(self.uri)
        mlflow.set_experiment("Heart Disease Detection")

        results = ClassificationModel().run_models()

        # Explicitly trust internal Scikit-Learn tree and calibration types for skops
        skops_trusted = [
            "sklearn.tree._tree.Tree",
            "sklearn.calibration._CalibratedClassifier",
            "sklearn.calibration._SigmoidCalibration",
        ]

        for model_name, (test_auc, report_dict, params, model) in results.items():
            with mlflow.start_run(run_name=f"{model_name} model"):
                # 1. Log Model Type Parameter
                mlflow.log_param("model_type", model_name)

                # 2. Log Hyperparameters
                clean_params = {
                    str(k): str(v)[:250]
                    for k, v in params.items()
                    if v is not None
                }
                mlflow.log_params(clean_params)

                # 3. Log Metrics
                metrics = {}
                if test_auc is not None:
                    metrics["test_roc_auc_ovr"] = float(test_auc)

                metrics["accuracy"] = float(report_dict["accuracy"])
                metrics["macro_precision"] = float(report_dict["macro avg"]["precision"])
                metrics["macro_recall"] = float(report_dict["macro avg"]["recall"])
                metrics["macro_f1"] = float(report_dict["macro avg"]["f1-score"])

                metrics["weighted_precision"] = float(report_dict["weighted avg"]["precision"])
                metrics["weighted_recall"] = float(report_dict["weighted avg"]["recall"])
                metrics["weighted_f1"] = float(report_dict["weighted avg"]["f1-score"])

                for class_id in ["0", "1", "2", "3", "4"]:
                    if class_id in report_dict:
                        metrics[f"class_{class_id}_precision"] = float(report_dict[class_id]["precision"])
                        metrics[f"class_{class_id}_recall"] = float(report_dict[class_id]["recall"])
                        metrics[f"class_{class_id}_f1"] = float(report_dict[class_id]["f1-score"])

                mlflow.log_metrics(metrics)

                # 4. Log Model Artifact & Register securely with skops
                registry_name = f"Heart_Disease_{model_name.replace(' ', '_')}"

                if "XGBoost" in model_name:
                    mlflow.xgboost.log_model(
                        xgb_model=model,
                        name="model",
                        registered_model_name=registry_name
                    )
                else:
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        name="model",
                        registered_model_name=registry_name,
                        skops_trusted_types=skops_trusted
                    )

                print(f"Successfully logged metrics, artifact, and registered: {registry_name}")


    def register_model(self):
        mlflow.set_tracking_uri(self.uri)

        run_id = input('Enter the run id: ').strip()
        artifact_path = input('Enter the artifact path (default is "model"): ').strip() or "model"
        model_name = input('Enter the model name for registry: ').strip()

        model_uri = f"runs:/{run_id}/{artifact_path}"

        try:
            result = mlflow.register_model(model_uri=model_uri, name=model_name)
            print(f"Successfully registered model '{model_name}' (Version {result.version})")
            return result
        except Exception as e:
            print(f"Failed to register model: {e}")


if __name__ == '__main__':
    MLFlowRunner().experiment()
    MLFlowRunner().register_model()
