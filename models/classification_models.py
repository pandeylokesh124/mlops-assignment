from sklearn.metrics import classification_report, roc_auc_score

from src.feature.prepare_data import PrepareData
from src.models.random_forest import RandomForestPipeline
from src.models.support_vector_classifier import SVClassification
from src.models.xgboost_model import XGBoostPipeline


class ClassificationModel:
    def __init__(self):
        print("Grab all the classification models")

    # Return all the models implemented
    def classification_model(self):
        random_forest = RandomForestPipeline(n_estimators=200, max_depth=5, min_samples_leaf=3)
        random_forest_model = random_forest._build_model()

        svc = SVClassification(C=1.0, kernel="rbf", gamma="scale")
        svc_model = svc._build_model()

        xgboost = XGBoostPipeline(use_sample_weights=True)
        xgboost_model = xgboost.build_model()

        return [
            ("Random Forest", random_forest_model),
            ("SVC", svc_model),
            ("XGBoost", xgboost_model)
        ]

    def run_models(self):
        reports = {}
        prep = PrepareData(data_path="../data/processed/heart_disease_clean.csv")
        X_train, X_test, y_train, y_test, preprocessor = prep.prepare_data()

        models = self.classification_model()

        for model_name, model in models:
            print(f"Evaluating {model_name}")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            test_auc = None
            if hasattr(model, "predict_proba"):
                try:
                    y_proba = model.predict_proba(X_test)
                    test_auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
                except Exception as e:
                    print(f"Could not compute ROC-AUC for {model_name}: {e}")

            report_dict = classification_report(y_test, y_pred, digits=4, zero_division=0, output_dict=True)
            params = model.get_params() if hasattr(model, "get_params") else {}

            reports[model_name] = [test_auc, report_dict, params, model]

        return reports


if __name__ == "__main__":
    pipeline = ClassificationModel()
    result = pipeline.run_models()
    print(result)

