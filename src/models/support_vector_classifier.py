from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, roc_auc_score

from src.feature.prepare_data import PrepareData


class SVClassification:
    def __init__(self, C: float = 1.0, kernel: str = "rbf", gamma: str = "scale"):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.pipeline = None

    def _build_model(self):
        base_svc = SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            class_weight="balanced",  # Adjusts margins for multi-class imbalance
            random_state=42
        )
        calibrated_model = CalibratedClassifierCV(
            estimator=base_svc,
            method="sigmoid",
            ensemble=False
        )
        return calibrated_model

    def train_and_evaluate(self):
        data_prep = PrepareData()
        # Retrieve the RAW splits and the unfitted preprocessor
        X_train, X_test, y_train, y_test, preprocessor = data_prep.prepare_data()

        # Build the full Scikit-Learn Pipeline
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', self._build_model())
        ])

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = cross_validate(
            self.pipeline,  # Pass the entire pipeline to cross_validate
            X_train,
            y_train,
            cv=skf,
            scoring=["accuracy", "f1_macro", "roc_auc_ovr"],
            n_jobs=-1
        )

        print("--- 5-Fold Cross-Validation Results (SVC) ---")
        print(f"Mean CV Accuracy: {cv_results['test_accuracy'].mean():.4f} +/- {cv_results['test_accuracy'].std():.4f}")
        print(f"Mean CV Macro F1: {cv_results['test_f1_macro'].mean():.4f} +/- {cv_results['test_f1_macro'].std():.4f}")
        print(f"Mean CV ROC-AUC:  {cv_results['test_roc_auc_ovr'].mean():.4f} +/- {cv_results['test_roc_auc_ovr'].std():.4f}\n")

        # Fit the entire pipeline on the raw training data
        self.pipeline.fit(X_train, y_train)

        # Predict using the fitted pipeline
        y_pred = self.pipeline.predict(X_test)
        y_proba = self.pipeline.predict_proba(X_test)

        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        test_auc = roc_auc_score(y_test, y_proba, multi_class="ovr")

        print("--- Test Set Evaluation ---")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print(f"Test Multi-Class ROC-AUC (OvR): {test_auc:.4f}\n")

        print("Classification Report:")
        print(classification_report(y_test, y_pred, digits=4, zero_division=0))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # Return the pipeline so your main script can log it to MLflow
        return self.pipeline


if __name__ == "__main__":
    svc_pipeline = SVClassification(C=1.0, kernel="rbf", gamma="scale")
    trained_pipeline = svc_pipeline.train_and_evaluate()
    print(trained_pipeline)
