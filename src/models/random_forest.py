from sklearn.pipeline import Pipeline
from src.feature.prepare_data import PrepareData
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

class RandomForestPipeline:
    def __init__(self, n_estimators: int = 200, max_depth: int = 5, min_samples_leaf: int = 3):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.pipeline = None

    def _build_model(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight="balanced_subsample",  # Recalculates weights per bootstrap sample
            random_state=42,
            n_jobs=-1
        )

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
            self.pipeline, # Pass the entire pipeline to cross_validate
            X_train,
            y_train,
            cv=skf,
            scoring=["accuracy", "f1_macro"],
            n_jobs=-1
        )

        print("--- 5-Fold Cross-Validation Results ---")
        print(f"Mean CV Accuracy: {cv_results['test_accuracy'].mean():.4f} +/- {cv_results['test_accuracy'].std():.4f}")
        print(f"Mean CV Macro F1: {cv_results['test_f1_macro'].mean():.4f} +/- {cv_results['test_f1_macro'].std():.4f}\n")

        # Fit the entire pipeline on the raw training data
        self.pipeline.fit(X_train, y_train)

        # Predict using the fitted pipeline
        y_pred = self.pipeline.predict(X_test)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)

        print("--- Test Set Evaluation ---")
        print(f"Balanced Accuracy: {balanced_acc:.4f}\n")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, digits=4, zero_division=0))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # Return the pipeline so your main script can log it to MLflow
        return self.pipeline


if __name__ == "__main__":
    rf_pipeline = RandomForestPipeline(n_estimators=200, max_depth=5, min_samples_leaf=3)
    trained_pipeline = rf_pipeline.train_and_evaluate()
    print(trained_pipeline)
