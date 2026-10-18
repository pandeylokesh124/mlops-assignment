from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

from src.feature.prepare_data import PrepareData


class XGBoostPipeline:
    def __init__(self, use_sample_weights: bool = True):
        self.use_sample_weights = use_sample_weights
        self.model = None

    def build_model(self) -> XGBClassifier:
        model = XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="mlogloss",  # Multi-class logloss
            objective="multi:softprob",  # Multi-class objective
            enable_categorical=True
        )
        return model

    def train_and_evaluate(self):
        # 1. Fetch preprocessed data
        data_prep = PrepareData()
        X_train, X_test, y_train, y_test, preprocessor = data_prep.prepare_data()

        # 2. Initialize Model
        self.model = self.build_model()

        # 3. Multi-Class Stratified Cross-Validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = cross_validate(
            self.model,
            X_train,
            y_train,
            cv=skf,
            scoring=["accuracy", "f1_macro", "roc_auc_ovr"],  # Fixed multi-class metrics
            n_jobs=-1
        )

        print("--- 5-Fold Cross-Validation Results ---")
        print(f"Mean CV Accuracy: {cv_results['test_accuracy'].mean():.4f} +/- {cv_results['test_accuracy'].std():.4f}")
        print(
            f"Mean CV F1-Macro:  {cv_results['test_f1_macro'].mean():.4f} +/- {cv_results['test_f1_macro'].std():.4f}")
        print(
            f"Mean CV ROC-AUC:   {cv_results['test_roc_auc_ovr'].mean():.4f} +/- {cv_results['test_roc_auc_ovr'].std():.4f}\n")

        # 4. Train on Full Training Set with Balanced Sample Weights
        if self.use_sample_weights:
            sample_weights = compute_sample_weight("balanced", y_train)
            self.model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            self.model.fit(X_train, y_train)

        # 5. Evaluate on Test Set
        y_pred = self.model.predict(X_test)

        # Predict probability distribution across ALL classes
        y_proba = self.model.predict_proba(X_test)

        print("--- Test Set Evaluation ---")
        # Multi-class ROC-AUC using One-vs-Rest (OvR)
        test_auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
        print(f"Test Multi-Class ROC-AUC (OvR): {test_auc:.4f}\n")

        print("Classification Report:")
        print(classification_report(y_test, y_pred, digits=4, zero_division=0))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        return self.model


if __name__ == "__main__":
    pipeline = XGBoostPipeline(use_sample_weights=True)
    trained_model = pipeline.train_and_evaluate()
    print(trained_model)

