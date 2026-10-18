import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class PrepareData:
    def __init__(self, data_path="../../data/processed/heart_disease_clean.csv"):
        self.data = pd.read_csv(data_path)

    def prepare_data(self):
        X = self.data.drop(columns=['num'])  # Feature array
        y = self.data['num']  # Target array

        num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        cat_cols = ['cp', 'restecg', 'slope', 'thal']
        passthrough_cols = ['sex', 'fbs', 'exang', 'ca']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_cols),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols),
                ('pass', 'passthrough', passthrough_cols)
            ]
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    prep = PrepareData()
    X_train, X_test, y_train, y_test, preprocessor = prep.prepare_data()
    print(f"X_train shape (raw features): {X_train.shape}")
    print(f"X_test shape (raw features): {X_test.shape}")
