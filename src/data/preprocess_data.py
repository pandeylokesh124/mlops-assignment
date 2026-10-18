from pathlib import Path
from validate_data import ValidateData

class PreprocessData:
    data = None
    output_path = "../../data/processed/heart_disease_clean.csv"

    def __init__(self):
        print("Preprocessing data ...")

    # Handle missing values
    def missing_values(self, data):
        # Fill missing 'ca' values with the mode using direct assignment
        ca_mode = data['ca'].mode()[0]
        data['ca'] = data['ca'].fillna(ca_mode)

        # Fill missing 'thal' values with the mode using direct assignment
        thal_mode = data['thal'].mode()[0]
        data['thal'] = data['thal'].fillna(thal_mode)
        return data

    # Save the clean file to a different location
    def save_data(self, output_path):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.data.to_csv(path, index=False)
        print(f"Preprocessed data saved to {path.resolve()}")

    def data_preprocess(self):
        dataset = ValidateData()
        self.data = dataset.load_dataset()
        self.data = self.missing_values(self.data) # Handle missing values
        print(self.data.head())
        print(self.data.describe())
        self.save_data(self.output_path)
        return self.data



if __name__ == "__main__":
    preprocess_data = PreprocessData()
    preprocess_data.data_preprocess()

