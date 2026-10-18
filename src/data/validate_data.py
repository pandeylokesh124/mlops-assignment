import pandas as pd

class ValidateData:
    data = None
    file_path = "../../data/raw/raw_data.csv"

    def __init__(self):
        print("Validating data ...")

    # Load dataset
    def load_dataset(self):
        try:
            self.data = pd.read_csv(self.file_path)
            return self.data
        except Exception as e:
            print("Failed to load data ", e)

    # Validate dataset
    def validate_data(self):
        self.data = self.load_dataset()
        print(self.data.head())
        print(self.data.describe())


if __name__ == "__main__":
    data = ValidateData()
    data.validate_data()

