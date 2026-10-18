import os
from ucimlrepo import fetch_ucirepo
from urllib.request import urlretrieve

class DownloadData:
    target_dir = "../../data/raw"
    file_name = "raw_data.csv"

    def __init__(self):
        print("Downloading UCI Diabetes Data ...")

    # Download dataset
    def download_data(self):
        try:
            # fetch dataset
            heart_disease = fetch_ucirepo(id=45)
            url = heart_disease.metadata.data_url
            destination_path = os.path.join(self.target_dir, self.file_name)
            os.makedirs(self.target_dir, exist_ok=True)
            urlretrieve(url, destination_path)
            print(f"File successfully saved to: {destination_path}")
        except Exception as e:
            print(e)

if __name__ == "__main__":
    data = DownloadData()
    data.download_data()

