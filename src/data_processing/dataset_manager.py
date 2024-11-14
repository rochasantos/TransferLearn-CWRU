import csv
import re
import yaml

class DatasetManager:
    def __init__(self, dataset_name=None, filter_config_path="config/filters_config.yaml", metainfo_path="data/annotation_file.csv"):
        self.metainfo_path = metainfo_path
        self.data = self._load_csv()
        self.filter_config = self._load_config(filter_config_path)
        self.dataset_name=dataset_name

    def _load_csv(self):
        data = []
        with open(self.metainfo_path, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
        return data

    def _load_config(self, file_path):
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)   

    def filter_data(self, filter_config=None):        
        
        if self.dataset_name:
            params = filter_config or {}
            filter_config = {"dataset_name": self.dataset_name, **params}

        if not filter_config:
            return self.data                  
        
        filtered_data = []
        for item in self.data:
            matches = all(
                item.get(key) in value if isinstance(value, list) else item.get(key) == value
                for key, value in filter_config.items()
            )
            if matches:
                filtered_data.append(item)
        
        return filtered_data


if __name__ == "__main__":
    metainfo = DatasetManager().filter_data({"filename": "97"})
    print(len(metainfo))
