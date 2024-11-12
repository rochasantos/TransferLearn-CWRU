import csv
import re
import yaml

class DatasetManager:
    def __init__(self, config_path="config/filters_config.yaml", filepath="data/annotation_file.csv"):
        self.filepath = filepath
        self.data = self._load_csv()
        self.config = self._load_config(config_path)

    def _load_csv(self):
        """Loads the CSV file
        
        Returns 
            A list of dictionaries with the data.
        """
        data = []
        with open(self.filepath, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
        return data


    def _load_config(self, file_path):
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)

    def get_filtered_metadata(self):
        filtered_data = []

        for dataset, config in self.config.items():
            for item in self.data:
                matches = all(
                    item.get(key) in value and item.get("dataset_name")==dataset if isinstance(value, list) 
                    else item.get(key) == value and item.get("dataset_name")==dataset
                    for key, value in config.items()
                )
                if matches:
                    filtered_data.append(item)
        
        return filtered_data

    def filter_dataset(self, **regex_filters):
        """
        Returns 
            Data filtered based on the regexes provided in the parameters.
            If no filter is provided, all data is returned.
        """
        if not regex_filters:
            return self.data  # Returns all data

        filters = regex_filters or self._load_filter()
        for key, pattern in filters.items():
            # Filters data using regex for each given key
            filtered_data = [row for row in self.data if re.search(pattern, row[key])]
        
        return filtered_data


    def filter_data(self, filter_params=None):
        data = self._load_csv()
        filtered_data = []

        for dataset, config in self.config.items():
            for item in data:
                matches = all(
                    item.get(key) in value and item.get("dataset_name")==dataset if isinstance(value, list) 
                    else item.get(key) == value and item.get("dataset_name")==dataset
                    for key, value in config.items()
                )
                if matches:
                    filtered_data.append(item)
        
        return filtered_data
    
    def filter_by_dataset_name(self, dataset_name):
        """Filter the data by the given dataset name."""
        filtered_data = []
        if dataset_name in self.config:
            config = self.config[dataset_name]
            for item in self.data:
                matches = all(
                    item.get(key) in value if isinstance(value, list) else item.get(key) == value
                    for key, value in config.items()
                )
                if matches and item.get("dataset_name") == dataset_name:
                    filtered_data.append(item)
        return filtered_data