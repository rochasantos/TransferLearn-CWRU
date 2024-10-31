import csv
import re

class AnnotationFileHandler:
    def __init__(self, filepath="data/annotation_file.csv"):
        self.filepath = filepath
        self.data = self._load_csv()

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

    def filter_data(self, **regex_filters):
        """
        Returns 
            Data filtered based on the regexes provided in the parameters.
            If no filter is provided, all data is returned.
        """
        if not regex_filters:
            return self.data  # Returns all data

        filtered_data = self.data

        for key, pattern in regex_filters.items():
            # Filters data using regex for each given key
            filtered_data = [row for row in filtered_data if re.search(pattern, row[key])]
        
        return filtered_data
