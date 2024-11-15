import csv
import yaml

def load_csv(path):
    data = []
    with open(path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

def load_yaml(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file) 