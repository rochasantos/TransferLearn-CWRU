import os
from datasets import CWRU, UORED, Paderborn, Hust

def download_rawfile(dataset_name = 'all'):
    datasets = [CWRU(), UORED(), Paderborn(), Hust()]
    if dataset_name == 'all':
        for dataset in datasets:
            dataset.download()
    else:
        eval(f'{dataset_name}().download()')        