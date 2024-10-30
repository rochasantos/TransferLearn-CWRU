import os

from datasets.cwru import CWRU
from datasets.uored import UORED
from datasets.hust import Hust
from datasets.paderborn import Paderborn

def download_rawfile(dataset_name = 'all'):
    datasets = [CWRU(), UORED(), Paderborn()]
    if dataset_name == 'all':
        for dataset in datsets:
            dataset.download()
    else:
        try:
            eval(f'{dataset_name}().download()')        
        except:
            raise f"The dataset {dataset_name} does not exist."