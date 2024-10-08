import numpy as np
import os
from datasets.cwru import CWRU
from datasets.paderborn import Paderborn
from datasets.hust import Hust
from datasets.uored import UORED
from src.feature_engineering.create_spectrogram import generate_spectrogram

class DatasetModelFactory:
    @staticmethod
    def create_dataset(dataset_name):
        try:
            dataset = eval(f'{dataset_name}(debug=True)')
        except:
            raise ValueError("Unknown dataset.")
        
        dataset._rawfilesdir = f'data/raw_test/{dataset}'
        dataset._spectdir = f'data/processed_test/{dataset}'

        if not os.path.exists(dataset.spectdir):
            os.makedirs(dataset.spectdir, exist_ok=True)

        return dataset

fct = DatasetModelFactory()

"""
Download and Extract files
"""
def test_download(dataset_name):
    dataset = fct.create_dataset(dataset_name)
    dataset.download()

"""
Load Signal
"""
def test_load_signal(dataset_name):
    dataset = fct.create_dataset(dataset_name)    
    dataset.load_signal()
    print('** testing load_signal function.')
    print(f'- data.shape: {dataset.data.shape}, label.shape: {dataset.label.shape}')
    print(f'- labels: {np.unique(dataset.label)}\n')

def test_load_signal_with_filter_file(dataset_name):
    if dataset_name == 'paderborn':
        regex = r'.*N09_M07_F10_K001_[12]\.mat$'
    elif dataset_name == 'cwru':
        regex = r'B\.007\.DE_[01]&12000.mat'
    else:
        regex = f'.*\.mat'
    dataset = fct.create_dataset(dataset_name) 
    dataset.load_signal(regex_filter=regex)
    print('** test_load_signal_with_filter')
    print(f'- data.shape: {dataset.data.shape}, label.shape: {dataset.label.shape}\n')

"""
Create Spectrograms
"""
def test_create_spectrograms(dataset_name):
    dataset = fct.create_dataset(dataset_name)
    dataset.load_signal()
    data, label, fs, specdir = dataset.data, dataset.label, dataset.sample_rate, dataset.spectdir
    generate_spectrogram(data, label, fs, specdir)


def test():
    dataset_name = 'UORED'
    # download
    test_download(dataset_name)
    
    # load signal
    test_load_signal(dataset_name)
    test_load_signal_with_filter_file(dataset_name)

    # generate spectrograms
    test_create_spectrograms(dataset_name)

if __name__ == '__main__':
    test()
