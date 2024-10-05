from datasets.cwru import CWRU
from datasets.paderborn import Paderborn
from datasets.hust import Hust
from datasets.uored import UORED
from src.feature_engineering.create_spectrogram import generate_spectrogram

class DatasetModelFactory:
    @staticmethod
    def create_dataset(dataset_name):
        if dataset_name == 'paderborn':
            return Paderborn()
        elif dataset_name == "cwru":
            return CWRU()
        elif dataset_name == "hust":
            return Hust()
        elif dataset_name == "uored":
            return UORED()
        else:
            raise ValueError("Unknown dataset.")

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
def test_load_signal_with_acquisition_maxsize_none(dataset_name):
    dataset = fct.create_dataset(dataset_name)    
    dataset.load_signal()
    print('** test_load_signal_with_acquisition_maxsize_none')
    print(f'- data.shape: {dataset.data.shape}, label.shape: {dataset.label.shape}\n')

def test_load_signal_with_acquisition_maxsize_differ_none(dataset_name):
    dataset = fct.create_dataset(dataset_name)    
    dataset.load_signal(acquisition_maxsize=64000)
    print('** test_load_signal_with_acquisition_maxsize_64000')
    print(f'- data.shape: {dataset.data.shape}, label.shape: {dataset.label.shape}\n')

def test_load_signal_with_filter_file(dataset_name):
    if dataset_name == 'paderborn':
        regex = r'.*N09_M07_F10_K001_[12]\.mat$'
    elif dataset_name == 'cwru':
        regex = r'B\.007\.DE_[01]&12000.mat'
    dataset = fct.create_dataset(dataset_name) 
    dataset.load_signal(acquisition_maxsize=64000, regex_filter=regex)
    print('** test_load_signal_with_acquisition_maxsize_64000_x_3_with_filter')
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
    dataset_name = 'paderborn'
    # download
    test_download(dataset_name)

    # load signal
    test_load_signal_with_acquisition_maxsize_none(dataset_name)
    test_load_signal_with_acquisition_maxsize_differ_none(dataset_name)
    test_load_signal_with_filter_file(dataset_name)

    # generate spectrograms
    test_create_spectrograms(dataset_name)


if __name__ == '__main__':
    test()
