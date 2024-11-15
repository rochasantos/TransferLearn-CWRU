from scripts.download_rawfile import download_rawfile
from scripts.spectrograms import generate_spectrogram
from scripts.copy_spectrogram_to_folds import copy_spectrogram_to_folds
from src.data_processing.dataset_manager import DatasetManager
from utils import load_yaml


# DOWNLOAD RAW FILES
def download():
    for dataset in ["CWRU", "Hust", "UORED", "Paderborn"]:
        download_rawfile(dataset)

# SPECTROGRAMS
def create_spectrograms():

    # Sets the number of segments
    num_segments = 20

    # Load the configuration files
    spectrogram_config = load_yaml('config/spectrogram_config.yaml')
    filter_config = load_yaml('config/filters_config.yaml')
    
    # Instantiate the data manager
    data_manager = DatasetManager()
        
    for dataset_name in spectrogram_config.keys():
        print(f"Starting the creation of the {dataset_name} spectrograms.")
        filter = filter_config[dataset_name]
        metainfo = data_manager.filter_data(filter)

        signal_length = spectrogram_config[dataset_name]["Split"]["signal_length"]
        spectrogram_setup = spectrogram_config[dataset_name]["Spectrogram"]
        
        # Creation of spectrograms    
        generate_spectrogram(metainfo, spectrogram_setup, signal_length, num_segments) 


if __name__ == '__main__':
    download()
    create_spectrograms()
    copy_spectrogram_to_folds()
    