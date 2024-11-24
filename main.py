from scripts.download_rawfile import download_rawfile
from scripts.spectrograms import generate_spectrogram
from scripts.copy_spectrogram_to_folds import copy_spectrogram_to_folds
from src.data_processing.dataset_manager import DatasetManager
from utils import load_yaml
from utils.dual_output import DualOutput  # Import the class from dual_output.py
from experimenter_vitclassifier_kfold import experimenter_vitclassifier_kfold
from run_pretrain import experimenter
import sys
from datetime import datetime
import os

# Create logs directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# Generate a timestamped log file name
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
log_filename = f"results/experiment_log_{timestamp}.txt"

# Redirect stdout
sys.stdout = DualOutput(log_filename)


# DOWNLOAD RAW FILES
def download():
    for dataset in ["CWRU", "UORED"]:
        download_rawfile(dataset)

# SPECTROGRAMS
def create_spectrograms():

    # Sets the number of segments
    num_segments = 20

    # Load the configuration files
    spectrogram_config = load_yaml('config/spectrogram_config.yaml')
    filter_config = load_yaml('config/filters_config.yaml')
    
    # Instantiate the data manager
        
    for dataset_name in spectrogram_config.keys():
        print(f"Starting the creation of the {dataset_name} spectrograms.")
        filter = filter_config[dataset_name]
        data_manager = DatasetManager(dataset_name)
        metainfo = data_manager.filter_data(filter)
        signal_length = spectrogram_config[dataset_name]["Split"]["signal_length"]
        spectrogram_setup = spectrogram_config[dataset_name]["Spectrogram"]
        
        # Creation of spectrograms    
        generate_spectrogram(metainfo, spectrogram_setup, signal_length, num_segments) 

# EXPERIMENTERS
def run_experimenter():
    #model = ResNet18() 
    experimenter_vitclassifier_kfold() #pre train and test


if __name__ == '__main__':
    #download()
    #create_spectrograms()
    run_experimenter()
    
    
    # Close the log file
    sys.stdout.close()
    sys.stdout = sys.__stdout__  # Reset stdout to the original