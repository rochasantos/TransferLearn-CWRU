import os
import yaml
from datasets import CWRU, UORED, Paderborn, Hust
from scipy import signal
from src.data_processing import DatasetManager
from src.spectrograms.generate_spectrogram import generate_spectrogram

def create_spectrograms(data_filter_path, preprocessing_pipeline, num_segments=None):

    data_info = DatasetManager(data_filter_path).filter_data()
    
    with open('config/spectrogram_config.yaml', 'r') as file:
        spect_info = yaml.safe_load(file)
    if not data_info:
        print("No data found to process.")
        return
    #print(data_info)
    for info in data_info:
        # Get infos
        dataset_name, basename, orig_sr = info['dataset_name'], info['filename'], int(info['sampling_rate'])
        
        # Load signal
        rawfilepath = os.path.join('data/raw', dataset_name.lower(), basename+'.mat') if info['dataset_name']!='Paderborn' else os.path.join('data/raw/paderborn', basename[12:16], basename+'.mat')
        signal, label = eval(f'{dataset_name}().load_signal_by_path("{rawfilepath}")')
        
        if signal is None or label is None:
            print(f"Failed to load signal or label for {basename}. Skipping.")
            continue

        # Get output path
        output_path = os.path.join('data/spectrograms', label, basename)

        # Set the config of spectrogram
        spect_config = spect_info.get(dataset_name)
        window_size = spect_config["Split"]["window_size"]
        spec_params = spect_config["Spectrogram"]

        # Preprocessing
        signal_processed = preprocessing_pipeline.process(signal, orig_sr)

        generate_spectrogram(signal_processed, output_path, window_size, spec_params, num_segments)

