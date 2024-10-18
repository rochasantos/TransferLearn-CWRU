import os
from src.feature_engineering.spectrogram.generate_spectrogram import generate_spectrogram
from src.data_processing.annotation_file import AnnotationFileHandler
from src.preprocessing import PreprocessingPipeline, ResamplingStrategy, ZeroMeanStrategy, OutlierRemovalStrategy
from src.feature_engineering.spectrogram.base import SpectrogramCreator


def create_spectrograms(dataset, window_size, spectrogram_creator, num_segments=None,  
                        **kwargs): # parameters are filters for annotation file
    
    dataset_name = dataset.__class__.__name__
    annot = AnnotationFileHandler().filter_data(dataset_name=dataset_name, **kwargs)
    
    for info in annot:
        filename, label, orig_sr = info['filename'], info['label'], int(info['sampling_rate'])
        filepath = os.path.join('data/raw', dataset_name.lower(), filename+'.mat')
        # load signal
        signal, label = dataset.load_signal_by_path(filepath)
        # Processes the data
        generate_spectrogram(signal, label, filename, window_size,
                             num_segments, orig_sr, spectrogram_creator)

