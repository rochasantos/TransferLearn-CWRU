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
        basename, label, orig_sr = info['filename'], info['label'], int(info['sampling_rate'])
        
        # load the data
        rawfile_path = os.path.join('data/raw', dataset_name.lower(), basename+'.mat')
        signal, label = dataset.load_signal_by_path(rawfile_path)
        
        output_path = os.path.join('data/spectrograms', label, basename)
        
        generate_spectrogram(signal, output_path, window_size,
                             num_segments, orig_sr, spectrogram_creator)

