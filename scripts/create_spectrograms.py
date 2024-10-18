import os
from src.feature_engineering.create_spectrogram import generate_spectrogram
from src.data_processing.annotation_file import AnnotationFileHandler
from datasets.cwru import CWRU
from datasets.uored import UORED
from datasets.hust import Hust
from datasets.paderborn import Paderborn
import librosa
from src.data_processing.annotation_file import AnnotationFileHandler


def create_spectrograms(dataset_name, target_sr, window_size, nperseg, noverlap, num_segments=None, **kwargs): # parameters are filters for annotation file
    annot = AnnotationFileHandler().filter_data(dataset_name=dataset_name, **kwargs)
    dataset = eval(dataset_name)()

    
    
    for info in annot:
        filename, label, sr = info['filename'], info['label'], int(info['sampling_rate'])
        filepath = os.path.join('data/raw', dataset_name.lower(), filename+'.mat')
        # load signal
        signal, label = dataset.load_signal_by_path(filepath)
        # resampling
        signal_resampled = librosa.resample(signal, orig_sr=sr, target_sr=target_sr)
        generate_spectrogram(signal_resampled, label, filename, target_sr, window_size=window_size, 
                             num_segments=num_segments, nperseg=nperseg, noverlap=noverlap)
