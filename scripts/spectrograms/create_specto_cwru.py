from matplotlib import pyplot as plt
import numpy as np
import os
import re
import csv
import scipy.io
from scipy import signal
from datasets.cwru import CWRU
from utils.filter_files import filter_files_with_regex

def create_spectrograms(preprocessing_pipeline, num_segments=None):

    dataset = CWRU()
    
    # Parameters
    fs = int(48e3)  
    signal_length = 12000
    window = 'hann'  
    nperseg = 600    
    noverlap = int(nperseg * 0.96)  
    nfft = 1600


    metainfo = []
    with open('data/annotation_file.csv', mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            metainfo.append(row)

    for info in metainfo:

        if info["dataset_name"] != 'CWRU' or info["bearing_type"] != '6205':
            continue

        basename = info["filename"]        
        filepath = os.path.join('data/raw/cwru', basename+'.mat')
        
        data, label = dataset.load_signal_by_path(filepath)
        
        detrended_data = signal.detrend(data)
        
        n_segments = data.shape[0] // signal_length
        n_max_segments = min([num_segments or n_segments, n_segments])
        
        for i in range(0, signal_length * n_max_segments, signal_length):
            sample = detrended_data[i:i+signal_length] 
            # STFT
            frequencies, times, Sxx = signal.stft(
                sample, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft
            )

            # Plotar o espectrograma
            fig = plt.figure(figsize=(10, 6))
            plt.imshow(np.log(np.abs(Sxx[: 382, :]**2)), cmap='jet', aspect='auto')
            plt.axis('off')
            plt.gca().invert_yaxis()

            # Save the spectrogram
            output = os.path.join('data/spectrograms', label, basename+'#{}.png'.format(int((i+1)/signal_length)))
            plt.savefig(output, bbox_inches='tight', pad_inches=0)
            plt.close(fig)