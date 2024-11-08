from matplotlib import pyplot as plt
import numpy as np
import os
import re
import scipy.io
from scipy import signal
from utils.filter_files import filter_files_with_regex

def create_spectrograms(preprocessing_pipeline, num_segments=None):
    
    # Parameters
    fs = int(42e3)  
    signal_length = 10500
    window = 'hann'  
    nperseg = 580    
    noverlap = int(nperseg * 0.96)  
    nfft = 1600

    filtered_files = filter_files_with_regex('data/raw/uored/', r'[HIOB]_\d+_[02].mat')
    for file in filtered_files:

        basename = os.path.basename(file)[:-4]
        label = basename[0] if basename[0] != 'H' else 'N'
        
        data = scipy.io.loadmat(file)[basename][:, 0]
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