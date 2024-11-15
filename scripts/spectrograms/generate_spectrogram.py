import os
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from datasets import CWRU, Paderborn, Hust, UORED

def generate_spectrogram(metainfo, spectrogram_setup, signal_length, num_segments=None):
    
    dataset_name = metainfo[0]["dataset_name"]
    dataset = eval(dataset_name+"()")

    for info in metainfo:
        basename = info["filename"]        
        filepath = os.path.join('data/raw/', dataset_name.lower(), basename+'.mat')
        
        data, label = dataset.load_signal_by_path(filepath)
        
        detrended_data = signal.detrend(data)
        
        n_segments = data.shape[0] // signal_length
        n_max_segments = min([num_segments or n_segments, n_segments])
        
        for i in range(0, signal_length * n_max_segments, signal_length):
            sample = detrended_data[i:i+signal_length] 
            # STFT
            f, t, Sxx = signal.stft( sample, **spectrogram_setup )

            # Plotar o espectrograma
            fig = plt.figure(figsize=(10, 6))
            plt.imshow(np.log(np.abs(Sxx[: 382, :]**2)), cmap='jet', aspect='auto')
            plt.axis('off')
            plt.gca().invert_yaxis()

            # Save the spectrogram
            output = os.path.join('data/spectrograms', dataset_name.lower(), label, basename+'#{}.png'.format(int((i+1)/signal_length)))
            plt.savefig(output, bbox_inches='tight', pad_inches=0)
            plt.close(fig)