import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from .base import SpectrogramCreator

class STFTSpectrogramCreator(SpectrogramCreator):
    def __init__(self, preprocessing_pipeline, fs, nperseg, noverlap):
        super().__init__(preprocessing_pipeline, fs, nperseg, noverlap)

    def create_spectrogram(self, data, output, fs):
            
        # Calculate the spectrogram
        f, t, Sxx = signal.stft(data, fs=fs, nperseg=self.nperseg, noverlap=self.noverlap)
        fig = plt.figure(figsize=(8, 6))
        plt.imshow(np.fliplr(abs(Sxx).T).T, cmap='viridis', aspect='auto',
                   extent=[t.min(), t.max(), f.min(), f.max()])
        plt.ylabel('Frequency [kHz]')
        plt.xlabel('Number of Samples')
        plt.axis('off')

        # Save the spectrogram
        plt.savefig(output, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
