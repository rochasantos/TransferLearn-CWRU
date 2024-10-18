import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def compute_and_save_spectrogram(segment, output, fs, nperseg, noverlap):        
    # Compute the Short-Time Fourier Transform (STFT) to get the spectrogram
    f, t, Sxx = signal.stft(segment, nperseg=nperseg, fs=fs, noverlap=noverlap)
    # Compute the spectrogram
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(np.fliplr(abs(Sxx).T).T, cmap='viridis', aspect='auto',
                extent=[t.min(), t.max(), f.min(), f.max()])
    plt.ylabel('Frequency [kHz]')
    plt.xlabel('Number of Samples')
    plt.axis('off')  # Turn off axis labels and ticks for the spectrogram
    
    plt.savefig(output, bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # Close the figure to free up memory