import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def _compute_and_save_spectrogram(segment, output, fs, nperseg, noverlap):        
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

def generate_spectrogram(data, label, fs, output_dir, noverlap=0, nperseg = 1024):
    """
    Generates and saves spectrograms from raw data stored in .mat files.
    The window size is set to represent 1 second of data in the time domain.

    Parameters:
    - input_dir (str): Directory where input .mat files are stored.
    - output_dir (str): Directory where the generated spectrograms will be saved.
    - sample_rate (int): The sample rate of the signal (default: 12,000 Hz).
    - overlap (int): The overlap between windows for the spectrogram (default: 0).
    """
    
    # Define the window size as 1 second of data in the time domain
    window_size = fs  # 1 second of data equals sample_rate number of points
    
    # Loop through all .mat files in the input directory
    for sample, spec_filename in zip(data, label):
        # Compute and save spectrograms for 1-second segments of the data
        for i in range(0, data.shape[1] - window_size + 1, window_size):  # Process 1-second intervals
            # Save the spectrogram image to the specified output directory
            output_file = os.path.join(output_dir, spec_filename + '_{}.png'.format(int(i/fs)))
            
            if os.path.exists(output_file):
                continue
            
            segment = sample[i:i + window_size]  # Extract a 1-second segment
            _compute_and_save_spectrogram(segment, output_file, fs, nperseg, noverlap)
            
    # Print a completion message after all spectrograms have been generated
    print('All files processed. Complete!')
