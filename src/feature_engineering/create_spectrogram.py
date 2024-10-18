import os
from src.feature_engineering.compute_spectrogram import compute_and_save_spectrogram

def generate_spectrogram(signal, label, basename, sampling_rate, window_size, num_segments=None, nperseg=1024, noverlap=0, compute_spectrogram=compute_and_save_spectrogram):
    """
    Generates and saves spectrograms from raw data stored in .mat files.
    The window size is set to represent 1 second of data in the time domain.

    Parameters:
    - input_dir (str): Directory where input .mat files are stored.
    - output (str): Directory where the generated spectrograms will be saved.
    - sample_rate (int): The sample rate of the signal (default: 12,000 Hz).
    - overlap (int): The overlap between windows for the spectrogram (default: 0).
    """    
    output_dir = "data/processed/spectrograms"

    # Compute and save spectrograms for 1-second segments of the signal
    n_segments = signal.shape[0] // window_size
    num_max_segments = min([num_segments or n_segments, n_segments])
    for i in range(0, window_size * num_max_segments, window_size):
        # Save the spectrogram image to the specified output directory
        output_file = os.path.join(output_dir, label, basename+'_{}.png'.format(int(i/sampling_rate)))
        
        if os.path.exists(output_file):
            continue
        
        segment = signal[i:i + window_size]
        compute_spectrogram(segment, output_file, sampling_rate, nperseg, noverlap)
        
    print(f'{basename}.png file processed.')
