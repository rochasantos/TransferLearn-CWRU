import os
import numpy as np
import scipy.io

from src.feature_engineering.compute_spectrogram import compute_and_save_spectrogram

def _extract_data(filepath, acquisition_maxsize=51200):
    """
    Extracts data from a MATLAB .mat file for a specific accelerometer position.

    Parameters:
    - filepath (str): The path to the .mat file.
    - acquisition_maxsize (int): Maximum number of samples to extract (default: 12000).
    
    Returns:
    - np.array: Extracted accelerometer data.
    """
    matlab_file = scipy.io.loadmat(filepath)
    if acquisition_maxsize:
        return matlab_file['data'][:, 0][:acquisition_maxsize]
    else:
        return matlab_file['data'][:, 0]


def generate_spectrogram(input_dir, output_dir, sample_rate=51200, nperseg=1024, overlap=0):
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
    window_size = sample_rate  # 1 second of data equals sample_rate number of points
    
    # Define parameters for spectrogram computation
    fs = sample_rate  # Sampling frequency
    nperseg = nperseg  # segments for the image
    noverlap = overlap  # Overlap between segments

    # Loop through all .mat files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.mat'):
            
            # Create the output directory for the spectrogram images if it does not exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Full path to the input .mat file
            input_path = os.path.join(input_dir, filename)
            
            # Retrieve the file's base name (excluding path and .mat extension)
            spec_filename = os.path.basename(filename).split('.')[0]

            # Extract the data from the .mat file using the _extract_data function
            data = _extract_data(input_path)

            # Compute and save spectrograms for 1-second segments of the data
            for i in range(0, np.size(data) - window_size + 1, window_size):  # Process 1-second intervals
                
                # Save the spectrogram image to the specified output directory
                output_path = os.path.join(output_dir, spec_filename + '_{}.png'.format(int(i/sample_rate)))
                
                if os.path.exists(output_path):
                    continue

                segment = data[i:i + window_size]  # Extract a 1-second segment
                compute_and_save_spectrogram(segment, output_path, fs, nperseg, noverlap)
    
    # Print a completion message after all spectrograms have been generated
    print('All files processed. Complete!')


