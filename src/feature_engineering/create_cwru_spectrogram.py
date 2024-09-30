import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io
import re


def _extract_data(filepath, acquisition_maxsize=12000*2):
    """
    Extracts data from a MATLAB .mat file for a specific accelerometer position.

    Parameters:
    - filepath (str): The path to the .mat file.
    - acquisition_maxsize (int): Maximum number of samples to extract (default: 12000).
    
    Returns:
    - np.array: Extracted accelerometer data.
    """
    # Load the .mat file into a dictionary
    matlab_file = scipy.io.loadmat(filepath)
    
    # Find keys that match the naming convention of time-domain data (e.g., 'X123_DE_time')
    keys = re.findall(r'X\d{3}_[A-Z]{2}_time', str(matlab_file.keys()))
    
    # Define the accelerometer positions of interest ('DE' for Drive End)
    positions = [ 'DE' ]  # Option to add more positions such as 'FE' for Fan End
    
    # Extract the bearing position from the filename based on its characters
    filename = os.path.basename(filepath)
    bearing_position = positions if filename[6:8] == 'NN' else [filename[6:8]]
    
    # Loop through the matching keys and return the data corresponding to the bearing position
    for key in keys:
        if key[-7:-5] in bearing_position:
            # Return the data limited by acquisition_maxsize or return the full data if no limit
            if acquisition_maxsize:
                return matlab_file[key][:acquisition_maxsize, :]
            else:
                return matlab_file[key]


def generate_spectrogram(input_dir, output_dir, sample_rate=12000, overlap=0, verbose=False):
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
    nperseg = window_size  # Number of samples per segment (1 second)
    noverlap = overlap  # Overlap between segments

    # Track progress across all files
    total_files = len([f for f in os.listdir(input_dir) if f.endswith('.mat')])
    file_count = 0

    # Loop through all .mat files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.mat'):
            file_count += 1  # Update the file counter
            # Map the severity level (extracted from filename) to a specific directory name
            severity = filename[2:5]
            map_out_dirname = {
                '000': '1_healthy_000',
                '007': '2_fault_severity_007',
                '014': '3_fault_severity_014',
                '021': '4_fault_severity_021',
                '028': '5_fault_severity_028'
            }
            # Extract the correct folder name based on fault severity level
            folder_name = map_out_dirname[severity]
            
            # Get the base name of the file (without directory path)
            spec_filename = os.path.basename(filename)
            
            # Create the output directory for the spectrogram images if it does not exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Full path to the input .mat file
            input_path = os.path.join(input_dir, filename)
            
            # Skip files that do not have a sampling rate of 12 kHz
            if filename[-9:-4] != '12000':
                continue
            
            # Extract the data from the .mat file using the _extract_data function
            data = _extract_data(input_path)
            not verbose or print(f"Processing file {file_count}/{total_files}: {filename}, data shape: {data.shape}")
            
            # Compute and save spectrograms for 1-second segments of the data
            num_segments = len(data) // window_size  # Number of 1-second segments in the data
            for i in range(0, len(data) - window_size + 1, window_size):  # Process 1-second intervals
                # Save the spectrogram image to the specified output directory
                output_filename = os.path.join(output_dir, folder_name, spec_filename + '_{}.png'.format(int(i/sample_rate)))
                
                if os.path.exists(output_filename):
                    continue

                segment = data[i:i + window_size, 0]  # Extract a 1-second segment
                
                # Compute the Short-Time Fourier Transform (STFT) to get the spectrogram
                f, t, Sxx = signal.stft(segment, fs=sample_rate, nperseg=1024)
                
                # Plot the spectrogram using matplotlib
                fig = plt.figure(figsize=(8, 6))
                plt.imshow(np.fliplr(abs(Sxx).T).T, cmap='viridis', aspect='auto',
                           extent=[t.min(), t.max(), f.min(), f.max()])
                plt.ylabel('Frequency [kHz]')
                plt.xlabel('Number of Samples')
                plt.axis('off')  # Turn off axis labels and ticks for the spectrogram
                
                
                plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
                plt.close(fig)  # Close the figure to free up memory
                
                # Print progress for each segment
                segment_num = int(i / window_size) + 1
                not verbose or print(r"  Segment {segment_num}/{num_segments} processed")

    # Print a completion message after all spectrograms have been generated
    print('All files processed. Complete!')


