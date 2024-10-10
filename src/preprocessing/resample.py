import librosa
import numpy as np

def resample_data(data, original_sr, target_sr=42000):
    """ Resamples the input data to the target sampling rate.

    Params
        data (list of np.array): List of signal samples to be resampled.
        original_sr (int): Original sampling rate of the signal.
        target_sr (int, optional): Desired target sampling rate (default is 42000).

    Return
        np.array: Array of resampled data.
    """
    resampled_data = []
    for sample in data:
        resampled_sample = librosa.resample(sample, orig_sr=original_sr, target_sr=target_sr)
        resampled_data.append(resampled_sample)
    return np.array(resampled_data)
