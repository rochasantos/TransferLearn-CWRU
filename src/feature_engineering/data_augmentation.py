import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import resample

class DataAugmentation:
    def frequency_flip(self, spectrogram):
        """ Flips the spectrogram vertically (frequency flip).
        
        Param 
            spectrogram: Original spectrogram.
        
        Return 
            Spectrogram with frequencies flipped.
        """
        return np.flipud(spectrogram)

    def time_flip(self, spectrogram):
        """ Flips the spectrogram horizontally (time flip).
        
        Param 
            spectrogram: Original spectrogram.
        
        Return: 
            Spectrogram with time axis flipped.
        """
        return np.fliplr(spectrogram)

    def gaussian_noise(self, signal, mean=0, std_dev=0.01):
        """ Adds Gaussian noise to the signal.
        
        Param 
            signal (np.array): Original signal.
            mean (float): Mean of the noise.
            std_dev (float): Standard deviation of the noise.
        
        Return: 
            Signal with added Gaussian noise.
        """
        noise = np.random.normal(mean, std_dev, signal.shape)
        noisy_signal = signal + noise
        return noisy_signal

    def pitch_shift(self, signal, sampling_rate, pitch_factor):
        """ Shifts the pitch of the signal.
        
        Params 
            signal (np.array): Original signal.
            sampling_rate (int): Sampling rate of the signal.
            pitch_factor: Pitch shift factor (positive for higher pitch, negative for lower pitch).
        
        Return
            Signal with altered pitch.
        """
        factor = 2 ** (pitch_factor / 12)  # Calculates the pitch shift factor in semitones
        new_length = int(len(signal) / factor)
        resampled_signal = resample(signal, new_length)
        return resample(resampled_signal, len(signal))  # Resample back to the original length
