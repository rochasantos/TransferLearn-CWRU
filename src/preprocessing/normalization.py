import numpy as np
from src.preprocessing.base import PreprocessingStrategy

class NormalizationStrategy(PreprocessingStrategy):
    def __init__(self, method='min_max'):
        """
        Normalizes the signal for spectrogram generation.

        Args:
            method (str): Normalization method. Options are 'min_max' for Min-Max normalization
                          and 'z_score' for Z-score normalization. Default is 'min_max'.
        """
        self.method = method

    def process(self, signal, _):
        """
        Normalizes the input signal.

        Args:
            signal (np.ndarray): The input audio signal to be normalized.

        Returns:
            np.ndarray: The normalized audio signal.
        """
        if self.method == 'min_max':
            # Min-Max normalization to the range [-1, 1]
            norm_signal = 2 * ((signal - np.min(signal)) / (np.max(signal) - np.min(signal))) - 1
        elif self.method == 'z_score':
            # Z-score normalization (mean 0, standard deviation 1)
            norm_signal = (signal - np.mean(signal)) / np.std(signal)
        else:
            raise ValueError("Invalid normalization method. Choose 'min_max' or 'z_score'.")

        return norm_signal
