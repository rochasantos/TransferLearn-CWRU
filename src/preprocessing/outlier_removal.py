import numpy as np

def outlier_remove(self, signal):
    """ Removes outliers from the signal based on the threshold.

    Params
        signal (np.array) : Original signal from which outliers will be removed.
        
    Return 
        Signal with outliers removed.
    """
    mean = np.mean(signal)
    std_dev = np.std(signal)
    filtered_signal = np.where(np.abs(signal - mean) > self.threshold * std_dev, mean, signal)
    return filtered_signal