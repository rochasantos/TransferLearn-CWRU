import numpy as np

def zero_mean(self, signal):
    """ Centers the signal at zero by subtracting the mean.

    Params 
        signal (np.array): Original signal.
        
    Return
        Signal centered at zero.
    """
    mean = np.mean(signal)
    zero_mean_signal = signal - mean
    return zero_mean_signal