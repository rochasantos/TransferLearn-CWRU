import numpy as np
from src.preprocessing.base import PreprocessingStrategy

class OutlierRemovalStrategy(PreprocessingStrategy):
    def __init__(self, threshold):
        self.threshold = threshold

    def process(self, signal, original_sr):
        mean = np.mean(signal)
        std_dev = np.std(signal)
        filtered_signal = np.where(np.abs(signal - mean) > self.threshold * std_dev, mean, signal)
        return filtered_signal
        