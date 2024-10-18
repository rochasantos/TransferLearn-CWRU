import numpy as np
from src.preprocessing.base import PreprocessingStrategy

class ZeroMeanStrategy(PreprocessingStrategy):
    def process(self, signal, original_sr):
        return signal - np.mean(signal)
    