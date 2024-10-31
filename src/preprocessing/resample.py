import librosa
import numpy as np
from src.preprocessing.base import PreprocessingStrategy

class ResamplingStrategy(PreprocessingStrategy):
    def __init__(self, target_sr):
        self.target_sr = target_sr

    def process(self, signal, original_sr):        
        return librosa.resample(signal, orig_sr=original_sr, target_sr=self.target_sr)

