from abc import ABC, abstractmethod

class SpectrogramCreator(ABC):
    def __init__(self, preprocessing_pipeline, fs, nperseg, noverlap):
        self.preprocessing_pipeline = preprocessing_pipeline
        self.target_sr = fs
        self.nperseg=nperseg
        self.noverlap=noverlap

    def generate_spectrogram(self, data, output, original_sr):
        # Data preprocessing
        preprocessed_data = self.preprocess_data(data, original_sr)
        # Creation of spectrograms
        self.create_spectrogram(preprocessed_data, output, self.target_sr)

    def preprocess_data(self, data, original_sr):
        data_processed = self.preprocessing_pipeline.process(data, original_sr)
        return data_processed

    @abstractmethod
    def create_spectrogram(self, data, output, fs, nperseg, noverlap):
        pass
