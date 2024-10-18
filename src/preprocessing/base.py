from abc import ABC, abstractmethod

class PreprocessingStrategy(ABC):
    def __init__(self):
        self.target_sr=None

    @abstractmethod
    def process(self, data, original_signal):
        pass
        