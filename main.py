import sys
import logging
from utils.logginout import LoggerWriter

from datasets import CWRU, UORED, Paderborn, Hust
from src.preprocessing import PreprocessingPipeline, ResamplingStrategy
from scripts.download_rawfile import download_rawfile
from scripts.spectrograms import create_cwru_spectrogram
from scripts.copy_spectrogram_to_folds import copy_spectrogram_to_folds

# SPECTROGRAMS
def create_spectrograms():
    num_segments = 20

    # Creates the preprocessing pipeline and add the strategies to the pipeline
    preprocessing_pipeline = PreprocessingPipeline()
    # preprocessing_pipeline.add_step(ResamplingStrategy())

    # Creation of spectrograms    
    create_cwru_spectrogram(preprocessing_pipeline, num_segments) 


if __name__ == '__main__':
    sys.stdout = LoggerWriter(logging.info)
    download_rawfile('CWRU')
    create_spectrograms()
    copy_spectrogram_to_folds()
    