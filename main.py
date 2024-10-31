from datasets.cwru import CWRU
from datasets.uored import UORED
from datasets.hust import Hust
from datasets.paderborn import Paderborn

from scripts.create_spectrograms import create_spectrograms

from src.preprocessing import PreprocessingPipeline, ResamplingStrategy, ZeroMeanStrategy, OutlierRemovalStrategy
from scripts.create_spectrograms import create_spectrograms
from src.feature_engineering.spectrogram import STFTSpectrogramCreator
from src.data_processing.annotation_file import AnnotationFileHandler
from scripts.experiments.kfold import kfold
from src.models import CNN2D, ResNet18
from scripts.download_rawfile import download_rawfile

# Download datasets
def download_raw_data(datasets):    
    for dataset in datasets:
        dataset.download()

# SPECTROGRAMS
## Handle spectrogram creation.
def run_create_spectrograms(dataset=CWRU()):
    # PARAMS to create the spectrograms
    window_size = 12000
    nperseg = 590
    noverlap = 0
    num_segments = 1
    target_sr = 12000

    # Creates the preprocessing pipeline and add the strategies to the pipeline
    preprocessing_pipeline = PreprocessingPipeline()
    preprocessing_pipeline.add_step(ResamplingStrategy(target_sr=target_sr))
    # preprocessing_pipeline.add_step(ZeroMeanStrategy())
    # preprocessing_pipeline.add_step(OutlierRemovalStrategy(threshold=3.0))

    # Creation of spectrograms 
    spectrogram_creator = STFTSpectrogramCreator(preprocessing_pipeline, target_sr, nperseg, noverlap)

    # Creating the spectrograms
    create_spectrograms(dataset, window_size, spectrogram_creator, num_segments,
                        label=r'N|I|O|B', condition_bearing_health=r'healthy|faulty') # add regex to filter the data                        


# EXPERIMENTERS
def run_experimenter():
    # Model
    model = ResNet18()    
    # Experiment script
    kfold(model, group_by="extent_damage")


if __name__ == '__main__':
    # Choose "all" to download all datasets
    # Choose "CWRU", "Paderborn", "UORED" or "Hust" to downlod them individually
    download_rawfile('CWRU')
    
    create_spectrograms()
    
    run_experimenter()
