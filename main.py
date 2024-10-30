from datasets.cwru import CWRU
from datasets.uored import UORED
from datasets.hust import Hust
from datasets.paderborn import Paderborn

from src.preprocessing import PreprocessingPipeline, ResamplingStrategy, ZeroMeanStrategy, OutlierRemovalStrategy
from scripts.create_spectrograms import create_spectrograms
from src.feature_engineering.spectrogram import STFTSpectrogramCreator
from src.data_processing.annotation_file import AnnotationFileHandler
from scripts.experiments.kfold import kfold
from src.models import CNN2D, ResNet18

from scripts import download_rawfile


# SPECTROGRAMS
## Handle spectrogram creation.
def run_create_spectrograms():
    # PARAMS to create the spectrograms
    dataset = CWRU() # Paderborn, Hust, UORED
    window_size = 600
    nperseg = 580
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
                        bearing_type=r'6203|6205') # add regex to filter the data                        


# EXPERIMENTERS
def experimenter_kfold():

    model = CNN2D().to('cuda')
    # model = ResNet18().to('cuda')

    file_info = AnnotationFileHandler().filter_data(label=r'N|I|B|O')
    kfold(model, file_info, group_by="extent_damage")


if __name__ == '__main__':        
    download_rawfile('Hust')
    