from datasets.cwru import CWRU
# from datasets.uored import UORED
# from datasets.hust import Hust
# from datasets.paderborn import Paderborn

from scripts.create_spectrograms import create_spectrograms

from src.preprocessing import PreprocessingPipeline, ResamplingStrategy, ZeroMeanStrategy, OutlierRemovalStrategy
from scripts.create_spectrograms import create_spectrograms
from src.feature_engineering.spectrogram import STFTSpectrogramCreator
from src.data_processing.annotation_file import AnnotationFileHandler
from scripts.experiments.kfold import kfold
from src.models import CNN2D, ResNet18

# SPECTROGRAMS
## Handle spectrogram creation.
def run_create_spectrograms():
    # PARAMS to create the spectrograms
    dataset = CWRU() # Paderborn, Hust, UORED
    window_size = 256
    nperseg = 512
    noverlap = 0
    num_segments = 30
    target_sr = 12000

    # Creates the preprocessing pipeline and add the strategies to the pipeline
    preprocessing_pipeline = PreprocessingPipeline()
    preprocessing_pipeline.add_step(ResamplingStrategy(target_sr=target_sr))
    # preprocessing_pipeline.add_step(ZeroMeanStrategy())
    # preprocessing_pipeline.add_step(OutlierRemovalStrategy(threshold=3.0))

    # Creation of spectrograms 
    spectrogram_creator = STFTSpectrogramCreator(preprocessing_pipeline, target_sr, nperseg, noverlap)

    # Creating the spectrograms
    create_spectrograms(dataset, window_size, spectrogram_creator, num_segments) #,
                        # bearing_type=r'6203|6205') # add regex to filter the data                        


# EXPERIMENTERS
def experimenter_kfold():

    model = CNN2D().to('cuda')
    # model = ResNet18().to('cuda')

    file_info = AnnotationFileHandler().filter_data(sampling_rate='12000', label=r'N|I|B|O')
    kfold(model, file_info, group_by="extent_damage")


if __name__ == '__main__':        
    run_create_spectrograms()
    experimenter_kfold()
