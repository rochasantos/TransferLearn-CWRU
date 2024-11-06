from datasets import CWRU, UORED, Paderborn, Hust
from scripts.create_spectrograms import create_spectrograms
from src.preprocessing import PreprocessingPipeline, ResamplingStrategy, NormalizationStrategy
from scripts.create_spectrograms import create_spectrograms
from src.data_processing import DatasetManager
from scripts.experiments.kfold import kfold
from src.models import CNN2D, ResNet18

# Download datasets
def download_raw_data(datasets):    
    for dataset in datasets:
        dataset.download()

# SPECTROGRAMS
def run_create_spectrograms():
    target_sr = 48000
    num_segments = 10
    filter_config_path = 'config/filters_config.yaml'

    # Creates the preprocessing pipeline and add the strategies to the pipeline
    preprocessing_pipeline = PreprocessingPipeline()
    preprocessing_pipeline.add_step(ResamplingStrategy(target_sr=target_sr))
    preprocessing_pipeline.add_step(NormalizationStrategy())

    # Creation of spectrograms    
    create_spectrograms(filter_config_path, preprocessing_pipeline, num_segments)                        

# EXPERIMENTERS
def run_experimenter():
    model = ResNet18()    
    kfold(model, group_by="extent_damage")


if __name__ == '__main__':
    # download_rawfile('CWRU')
    # run_create_spectrograms()
    run_experimenter()
