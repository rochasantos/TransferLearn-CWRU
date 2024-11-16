from datasets import CWRU, UORED, Paderborn, Hust
from scripts.create_spectrograms import create_spectrograms
from src.preprocessing import PreprocessingPipeline, ResamplingStrategy, NormalizationStrategy
from src.data_processing import DatasetManager
from src.models import CNN2D, ResNet18, ViTClassifier
from scripts.experiments.modelvalidation import resubstitution_test, one_fold_with_bias, one_fold_without_bias, kfold_cross_validation
from src.models.vitclassifier import train_and_save, load_trained_model
from scripts.download_rawfile import download_rawfile
from torchvision import transforms
from src.data_processing.dataset import SpectrogramImageDataset
from torch.utils.data import DataLoader, Subset
import numpy as np
from utils.dual_output import DualOutput  # Import the class from dual_output.py
from datetime import datetime
import sys
import os

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Generate a timestamped log file name
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
log_filename = f"logs/experiment_log_{timestamp}.txt"

# Redirect stdout
sys.stdout = DualOutput(log_filename)

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
    #model = ResNet18() 
    num_epochs_vit_train = 20
    lr_vit_train = 0.0002
    save_path = "vit_classifier.pth"  # Define path to save the ViT trained model
        
    dataset_manager = DatasetManager()

    # Experimenter parameters
    root_dir = 'data/spectrograms'

    transform = transforms.Compose([
        # transforms.Resize((516, 516)),  
        transforms.ToTensor(),
     ])

    # The dataset1 = CWRU
    class_names = ['N', 'I', 'O', 'B']  # Your dataset class names
    
     # Load only CWRU dataset
    file_info_CWRU = dataset_manager.filter_dataset(dataset_name=r'^CWRU$')
    dataset_CWRU = SpectrogramImageDataset(root_dir, file_info_CWRU, class_names, transform)

    # Load only UORED dataset
    file_info_UORED = dataset_manager.filter_dataset(dataset_name=r'^UORED$')
    dataset_UORED = SpectrogramImageDataset(root_dir, file_info_UORED, class_names, transform)
 
    dataset1 = dataset_CWRU
    dataset2 = dataset_UORED
    
    print("Train dataset:" ,dataset2.get_dataset_name())
    print("Test dataset:" ,dataset1.get_dataset_name())
 
    # Train ViTClassifier with dataset2 = UORED
    vit_train_dataset = dataset2
    vit_train_dataloader = DataLoader(vit_train_dataset, batch_size=32, shuffle=True)
    
    # Instantiate the ViTClassifier and train it with dataset2 to narrow the model context
    #model = ViTClassifier().to("cuda")
    #train_and_save(model, vit_train_dataloader, num_epochs_vit_train, lr_vit_train, save_path)  # Train and save the model

    # Load the trained model for testing/evaluation
    model = load_trained_model(ViTClassifier, save_path, num_classes=len(class_names)).to("cuda")
    
    num_epochs = 10
    lr = 0.002
    group_by = "rpm" 
    #group_by = "extent_damage"
    #group_by = "condition_bearing_health"
    #group_by = "damage_method"
    #group_by = ""
    
    #resubstitution_test(model, dataset2, num_epochs, lr, class_names)               # Resubstitution error validation
    #one_fold_with_bias(model, dataset1, num_epochs, lr, class_names)                # Train and test with 1 fold and bias
    #one_fold_without_bias(model, dataset1, num_epochs, lr, class_names)             # Train and test with 1 fold without bias
    
    kfold_cross_validation(model, dataset1, num_epochs, lr, group_by, class_names, n_splits=4)


if __name__ == '__main__':
    #download_rawfile('CWRU')
    #download_rawfile('UORED')
    #run_create_spectrograms()
    run_experimenter()



    # Close the log file
    sys.stdout.close()
    sys.stdout = sys.__stdout__  # Reset stdout to the original