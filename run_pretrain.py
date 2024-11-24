import sys
import logging
from utils.logginout import LoggerWriter
from utils.print_info import print_info

from src.models import CNN2D, ViTClassifier
from scripts.pretrain_model import pretrain_model


def experimenter():
    # Getting a model factory
    model = ViTClassifier()

    # Training parameters 
    num_epochs = 10
    learning_rate = 0.0001
    batch_size = 32
    
    # Datasets that will be used in pre-training
    datasets_name = ["UORED"]
    #save_model_path = f"saved_models/cnn2d-uored-e{num_epochs}-lr{str(learning_rate).split('.')[1]}.pth"
    save_model_path = f"saved_models/vit_classifier.pth"

    # Experiment log
    experiment_title = f"Pretrain - {datasets_name}"
    print_info("Experiment", [experiment_title.lower()])
    #print_info("\nModel", [str(model)])
    print_info("\nTrain Parameters", [
         f"num_epochs: {num_epochs}",
         f"learning_rate: {learning_rate}",
         f"batch_size: {batch_size}"])
    
    # Running the pre-train 
    pretrain_model(model, datasets_name, num_epochs, batch_size, learning_rate, device='cuda', save_path=save_model_path)


if __name__ == "__main__":
    datasetname = "UORED"
    sys.stdout = LoggerWriter(logging.info, f"pretrain-{datasetname}")
    experimenter()