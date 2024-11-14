from scripts.experiments.kfold import kfold
from src.models.cnn2d import CNN2D

import sys
import logging
from utils.logginout import LoggerWriter

def experimenter():
    # Getting the model
    model = CNN2D()

    # Training parameters 
    num_epochs = 15
    learning_rate = 0.001
    batch_size = 32

    # Set folds
    train_folds = ["fold2"]
    test_folds = ["fold2"]

    # Running the experiment    
    kfold(model, train_folds, test_folds, num_epochs, learning_rate, batch_size)

if __name__ == "__main__":
    sys.stdout = LoggerWriter(logging.info)
    experimenter()