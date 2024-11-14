from scripts.experiments.kfold import kfold
from src.models.cnn2d import CNN2D
from src.models.factory import ModelFactory

import sys
import logging
from utils.logginout import LoggerWriter
from utils.print_info import print_info


def experimenter():
    # Getting a model factory
    model_factory = ModelFactory(CNN2D)

    # Training parameters 
    num_epochs = 5
    learning_rate = 0.001
    batch_size = 32

    # Define the experiment setup.    
    fold_split_sequence = [
        {"train": (2, 3, 4), "test": (1,)},
        {"train": (1, 3, 4), "test": (2,)},
        {"train": (1, 2, 4), "test": (3,)},
        {"train": (1, 2, 3), "test": (4,)},
    ]

    # Log
    experiment_title = "Sequential Cross-Validation Leave-P-Out"
    print_info("Experiment", [experiment_title])
    print_info("\nModel", [str(model_factory.create_model())])
    print_info("\nTrain Parameters", [
         f"num_epochs: {num_epochs}",
         f"learning_rate: {learning_rate}",
         f"batch_size: {batch_size}"])
    print_info("\nSequence of setups for the experiment")
    for sequence in fold_split_sequence:
        print(f"{list(sequence.items())[0][0]}: {list(sequence.items())[0][1]} / {list(sequence.items())[1][0]}: {list(sequence.items())[1][1]}")
    print("")

    # Running the experiment 
    kfold(model_factory, fold_split_sequence, num_epochs, learning_rate, batch_size)


if __name__ == "__main__":
    sys.stdout = LoggerWriter(logging.info, "kfold")
    experimenter()