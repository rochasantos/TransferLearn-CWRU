from scripts.experiments.biased_kfold import biesed_kfold
from src.models.cnn2d import CNN2D

import sys
import logging
from utils.logginout import LoggerWriter
from utils.print_info import print_info


def experimenter():
    # Getting a model factory
    model = CNN2D()

    # Training parameters 
    num_epochs = 20
    learning_rate = 0.0005
    batch_size = 32    

    # Experiment log
    experiment_title = "Kfold (No Similarity Bias Mitigation)"
    print_info("Experiment", [experiment_title.lower()])
    print_info("\nModel", [str(model)])
    print_info("\nTrain Parameters", [
         f"num_epochs: {num_epochs}",
         f"learning_rate: {learning_rate}",
         f"batch_size: {batch_size}"])
    print_info("\nSequence of setups for the experiment\n")    

    # Running the experiment 
    biesed_kfold(model, num_epochs, learning_rate, batch_size, repetition=1)


if __name__ == "__main__":
    sys.stdout = LoggerWriter(logging.info, "kfold_no-similarity-bias-mitigation")
    experimenter()