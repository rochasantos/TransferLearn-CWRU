import torch
from scripts.experiments.mitigated_kfold import mitigated_kfold
from src.models.cnn2d import CNN2D

import sys
import logging
from utils.logginout import LoggerWriter
from utils.print_info import print_info


def experimenter():
    
    # Loads the pre-trained model
    model = CNN2D()
    saved_model_path = "saved_models/cnn2d-uored-e30-lr0001.pth"
    model.load_state_dict(torch.load(saved_model_path, weights_only=True))
    
    # Freeze layers
    for name, param in model.named_parameters():
        if name.split(".")[0] not in ["conv3", "fc1", "fc2"]:
            param.requires_grad = False

    # Training parameters 
    num_epochs = 20
    learning_rate = 0.001
    batch_size = 32
    repetition = 1

    # Define the experiment setup.
    # resubstitution
    """
    fold_split_sequence = [
        {"train": (1,), "test": (1,)},
        {"train": (2,), "test": (2,)},
        {"train": (3,), "test": (3,)},
        {"train": (4,), "test": (4,)},
    ]
    """
    # kfold    
    fold_split_sequence = [
        {"train": (2, 3, 4), "test": (1,)},
        {"train": (1, 3, 4), "test": (2,)},
        {"train": (1, 2, 4), "test": (3,)},
        {"train": (1, 2, 3), "test": (4,)},
    ] 
    
    # Experiment log
    title = "Experiment KFold Mitigated by Severity With Pre-trained Model on UORED datasets."
    print(f"\n{title}")
    print_info("Model", [str(model)])
    print_info("\nLayers")
    for name, param in model.named_parameters():   
        print(f"{name}, requires_grad: {param.requires_grad}")
    print_info("\nTrain Parameters", [
         f"num_epochs: {num_epochs}",
         f"learning_rate: {learning_rate}",
         f"batch_size: {batch_size}",
         f"repetition: {repetition}"
        ])
    print_info("\nSequence of setups for the experiment")
    for sequence in fold_split_sequence:
        print(f"{list(sequence.items())[0][0]}: {list(sequence.items())[0][1]} / {list(sequence.items())[1][0]}: {list(sequence.items())[1][1]}")
    print("")

    # Running the experiment 
    mitigated_kfold(model, fold_split_sequence, num_epochs, learning_rate, batch_size, repetition=repetition)


def run_experimenter():     
    experimenter()


if __name__ == "__main__":
    sys.stdout = LoggerWriter(logging.info, "kfold-mitigated-bias-with-pre-trained-model-test")
    run_experimenter()