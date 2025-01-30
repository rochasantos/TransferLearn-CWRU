import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from src.models import SignalFeatureCNN1D
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import os
import logging
from utils import LoggerWriter

# from utils.download_rawfile import download_rawfile
from utils import load_yaml
from src.data_processing import PtDataset


# DOWNLOAD RAW FILES
# def download():
#     for dataset in ["Paderborn"]:
#         download_rawfile(dataset)

    
# Define triplet loss
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Calculate distances
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        # Triplet loss formula
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()
    

def train_model(dataloader, model, n_epochs=100, learning_rate=0.0001, with_triplet_loss=False, save_path=None):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    cross_entropy_loss = nn.CrossEntropyLoss()
    triplet_loss_fn = TripletLoss(margin=1.0)

    # Training loop
    for epoch in range(n_epochs):  # Number of epochs
        total_loss = 0
        for batch_data, batch_labels in dataloader:

            # Forward pass
            features, outputs = model(batch_data)

            # Cross-entropy loss
            ce_loss = cross_entropy_loss(outputs, batch_labels)

            # Triplet loss
            anchor = features[::3]  # Every third example as anchor
            positive = features[1::3]  # Every third example + 1 as positive
            negative = features[2::3]  # Every third example + 2 as negative

            # Ensure the triplet sizes match
            min_triplet_size = min(len(anchor), len(positive), len(negative))
            anchor = anchor[:min_triplet_size]
            positive = positive[:min_triplet_size]
            negative = negative[:min_triplet_size]

            triplet_loss = triplet_loss_fn(anchor, positive, negative)

            # Combine losses
            if with_triplet_loss:
                loss = ce_loss + triplet_loss
            else:
                loss = ce_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Total Loss: {total_loss:.4f}")

    # Saving model
    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.mkdir("saved_models")
        torch.save(model.state_dict(), save_path)   


def validation(model, dataset):
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    criterion = nn.CrossEntropyLoss()   
    
    model.eval()

    validation_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            features, outputs = model(inputs)
            loss = criterion(outputs, labels)  # Define sua função de perda
            
            validation_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)  # Para problemas de classificação
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Accuracy: {accuracy:.4f}')  

    print(confusion_matrix(all_labels, all_preds))

    return accuracy


def baseline(sample_size=12000, n_epochs=100):
    
    model = SignalFeatureCNN1D(sample_size)
    torch.save(model.state_dict(), "initial_weights.pth")

    data_filter = load_yaml('config/filters_config.yaml')["CWRU"]
    
    ds_train = ConcatDataset([
        # PtDataset(data_filter, sample_size, apply_augmentation=True),
        PtDataset(data_filter, sample_size, apply_augmentation=True)
    ])
    ds_test = PtDataset(data_filter, sample_size, apply_augmentation=False)
    
    dataloader_tr = DataLoader(ds_train, batch_size=32, shuffle=True)

    train_model(dataloader_tr, model, n_epochs=150, with_triplet_loss=True, learning_rate=0.0001)    
    model.freeze_layers(["conv1", "conv2"])
    train_model(
        DataLoader(PtDataset(data_filter, sample_size, apply_augmentation=False), batch_size=32, shuffle=True),
        model, n_epochs=n_epochs, with_triplet_loss=False, learning_rate=0.0001
    )           
    accuracy = validation(model, ds_test)
    print(f"Accuracy: {accuracy}")


def kfold(sample_size=12000, n_epochs=100, with_augmentation=False, repetition=10):

    model = SignalFeatureCNN1D(sample_size)
    torch.save(model.state_dict(), "initial_weights.pth")

    total_accuracies = []
    for rep in range(repetition):
        accuracies = []
        print(f"Repetition: {rep+1}")
        sample_size = 12000
        dataset_name = "CWRU"
        data_filter = load_yaml('config/filters_config.yaml')[dataset_name]

        attributes = ["007", "014", "021"]
        for att_test in attributes:
            ds_test = PtDataset({**data_filter, "extent_damage": att_test}, sample_size, apply_augmentation=False)
            att_train = [el for el in attributes if el != att_test]
            if with_augmentation:
                ds_train = ConcatDataset([
                    PtDataset({**data_filter, "extent_damage": [el for el in att_train]}, sample_size, apply_augmentation=False),
                    PtDataset({**data_filter, "extent_damage": [el for el in att_train]}, sample_size, apply_augmentation=True)
                ])
            else:
                ds_train = PtDataset({**data_filter, "extent_damage": [el for el in att_train]}, sample_size=sample_size, apply_augmentation=False)

            print({**data_filter, "extent_damage": att_test})
            print(f"Total test samples: {len(ds_test)}")
            print(f"Total train samples: {len(ds_train)}")
            
            dl_train = DataLoader(ds_train, batch_size=32, shuffle=True)
            
            model.load_state_dict(torch.load("initial_weights.pth", weights_only=False))
            
            train_model(dl_train, model, n_epochs=n_epochs, learning_rate=0.0001)
            
            accuracy = validation(model, ds_test)
            accuracies.append(accuracy)
            print(f"accuracies: {accuracies}")
        total_accuracies.append(accuracies)
        print(f"total_accuracy: {total_accuracies}")
    total_accuracies = np.array(total_accuracies)
    print(f"shape total accuracy: {total_accuracies.shape}")
    total = np.mean(total_accuracies)
    print(f"Total: {np.round(total, 2)}")
    print(f"Accuracy per fold: {np.mean(total_accuracies, axis=0)}")



if __name__ == '__main__':
    # download()
   
    # Parameters
    n_epochs = 50
    repetitions = 5
    sample_size = 12000
    log_path = "kfold_mitigated_withou_da"

    sys.stdout = LoggerWriter(logging.info, log_path)

    # Experimenter
    # kfold(sample_size=sample_size, n_epochs=n_epochs, with_augmentation=False, repetition=repetitions)
    baseline(sample_size=sample_size, n_epochs=n_epochs)