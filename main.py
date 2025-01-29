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
from src.data_processing import PtDataset, TransformPipeline


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
    

def train_model(dataloader, model, n_epochs=100, learning_rate=0.0001, save_path=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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
            loss = ce_loss + triplet_loss

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
    validation_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    criterion = nn.CrossEntropyLoss()   
    
    model.eval()

    validation_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for data in validation_loader:
            inputs, labels = data
            features, outputs = model(inputs)
            loss = criterion(outputs, labels)  # Define sua função de perda
            
            validation_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)  # Para problemas de classificação
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.numpy())

    # Calcula a perda média
    validation_loss = validation_loss / len(validation_loader.dataset)

    # Calcula a acurácia
    accuracy = accuracy_score(all_labels, all_preds)

    print(f'Validation Loss: {validation_loss:.4f}')
    print(f'Validation Accuracy: {accuracy:.4f}')  

    # Exibir a matriz de confusão usando matplotlib
    print(confusion_matrix(all_labels, all_preds))

    return accuracy

# def baseline():
#     sample_size = 100_000
#     data_filter = load_yaml('config/filters_config.yaml')["CWRU"]
#     ds_test = PtDataset(data_filter, sample_size, apply_augmentation=False)
#     print(f"Total test samples: {len(ds_test)}")            
#     ds_train = PtDataset({**data_filter, "extent_damage": [el[1] for el in att_train]}, sample_size, apply_augmentation=False),
#         PtDataset({**data_filter, "hp": [el[0] for el in att_train], "label": "N"}, sample_size, apply_augmentation=False),
#         # PtDataset({**data_filter, "extent_damage": [el[1] for el in att_train]}, sample_size, apply_augmentation=True),
#         # PtDataset({**data_filter, "hp": [el[0] for el in att_train], "label": "N"}, sample_size, apply_augmentation=True)
#         ])
#     print(f"Total train samples: {len(ds_train)}")
#     dl_train = DataLoader(ds_train, batch_size=32, shuffle=True)
#     model.load_state_dict(torch.load("initial_weights.pth", weights_only=False))
#     train_model(dl_train, model, n_epochs=n_epochs, learning_rate=0.0001)
#     accuracy = validation(model, ds_test)

def kfold(model, n_epochs=100, repetition=10):
    total_accuracies = []
    for rep in range(repetition):
        accuracies = []
        print(f"Repetition: {rep+1}")
        sample_size = 12000
        dataset_name = "CWRU"
        label_mapping = {"N": 0, "I": 1, "O": 2, "B": 3}
        n_class = len(label_mapping)

        data_filter = load_yaml('config/filters_config.yaml')[dataset_name]
        attributes = [("0", "007"), ("1", "014"), ("2", "021")]
        for att in attributes:
            att_test = att
            att_train = [el for el in attributes if el[1] != att[1]]
            
            print("Test")
            ds_test = ConcatDataset(
                [PtDataset({**data_filter, "extent_damage": att_test[1]}, sample_size, apply_augmentation=False),
                # PtDataset({**data_filter, "hp": att_test[0], "label": "N"}, sample_size, apply_augmentation=False)
                ])
            print(f"Total test samples: {len(ds_test)}")
            
            print("Train")
            ds_train = ConcatDataset([
                PtDataset({**data_filter, "extent_damage": [el[1] for el in att_train]}, sample_size, apply_augmentation=False),
                # PtDataset({**data_filter, "hp": [el[0] for el in att_train], "label": "N"}, sample_size, apply_augmentation=False),
                # PtDataset({**data_filter, "extent_damage": [el[1] for el in att_train]}, sample_size, apply_augmentation=True),
                # PtDataset({**data_filter, "hp": [el[0] for el in att_train], "label": "N"}, sample_size, apply_augmentation=True)
                ])
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
    sys.stdout = LoggerWriter(logging.info, "kfold_mitigated_with_da")
    
    # download()   
    
    # Parameters
    n_epochs = 50
    repetitions = 1
    
    # Experimenter
    model = SignalFeatureCNN1D()
    torch.save(model.state_dict(), "initial_weights.pth")

    kfold(model, n_epochs=n_epochs, repetition=repetitions)
