import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from src.data_processing.dataset import SpectrogramImageDataset
import copy
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import confusion_matrix
from scripts.experiments.helper import grouper
from scripts.train_model import train_model
from src.data_processing import DatasetManager

def print_confusion_matrix(cm, class_names):
    """Displays the confusion matrix in the console."""
    print(f"{'':<3}" + "".join(f"{name:<5}" for name in class_names))
    for i, row in enumerate(cm):
        print(f"{class_names[i]:<3}" + "".join(f"{val:<5}" for val in row))

def kfold(model, train_folds, test_folds, num_epochs=50, learning_rate=0.001, batch_size=32, device="cuda"):

    transform = transforms.Compose([
        transforms.Resize((224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
    ]) 

    root_dir = "data/spectrograms/cwru_cv/"
    
    train_datasets = [
        ImageFolder(root_dir + fold, transform) for fold in train_folds 
    ]

    test_datasets = [
        ImageFolder(root_dir + fold, transform) for fold in test_folds   
    ]    

    initial_state = copy.deepcopy(model.state_dict())
    
    total_accuracy = []
    for train_dataset in train_datasets:                   
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        print('Starting model training...')   
        model = train_model(model, train_loader, num_epochs, learning_rate, device, initial_state)

        print('Training completed. Starting model evaluation...')
        
        for i, test_dataset in enumerate(test_datasets):
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            correct = 0
            total = 0
            fold_true_labels = []
            fold_predicted_labels = []

            # Model evaluation loop
            with torch.no_grad():
                for batch in test_loader:
                    images, labels = batch  # Unpack the tuple (images, labels)
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    fold_true_labels.extend(labels.cpu().numpy())  # Add true labels
                    fold_predicted_labels.extend(predicted.cpu().numpy())  # Add predictions

            accuracy = 100 * correct / total
            total_accuracy.append(accuracy)
            print(f'Model accuracy on the test set for Fold {i + 1}: {accuracy:.2f}%')

            # Calculate and display the confusion matrix for the current fold, ensuring all classes are shown
            cm = confusion_matrix(fold_true_labels, fold_predicted_labels)
            print(f'Confusion Matrix for Fold {i + 1}:')
            print(cm)

    # total_accuracy_mean = sum(total_accuracy) / len(total_accuracy)
    # print(f'Total Accuracy Mean: {total_accuracy_mean:.2f}%')
