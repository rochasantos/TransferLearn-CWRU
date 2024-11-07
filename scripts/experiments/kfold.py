import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from src.data_processing.dataset import SpectrogramImageDataset
import copy
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix
from scripts.experiments.helper import grouper
from scripts.train_model import train_model
from src.data_processing import DatasetManager

def print_confusion_matrix(cm, class_names):
    """Displays the confusion matrix in the console."""
    print("Confusion Matrix:")
    print(f"{'':<5}" + "".join(f"{name:<5}" for name in class_names))
    for i, row in enumerate(cm):
        print(f"{class_names[i]:<3}" + "".join(f"{val:<5}" for val in row))

def kfold(model, group_by="extent_damage"):

    file_info = DatasetManager().filter_data()

    # Experimenter parameters
    root_dir = 'data/spectrograms'
    n_splits = 4
    
    # Training parameters
    num_epochs = 30
    learning_rate = 0.005
    batch_size = 32

    transform = transforms.Compose([
        # transforms.Resize((516, 516)),  
        transforms.ToTensor()          
    ])

    class_names = ['N', 'I', 'O', 'B']  # Your dataset class names
    dataset = SpectrogramImageDataset(root_dir, file_info, class_names, transform)

    X = np.arange(len(dataset))  # get index from spectrograms
    y = dataset.targets  # class tags
    print(X)
    groups = grouper(dataset, group_by)
    initial_state = copy.deepcopy(model.state_dict())
    total_accuracy = []
    skf = StratifiedGroupKFold(n_splits=n_splits)
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y, groups)):
        print(f"Fold {fold + 1}")
        
        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4)

        print('Starting model training...')
        
        model = model.to('cuda')

        # Initialize the weights        
        model.load_state_dict(initial_state)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch in train_loader:
                images, labels = batch  # Unpack the tuple (images, labels)
                images, labels = images.to('cuda'), labels.to('cuda')

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Initialize the weights        
        model.load_state_dict(initial_state)

        # Training model
        train_model(model, train_loader, num_epochs, batch_size, learning_rate)

        print('Training completed. Starting model evaluation...')
        correct = 0
        total = 0
        fold_true_labels = []
        fold_predicted_labels = []

        # Model evaluation loop
        with torch.no_grad():
            for batch in test_loader:
                images, labels = batch  # Unpack the tuple (images, labels)
                images, labels = images.to('cuda'), labels.to('cuda')
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                fold_true_labels.extend(labels.cpu().numpy())  # Add true labels
                fold_predicted_labels.extend(predicted.cpu().numpy())  # Add predictions

        accuracy = 100 * correct / total
        total_accuracy.append(accuracy)
        print(f'Model accuracy on the test set for Fold {fold + 1}: {accuracy:.2f}%')

        # Calculate and display the confusion matrix for the current fold, ensuring all classes are shown
        cm = confusion_matrix(fold_true_labels, fold_predicted_labels, labels=[np.arange(len(class_names))])
        print(f'Confusion Matrix for Fold {fold + 1}:')
        print_confusion_matrix(cm, class_names)

    total_accuracy_mean = sum(total_accuracy) / len(total_accuracy)
    print(f'Total Accuracy Mean: {total_accuracy_mean:.2f}%')
