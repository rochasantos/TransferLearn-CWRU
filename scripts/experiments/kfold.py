import os
import numpy as np
from torchvision import datasets, transforms
from src.data_processing.custom_image_dataset import CustomImageDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix
# from src.models.cnn2d import CNN2DFactory
from scripts.experiments.grouper import grouper

def print_confusion_matrix(cm, class_names):
    """Displays the confusion matrix in the console."""
    print("Confusion Matrix:")
    print(f"{'':<7}" + "".join(f"{name:<7}" for name in class_names))
    for i, row in enumerate(cm):
        print(f"{class_names[i]:<7}" + "".join(f"{val:<7}" for val in row))

def kfold(model, file_info, group_by="extent_damage"):
    
    root_dir = 'data/processed/spectrograms'
    model_save_path = 'saved_models/cnn2d.pth'
    num_epochs = 15
    learning_rate = 0.001
    batch_size = 32

    transform = transforms.Compose([
        transforms.Resize((516, 516)),  
        transforms.ToTensor()          
    ])

    dataset = CustomImageDataset(root_dir, file_info, transform)

    X = np.arange(len(dataset))  # get index from spectrograms
    y = dataset.targets  # class tags

    groups = grouper(dataset, group_by)
    
    skf = StratifiedGroupKFold(n_splits=4)

    
    total_accuracy = []
    
    # Class labels for the confusion matrix
    class_names = ['N', 'I', 'O', 'B']  # Your dataset class names
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y, groups)):
        print(f"Fold {fold + 1}")
        
        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4)

        print('Starting model training...')
        
        # Initialize the model, loss function, and optimizer
        
        # Reset model weights for each fold
        model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

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

                running_loss += loss.item()

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

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
        cm = confusion_matrix(fold_true_labels, fold_predicted_labels, labels=np.arange(len(class_names)))
        print(f'Confusion Matrix for Fold {fold + 1}:')
        print_confusion_matrix(cm, class_names)

    total_accuracy_mean = sum(total_accuracy) / len(total_accuracy)
    print(f'Total Accuracy Mean: {total_accuracy_mean:.2f}%')
