import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedGroupKFold, StratifiedKFold
from scripts.experiments.helper import grouper 

def print_confusion_matrix(cm, class_names):
    """Displays the confusion matrix in the console."""
    print("Confusion Matrix:")
    print(f"{'':<5}" + "".join(f"{name:<5}" for name in class_names))
    for i, row in enumerate(cm):
        print(f"{class_names[i]:<3}" + "".join(f"{val:<5}" for val in row))

def resubstitution_test(model, dataset, num_epochs, lr, class_names):
    # Set up data loader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Move model to GPU if available
    model = model.to('cuda')
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)
    
    # Training loop
    print('Starting Resubstitution Test Training...')
    print(dataset.get_dataset_name())
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Average loss for the epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}')

    # Evaluation phase
    print('Resubstitution Evaluation...')
    model.eval()
    all_labels, all_predictions = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    # Calculate accuracy and confusion matrix
    accuracy = accuracy_score(all_labels, all_predictions) * 100
    cm = confusion_matrix(all_labels, all_predictions)
    print(f'Resubstitution Accuracy: {accuracy:.2f}%')
    #print('Confusion Matrix:\n', cm)
    print(dataset.get_dataset_name())
    print_confusion_matrix(cm, class_names)

def one_fold_with_bias(model, dataset, num_epochs, lr, class_names):
    # Split the data with bias (random train-test split)
    train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2, stratify=dataset.targets, random_state=42)
    
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=32, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=32, shuffle=False)
    
    # Move model to GPU if available
    model = model.to('cuda')
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)
    
    # Training loop
    print('Starting One-Fold (With Bias) Training...')
    print(dataset.get_dataset_name())
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}')
    print('Training completed. Starting evaluation...')
    
    # Evaluation
    model.eval()
    all_labels, all_predictions = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    # Calculate accuracy and confusion matrix
    accuracy = accuracy_score(all_labels, all_predictions) * 100
    cm = confusion_matrix(all_labels, all_predictions)
    print(f'One-Fold (With Bias) Accuracy: {accuracy:.2f}%')
    #print('Confusion Matrix:\n', cm)
    print(dataset.get_dataset_name())
    print_confusion_matrix(cm, class_names)

def one_fold_without_bias(model, dataset, num_epochs, lr, class_names):
    # Stratified split to reduce bias
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    X = np.arange(len(dataset))
    y = dataset.targets
    for train_idx, test_idx in sss.split(X, y):
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=32, shuffle=True)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=32, shuffle=False)
    
    # Move model to GPU if available
    model = model.to('cuda')
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)
    
    # Training loop
    print('Starting One-Fold (Without Bias) Training...')
    print(dataset.get_dataset_name())
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}')
    print('Training completed. Starting evaluation...')
    
    # Evaluation
    model.eval()
    all_labels, all_predictions = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    # Calculate accuracy and confusion matrix
    accuracy = accuracy_score(all_labels, all_predictions) * 100
    cm = confusion_matrix(all_labels, all_predictions)
    print(f'One-Fold (Without Bias) Accuracy: {accuracy:.2f}%')
    #print('Confusion Matrix:\n', cm) 
    print(dataset.get_dataset_name())
    print_confusion_matrix(cm, class_names)
    
def kfold_cross_validation(model, dataset, num_epochs, lr, group_by="", class_names=[], n_splits=4):
    """Performs K-Fold Cross-Validation for ViT models with optional grouping on the provided dataset."""
    batch_size = 32
    X = np.arange(len(dataset))
    y = dataset.targets  # Class labels for stratification

    # If `group_by` is empty, use regular StratifiedKFold, otherwise use StratifiedGroupKFold
    if group_by:
        groups = grouper(dataset, group_by)
        skf = StratifiedGroupKFold(n_splits=n_splits)
        split = skf.split(X, y, groups)
    else:
        print('Group by: none')
        skf = StratifiedKFold(n_splits=n_splits)
        split = skf.split(X, y)

    # Save initial model state for reinitialization before each fold
    #initial_state = model.state_dict()
    initial_state = copy.deepcopy(model.state_dict())
    fold_accuracies = []

    print('LR: ', lr)
    print('Starting K-Fold Cross-Validation Loop...')   
    # K-Fold Cross-Validation Loop
    for fold, (train_idx, test_idx) in enumerate(split):
        print(f"\nStarting Fold {fold + 1}")

        # Skip this fold if the test split is empty
        if len(test_idx) == 0 or len(train_idx) == 0:
            print(f"Skipping Fold {fold + 1} as the test set is empty.")
            continue

        # Prepare train and test loaders for the current fold
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)

        # Reset model to initial state and move to GPU
        print("Initial weight sample before reset:",  model.vit.classifier.weight[0][:5])  # Example layer and slice
        model.load_state_dict(copy.deepcopy(initial_state))
        print("Initial weight sample after reset:",  model.vit.classifier.weight[0][:5])
        
        # Reinitialize the optimizer for each fold
  
        # Layer-Wise Learning Rate Decay (LRD)
        # Layers closer to the output layer receive a higher learning rate than those closer to the input.
        # This can help preserve the pre-trained representations.
        optimizer = AdamW(
            [
                {"params": model.vit.vit.parameters(), "lr": lr * 0.1},  # Transformer encoder layers
                {"params": model.vit.classifier.parameters(), "lr": lr}  # Classification head
            ],
            lr=lr,
            weight_decay=0.01
        )
        
        # Define loss
        criterion = nn.CrossEntropyLoss()

        # Training Loop
        model.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to('cuda'), labels.to('cuda')
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Fold {fold + 1} | Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

        # Evaluation Loop
        print("Evaluating Fold...")

        model.eval()
        all_labels, all_predictions = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to('cuda'), labels.to('cuda')
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        # Check if there are predictions to evaluate
        if all_labels:
            accuracy = accuracy_score(all_labels, all_predictions) * 100
            fold_accuracies.append(accuracy)
            cm = confusion_matrix(all_labels, all_predictions)
            print(f"Fold {fold + 1} Accuracy: {accuracy:.2f}%")
            #print("Confusion Matrix:\n", cm)
            print(dataset.get_dataset_name())
            print_confusion_matrix(cm, class_names)
        else:
            print(f"No test data for evaluation in Fold {fold + 1}. Accuracy cannot be computed.")

    # Summary of cross-validation results
    if fold_accuracies:
        mean_accuracy = np.mean(fold_accuracies)
        print(dataset.get_dataset_name())
        print(f"\nMean Cross-Validation Accuracy: {mean_accuracy:.2f}%")
    else:
        print("No valid folds with test data to compute cross-validation accuracy.")