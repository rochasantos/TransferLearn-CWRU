import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit 

def resubstitution_test(model, dataset, num_epochs, lr):
    # Set up data loader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Move model to GPU if available
    model = model.to('cuda')
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)
    
    # Training loop
    print('Starting Resubstitution Test Training...')
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
    print('Confusion Matrix:\n', cm)
    


def one_fold_with_bias(model, dataset, num_epochs, lr):
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
    print('Confusion Matrix:\n', cm)

def one_fold_without_bias(model, dataset, num_epochs, lr):
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
    print('Confusion Matrix:\n', cm) 