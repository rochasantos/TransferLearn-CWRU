import os
import numpy as np

from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.data import DataLoader

from sklearn.model_selection import StratifiedGroupKFold

from src.models.cnn2d import CNN2DFactory

from src.data_processing.annotation_file import AnnotationFileHandler
from scripts.experiments.helper import grouper

def kfold():
    
    transform = transforms.Compose([
        transforms.Resize((516, 516)),  
        transforms.ToTensor()          
    ])
    dataset = datasets.ImageFolder(root='data/processed/spectrograms', transform=transform)

    X = np.arange(len(dataset))  # get index from spectrograms
    y = dataset.targets  # class tags

    groups = grouper(dataset, "extent_damage")
    
    skf = StratifiedGroupKFold(n_splits=3)

    model_save_path = 'saved_models/cnn2d.pth'
    num_epochs = 15
    learning_rate = 0.001
    batch_size=32

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y, groups)):
        print(f"Fold {fold + 1}")
        
        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4)

        print('Starting model training...')
        
        # Initialize the model, loss function, and optimizer
        model = CNN2DFactory.create().to('cuda')
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(num_epochs):
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to('cuda'), labels.to('cuda')

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

        # Save the trained model
        # torch.save(model.state_dict(), model_save_path)
        # print(f'Model trained and saved as {model_save_path}')

        print('Training completed. Starting model evaluation...')
        correct = 0
        total = 0

        # Model evaluation loop
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to('cuda'), labels.to('cuda')
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Model accuracy on the test set: {accuracy:.2f}%')
