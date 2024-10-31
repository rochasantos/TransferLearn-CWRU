import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def pretrain_model(model, dataset, epochs=10, batch_size=32, learning_rate=1e-3, device='cuda', save_path='saved_models/pretrained_model.pth'):
    """
    Function to pretrain a PyTorch model and save it.
    
    Parameters:
    - model: PyTorch model instance.
    - dataset: PyTorch dataset for pretraining.
    - epochs: number of epochs for pretraining.
    - batch_size: batch size.
    - learning_rate: learning rate for the optimizer.
    - device: device ('cuda' or 'cpu') for training.
    - save_path: path to save the pretrained model.
    
    Returns:
    - model: trained model.
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Setting up DataLoader for the pretraining dataset
    pretrain_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Setting up the loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Use an appropriate loss function for your problem
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)

    # Training loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0

        for images, labels in pretrain_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(pretrain_loader):.4f}")

    # Saving the pretrained model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")
    
    return model