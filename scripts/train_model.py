import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

def train_model(
    model,
    train_loader,
    val_loader=None,  # Validation DataLoader is optional
    num_epochs=50,
    learning_rate=0.001,
    batch_size=32,
    device="cuda",
):
    from scripts import EarlyStopping  # Custom early stopping class
    
    model = model.to(device)  # Move the model to the specified device (CPU or GPU)

    # Collects training labels to calculate class weights
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.cpu().numpy())
    
    class_weights = compute_class_weight("balanced", classes=np.unique(all_labels), y=all_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    
    criterion = nn.CrossEntropyLoss() #weight=class_weights)  # Loss function for classification
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=learning_rate, 
        weight_decay=1e-4  # L2 regularization
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
        
    # Dictionaries to store loss and accuracy history for training and validation
    loss_history = {'train': []}
    accuracy_history = {'train': []}

    if val_loader:  # Initialize validation history if a validation loader is provided
        loss_history['val'] = []
        accuracy_history['val'] = []

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=7, delta=0.001, save_path="best_model.pth")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()  # Set the model to training mode
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Iterate through the training data
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # Reset gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update model parameters

            train_loss += loss.item() * images.size(0)  # Accumulate batch loss
            _, predicted = torch.max(outputs, 1)  # Get predicted labels
            train_correct += (predicted == labels).sum().item()  # Count correct predictions
            train_total += labels.size(0)  # Update total samples
        
        # Calculate training loss and accuracy for the epoch
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_accuracy = 100 * train_correct / train_total

        # Store training metrics
        loss_history['train'].append(epoch_train_loss)
        accuracy_history['train'].append(epoch_train_accuracy)

        # Print training metrics
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.2f}%")

        # Validation phase (if validation loader is provided)
        if val_loader:            
            model.eval()  # Set the model to evaluation mode
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():  # Disable gradient computation for validation
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)

                    outputs = model(images)  # Forward pass
                    loss = criterion(outputs, labels)  # Compute loss
                    val_loss += loss.item() * images.size(0)  # Accumulate batch loss
                    _, predicted = torch.max(outputs, 1)  # Get predicted labels
                    val_correct += (predicted == labels).sum().item()  # Count correct predictions
                    val_total += labels.size(0)  # Update total samples

            # Calculate validation loss and accuracy for the epoch
            epoch_val_loss = val_loss / len(val_loader.dataset)
            epoch_val_accuracy = 100 * val_correct / val_total

            # Store validation metrics
            loss_history['val'].append(epoch_val_loss)
            accuracy_history['val'].append(epoch_val_accuracy)

            # Print validation metrics
            print(f"Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_val_accuracy:.2f}%")
            print("-------------------------------")

            scheduler.step(epoch_val_loss)

            # Check for early stopping condition
            if early_stopping(epoch_val_loss, model):
                print("Early stopping triggered. Best model saved at:", early_stopping.save_path)
                break

    # Print final results
    print(f"loss_history={loss_history}")
    print(f"accuracy_history={accuracy_history}")
    print("Training completed.")
    
    return model
