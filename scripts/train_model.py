import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.models import ResNet18, CNN2D
from scripts.early_stopping import EarlyStopping
from sklearn.metrics import f1_score

def train_model(
    model,
    train_loader,
    val_loader,
    epochs=50,
    learning_rate=0.001,
    device="cuda",
    save_path='best_model.pth',
    save_model=True,
):
    # Loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=learning_rate, weight_decay=1e-4)  # Add weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    model.to(device)
    early_stopping = EarlyStopping(patience=5, delta=0.01, save_path=save_path, save_model=save_model)
    
    best_epoch = 0  # Variável para armazenar a melhor época
    best_val_loss = np.inf  # Melhor loss de validação inicializada com infinito

    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_accuracy = 100 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_accuracy = 100 * np.sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)

        # Save the best epoch
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch + 1  # Armazena a melhor época (começando de 1)

        # Scheduler step
        scheduler.step(epoch_val_loss)

        # Print results
        print(f"Epoch [{epoch + 1}/{epochs}]")
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.2f}%")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_val_accuracy:.2f}%")
        print("-------------------------------")

        # Early stopping
        if early_stopping(epoch_val_loss, model):
            print("Early stopping triggered. Restoring best model.")
            break

    # Load the best model weights
    model.load_state_dict(torch.load(early_stopping.save_path))

    print(f"The best model was saved at {early_stopping.save_path}")
    print(f"Best epoch: {best_epoch}, Best Validation Loss: {best_val_loss:.4f}")
    return model
