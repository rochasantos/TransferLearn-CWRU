import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_model(model, train_dataset, num_epochs=50, learning_rate=0.001, batch_size=32, device="cuda"):
    from scripts import EarlyStopping
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = model.to(device)  
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    
    model.train()

    # Training loop
    loss_history = []
    accuracy_history = []  # List to store accuracy history
    early_stopping = EarlyStopping(patience=5, delta=0.01, save_path="best_model.pth", no_save_model=True)
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = 100 * correct_predictions / total_samples  # Calculate percentage

        loss_history.append(epoch_loss)
        accuracy_history.append(epoch_accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')
        early_stopping(epoch_loss, model)
    
        if early_stopping.early_stop:
            print("Treinamento interrompido por convergência.")
            break

    # print(f"loss_history={loss_history}")
    # print(f"accuracy_history={accuracy_history}")
    print("Training completed.")

    return model
