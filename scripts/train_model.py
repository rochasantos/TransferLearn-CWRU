import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_model(model, train_dataset, num_epochs=50, learning_rate=0.001, batch_size=32, device="cuda"):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = model.to(device)
    # ensure that the fc layer is unfreeze
    # for param in model.fc2.parameters():
    #     param.requires_grad = True
    
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    # optimizer = optim.Adam([
    #     {"params": model.fc1.parameters(), 'lr': 0.001},
    #     {"params": model.fc2.parameters(), "lr": 0.001},
    #     {"params": model.conv3.parameters(), "lr": 0.001}
    #     ])

    model.train()

    # Training loop
    loss_history = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        loss_history.append(running_loss/len(train_loader))
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
    
    print(f"loss_history={loss_history}")
    print("Training completed.")
    return model