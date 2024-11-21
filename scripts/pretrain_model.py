import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

def pretrain_model(model, datasets_name, epochs=10, batch_size=32, learning_rate=1e-3, device='cuda', save_path='saved_models/pretrained_model.pth'):
    
    if os.path.exists(save_path):
        print(f"There is already an estimator on path '{save_path}'")
        model.load_state_dict(torch.load(save_path))
        return model

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
    ])
    
    root_dir = "data/spectrograms"
    train_datasets = [ImageFolder(os.path.join(root_dir, ds.lower()), transform) for ds in datasets_name]
    train_concated_dataset = ConcatDataset(train_datasets)
    dataloader = DataLoader(train_concated_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Freeze layers
    """
    for name, param in model.named_parameters():
        if "layer4" in name in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    """

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Setting up the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)

    # Training loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader):.4f}")

    # Saving the pretrained model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")
    
    return model