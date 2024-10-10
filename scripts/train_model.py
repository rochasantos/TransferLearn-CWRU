import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data_processing.dataset import SpectrogramDataset
from src.estimators.cnn2d import CNN2D

def train_model(dataset_path, model_save_path='trained_model.pth', num_epochs=10, batch_size=32, learning_rate=0.001):
    """ Train a CNN model using spectrogram dataset and save the trained model.

    Parameters
    ----------
    dataset_path : str
        Path to the directory containing the spectrogram dataset for training.
    model_save_path : str, optional
        Path to save the trained model file (default is 'trained_model.pth').
    num_epochs : int, optional
        Number of epochs for training (default is 10).
    batch_size : int, optional
        Size of each mini-batch for training (default is 32).
    learning_rate : float, optional
        Learning rate for the optimizer (default is 0.001).

    Returns
    -------
    None

    Description
    -----------
    Initializes the dataset and dataloader, sets up the CNN model, loss function, and optimizer.
    Performs the training loop over the specified number of epochs, calculates the loss, 
    and updates the model weights. Finally, saves the trained model to the specified path.
    """

    # Initialize the dataset and dataloader
    dataset = SpectrogramDataset(root=dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize the model, loss function, and optimizer
    model = CNN2D().to('cuda')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')

    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print(f'Model trained and saved as {model_save_path}')


