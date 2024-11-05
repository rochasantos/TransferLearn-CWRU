import torch
from torch.utils.data import DataLoader
from src.data_processing import SpectrogramImageDataset
from src.models.cnn2d import CNN2D
import torch.nn.functional as F

def evaluate_model(dataset_path, model_path='saved_models/trained_model.pth'):
    """ Evaluate a trained CNN model using spectrogram dataset.

    Parameters
    ----------
    dataset_path : str
        Path to the directory containing the spectrogram dataset for evaluation.
    model_path : str, optional
        Path to the trained model file (default is 'trained_model.pth').

    Returns
    -------
    None

    Description
    -----------
    Initializes the dataset and dataloader, loads the trained model, and performs 
    evaluation on the test data. It calculates and prints the model's accuracy on 
    the provided dataset.
    """
    # Initialize the test dataset and dataloader
    dataset = SpectrogramImageDataset(root=dataset_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # Load the trained model
    model = CNN2D().to('cuda')
    model.load_state_dict(torch.load(model_path))
    model.eval()

    correct = 0
    total = 0

    # Model evaluation loop
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Model accuracy on the test set: {accuracy:.2f}%')
