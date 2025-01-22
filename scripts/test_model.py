from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from scripts import evaluate_model

def test_model(model, dataset_dir):
    # EVALUATION
    print('\nStarting model EVALUATION...')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
    ]) 

    test_dataset = ImageFolder(dataset_dir, transform)                
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4)

    # Evaluating the model
    accuracy = evaluate_model(model, test_loader, ['B', 'I', 'N', 'O'], 'cuda')
    print(f"accuracy: {accuracy}")