import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset
from scripts import train_model, evaluate_model

def kfold(model_factory, fold_split_sequence, num_epochs=50, learning_rate=0.001, batch_size=32, device="cuda"):

    root_dir = "data/spectrograms/cwru_cv/"
    
    transform = transforms.Compose([
        transforms.Resize((224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
    ])

    total_accuracy = []
    for config in fold_split_sequence:
        train_datasets = [ImageFolder(os.path.join(root_dir, f"fold{n_fold}"), transform) for n_fold in config["train"]]
        train_concated_dataset = ConcatDataset(train_datasets)
        
        print(f"Training folds: {list(config['train'])} / Testing folds: {list(config['test'])}")  
        
        # TRAINING
        print('Starting model TRAINING...') 
        train_loader = DataLoader(train_concated_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        # Load the initial weights and biases
        model = model_factory.create_model()

        # Training the model
        model = train_model(model, train_loader, num_epochs, learning_rate, device)

        # EVALUATION
        print('\nStarting model EVALUATION...')      

        for n_test_fold in config["test"]:
            test_dataset_dir = os.path.join(root_dir, f"fold{n_test_fold}")
            
            test_dataset = ImageFolder(test_dataset_dir, transform)                
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

            dataset_class = test_dataset.classes

            # Evaluating the model
            print(f"Evaluating the model on the fold {n_test_fold}.")
            accuracy = evaluate_model(model, test_loader, dataset_class, device)
            total_accuracy.append(accuracy)

    total_accuracy_mean = sum(total_accuracy) / len(total_accuracy)
    print(f'\nTotal Accuracy Mean: {total_accuracy_mean:.2f}%')
    print("")
