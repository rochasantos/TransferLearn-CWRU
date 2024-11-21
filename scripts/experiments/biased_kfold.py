import os
import copy
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from scripts import train_model, evaluate_model

def biesed_kfold(model, num_epochs=50, learning_rate=0.001, batch_size=32, device="cuda", repetition=5):

    root_dir = "data/spectrograms/"
    num_folds = 4
    total_accuracies = [] 
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
    ])

    # Load the initial weights and biases
    initial_state = copy.deepcopy(model.state_dict())        
    
    dataset = ImageFolder(os.path.join(root_dir, "cwru"), transform)
    labels = np.array([sample[1] for sample in dataset.samples])
    
    for i in range(repetition):
        print(f"\nRepetition: {i}")
        accuracies = []
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            print(f"Fold {fold + 1}/{num_folds}")        
            
            train_subset = Subset(dataset, train_idx)
            test_subset = Subset(dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size, shuffle=True, num_workers=4)
            test_loader = DataLoader(test_subset, batch_size, shuffle=True, num_workers=4)
            

            # TRAINING
            print('Starting model TRAINING...') 
            # Load the initial state of the model   
            model.load_state_dict(initial_state)
            # Training the model
            model = train_model(model, train_loader, num_epochs, learning_rate, device)
            
            model = model.to(device)

            # EVALUATION
            dataset_class = dataset.classes
            print(f"\nEvaluating the model on the fold {fold + 1}.")
            accuracy = evaluate_model(model, test_loader, dataset_class, device)
            
            accuracies.append(accuracy)

        mean_accuracy = sum(accuracies) / len(accuracies)
        total_accuracies.append(mean_accuracy)
        print(f'\nMean Accuracy: {mean_accuracy:.2f}%')

    print(f"Total Mean Accuracy: {np.mean(total_accuracies):.4f}, Std: {np.std(total_accuracies):.4f}")
