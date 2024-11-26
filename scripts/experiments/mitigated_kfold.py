import os
import numpy as np
import copy
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset
from scripts import train_model, evaluate_model

def mitigated_kfold(model, fold_split_sequence, num_epochs=50, learning_rate=0.001, batch_size=32, device="cuda", repetition=1):

    root_dir = "data/spectrograms/cwru_cv/"
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
    ])

    # Load the initial weights and biases
    initial_state = copy.deepcopy(model.state_dict())        
    
    total_accuracy = []
    for i in range(repetition):
        print(f"Repetition: {i}")
        print(f"--------------")
        accuracies = []
        for config in fold_split_sequence:
            if config["train"]:
                train_datasets = [ImageFolder(os.path.join(root_dir, f"fold{n_fold}"), transform) for n_fold in config["train"]]
                train_dataset = ConcatDataset(train_datasets)
                
                
                # TRAINING
                print('Starting model TRAINING...')                
                print(f"Training folds: {list(config['train'])}")  

                # Load the initial state of the model   
                model.load_state_dict(initial_state)
                # Training the model
                model = train_model(model, train_dataset, num_epochs, learning_rate, batch_size, device)
            
            model = model.to(device)

            # EVALUATION
            print('\nStarting model EVALUATION...')      

            for n_test_fold in config["test"]:
                test_dataset_dir = os.path.join(root_dir, f"fold{n_test_fold}")                        
                test_dataset = ImageFolder(test_dataset_dir, transform)                
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

                # Evaluating the model
                print(f"Evaluating the model on the fold {n_test_fold}.")
                accuracy = evaluate_model(model, test_loader, train_datasets[0].classes, device)
                accuracies.append(accuracy)

        mean_accuracy = np.mean(accuracies)
        print(f'Mean Accuracy: {np.round(mean_accuracy, 2)}')
        total_accuracy.append(mean_accuracy)
    
    total_mean_accuracy = np.mean(total_accuracy)
    std = np.std(total_accuracy)
    print(f'\nTotal Mean Accuracy: {np.round(total_mean_accuracy, 2)}, Std: {np.round(std, 4)}')
    print("")
