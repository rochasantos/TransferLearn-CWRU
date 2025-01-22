import os
import numpy as np
import copy
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.model_selection import train_test_split
from scripts import train_model, evaluate_model

def mitigated_kfold(model, fold_split_sequence, num_epochs=50, learning_rate=0.001, batch_size=32, device="cuda", repetition=1, config='extent_damage'):

    root_dir = os.path.join("data/spectrograms", "cwru_"+config)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
    ])

    # Load the initial weights and biases
    initial_state = copy.deepcopy(model.state_dict())        
    
    total_accuracy = []
    mean_accuracy_by_fold = {"fold1": [], "fold2": [], "fold3": [], "fold4": []}
    for i in range(repetition):
        print(f"\n--------------")
        print(f"Repetition: {i+1}")
        print(f"--------------")
        accuracies = []
        for seq in fold_split_sequence:
            list_train_datasets = [ImageFolder(os.path.join(root_dir, f"fold{n_fold}"), transform) for n_fold in seq["train"]]
            train_dataset = ConcatDataset(list_train_datasets)
            
            if "val" in seq:
                val_dataset = ImageFolder(os.path.join(root_dir, f"fold{seq['val'][0]}"), transform)
                val_loader = DataLoader(val_dataset, batch_size, shuffle=True, num_workers=4)
            else:
                targets = [label for dataset in list_train_datasets for _, label in dataset.samples]
                train_indices, val_indices = train_test_split(
                        list(range(len(train_dataset))),
                        test_size=0.20,
                        stratify=targets,
                        random_state=42
                    )
                train_dataset = Subset(train_dataset, train_indices)
                val_subset = Subset(train_dataset, val_indices)
                val_loader = DataLoader(val_subset, batch_size, shuffle=True, num_workers=4)
            
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
            # TRAINING
            print('Starting model TRAINING...')                
            print(f"Training folds: {list(seq['train'])}")  
            if "val" in seq:
                print(f"Validation folds: {list(seq['val'])}")

            # Load the initial state of the model   
            model.load_state_dict(initial_state)
            
            # Training the model
            model = train_model(model, train_loader, val_loader, num_epochs, learning_rate, device, save_path='best_model.pth', save_model=True)
            
            model = model.to(device)

            # EVALUATION
            print('\nStarting model EVALUATION...')      

            for n_test_fold in seq["test"]:
                test_dataset_dir = os.path.join(root_dir, f"fold{n_test_fold}")                        
                test_dataset = ImageFolder(test_dataset_dir, transform)                
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

                # Evaluating the model
                print(f"Evaluating the model on the fold {n_test_fold}.")                
                accuracy = evaluate_model(model, test_loader, train_dataset.datasets[0].classes, device)
                mean_accuracy_by_fold[f"fold{n_test_fold}"].append(accuracy)
                accuracies.append(accuracy)

        mean_accuracy = np.mean(accuracies)
        print(f'Mean Accuracy: {np.round(mean_accuracy, 2)}')
        total_accuracy.append(mean_accuracy)
    
    total_mean_accuracy = np.mean(total_accuracy)
    std = np.std(total_accuracy)
    acc_f1 = mean_accuracy_by_fold["fold1"]
    acc_f2 = mean_accuracy_by_fold["fold2"]
    acc_f3 = mean_accuracy_by_fold["fold3"]
    acc_f4 = mean_accuracy_by_fold["fold4"]
    m_acc_f1, std_f1 = np.round(np.mean(acc_f1), decimals=2), np.round(np.std(acc_f1), decimals=2)
    m_acc_f2, std_f2 = np.round(np.mean(acc_f2), decimals=2), np.round(np.std(acc_f2), decimals=2)
    m_acc_f3, std_f3 = np.round(np.mean(acc_f3), decimals=2), np.round(np.std(acc_f3), decimals=2)
    m_acc_f4, std_f4 = np.round(np.mean(acc_f4), decimals=2), np.round(np.std(acc_f4), decimals=2)
    print(f'\nMean Accuracy of folder: \nfold1: {m_acc_f1} +- {std_f1} \nfold2: {m_acc_f2} +- {std_f2} \nfold3: {m_acc_f3} +- {std_f3} \nfold4: {m_acc_f4} +- {std_f4}')
    print(f'\nTotal Mean Accuracy: {np.round(total_mean_accuracy, 2)}, Std: {np.round(std, 4)}')
    print("")

