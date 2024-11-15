import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset
import os
from sklearn.metrics import confusion_matrix
from scripts.train_model import train_model

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
        # TRAINING
        print('Starting model TRAINING...') 
        print(f"Training folds: {list(config['train'])}")  
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

            correct = 0
            total = 0
            fold_true_labels = []
            fold_predicted_labels = []
           
            # Model evaluation loop
            with torch.no_grad():
                for batch in test_loader:
                    images, labels = batch  # Unpack the tuple (images, labels)
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    fold_true_labels.extend(labels.cpu().numpy())  # Add true labels
                    fold_predicted_labels.extend(predicted.cpu().numpy())  # Add predictions

            accuracy = 100 * correct / total
            total_accuracy.append(accuracy)
            print(f'Model accuracy on the test set for fold {n_test_fold}: {accuracy:.2f}%')

            # Calculate and display the confusion matrix for the current fold, ensuring all classes are shown
            print(f'Confusion Matrix')
            class_names = test_dataset.classes
            class_indexes = list(range(len(class_names)))
            cm = confusion_matrix(fold_true_labels, fold_predicted_labels, labels=class_indexes)
            print(class_names)
            print(cm)
            print("")

    total_accuracy_mean = sum(total_accuracy) / len(total_accuracy)
    print(f'\nTotal Accuracy Mean: {total_accuracy_mean:.2f}%')
    print("")
