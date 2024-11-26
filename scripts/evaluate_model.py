import torch
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


def evaluate_model(model, test_loader, dataset_class, device):
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
    print(f'Model accuracy on the test set: {accuracy:.2f}%')

    # Calculate and display the confusion matrix for the current fold, ensuring all classes are shown
    print(f'Confusion Matrix')
    class_indexes = list(range(len(dataset_class)))
    cm = confusion_matrix(fold_true_labels, fold_predicted_labels, labels=class_indexes)
    cls_report = classification_report(fold_true_labels, fold_predicted_labels, digits=4, zero_division=1)
    print(dataset_class)
    print(cm)
    print(cls_report)

    return accuracy