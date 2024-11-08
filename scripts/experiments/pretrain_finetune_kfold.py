import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
import numpy as np
import copy
from src.data_processing.dataset import SpectrogramImageDataset
from src.data_processing import DatasetManager
from sklearn.model_selection import StratifiedGroupKFold
from scripts.experiments.helper import grouper
from sklearn.metrics import confusion_matrix

# # Hiperparâmetros
# num_epochs_pretrain = 20
# num_epochs_finetune = 50
# k_folds = 4
# learning_rate_pretrain = 1e-3
# learning_rate_finetune = 1e-4
batch_size = 32

# # Endereços para arquivos
# root_dir = 'data/spectrograms'

# filter_uored = {"UORED": {"label": ["N", "I", "O", "B"], "condition_bearing_health": ["faulty" , "healthy"]}}
# filter_cwru = {"CWRU": {"label": ["N", "I", "O", "B"], "bearing_type": "6205"}}
# file_info_uored = DatasetManager().filter_data(filter_uored)
# file_info_cwru = DatasetManager().filter_data(filter_cwru)

# # Transforms para imagens
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),
# ])

# # Carregar os datasets
class_names = ['N', 'I', 'O', 'B']  # Your dataset class names
# dataset_uored = SpectrogramImageDataset(root_dir, file_info_uored, class_names, transform)

# # DataLoaders para pré-treinamento
# dataloader_uored = DataLoader(dataset_uored, batch_size=batch_size, shuffle=True)

# Função para treinar uma época
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

# Pré-treinamento das camadas intermediárias
def pretrain_model(model, dataloader1, dataloader2, num_epochs, learning_rate, device):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Congelar primeiras camadas
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # Alternar entre os dois datasets
    for epoch in range(num_epochs):
        loss1 = train_epoch(model, dataloader1, criterion, optimizer, device)
        loss2 = train_epoch(model, dataloader2, criterion, optimizer, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss Dataset 1: {loss1:.4f}, Loss Dataset 2: {loss2:.4f}')

def print_confusion_matrix(cm, class_names):
    """Displays the confusion matrix in the console."""
    print("Confusion Matrix:")
    print(f"{'':<3}" + "".join(f"{name:<5}" for name in class_names))
    for i, row in enumerate(cm):
        print(f"{class_names[i]:<3}" + "".join(f"{val:<5}" for val in row))

# k-Fold cross-validation para fine-tuning
def k_fold_finetune(model, dataset, num_folds, num_epochs, learning_rate, device, 
                    pretrained_model_path=None, group_by="extent_damage"):
    criterion = nn.CrossEntropyLoss()

    X = np.arange(len(dataset))  # get index from spectrograms
    y = dataset.targets  # class tags
    groups = grouper(dataset, group_by)
    initial_state = copy.deepcopy(model.state_dict())
    total_accuracy = []
    skf = StratifiedGroupKFold(n_splits=num_folds)
    # Loop sobre os folds
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y, groups)):
        print(f"Fold {fold + 1}")
        
        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # Recarregar os pesos pré-treinados
        if pretrained_model_path:
            model.load_state_dict(torch.load(pretrained_model_path))
        else:
            model.load_state_dict(initial_state)
        
        # Descongelar as últimas camadas para fine-tuning
        for name, param in model.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Definir otimizador
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
        
        # Treinamento para o fold atual
        for epoch in range(num_epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            print(f'Fold {fold+1}, Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}')
            
        # Avaliação do fold atual
        model.eval()
        val_loss, correct = 0.0, 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(test_loader)
        accuracy = 100 * correct / len(test_subset)
        total_accuracy.append(accuracy)
        print(f'Fold {fold+1} Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%\n')

        # Calculate and display the confusion matrix for the current fold, ensuring all classes are shown
        cm = confusion_matrix(all_labels, all_preds, labels=[np.arange(len(class_names))])
        print(f'Confusion Matrix for Fold {fold + 1}:')
        print_confusion_matrix(cm, class_names)
    
    print(f'Total Accuracy Mean: {sum(total_accuracy)/len(total_accuracy):.2f}%\n')
