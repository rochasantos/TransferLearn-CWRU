import numpy as np
import copy
import torch
import cv2
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import sys
import os
from src.data_processing import SpectrogramImageDataset
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, Subset, ConcatDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedGroupKFold, StratifiedKFold
from scripts.experiments.helper import grouper, grouper_distribution 

def print_confusion_matrix(cm, class_names, all_labels, all_predictions):
    """Displays the confusion matrix in the console."""
    # Ensure alignment between labels and predictions
    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=np.arange(len(class_names))  # Explicitly use all expected class labels
    )
    print("Confusion Matrix:")
    print("     " + "    ".join(class_names))
    for i, row in enumerate(cm):
        print(f"{class_names[i]:<4}" + "  ".join(f"{val:4}" for val in row))
    
def resubstitution_test(model, dataset, num_epochs, lr, class_names):
    # Set up data loader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Move model to GPU if available
    model = model.to('cuda')
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)
    
    # Training loop
    print('Starting Resubstitution Test Training...')
    #print(dataset.get_dataset_name())
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            logits, attentions = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Average loss for the epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}')

    # Evaluation phase
    print('Resubstitution Evaluation...')
    model.eval()
    all_labels, all_predictions = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to('cuda'), labels.to('cuda')
            logits, attentions = model(images)
            _, predicted = torch.max(logits, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    # Calculate accuracy and confusion matrix
    accuracy = accuracy_score(all_labels, all_predictions) * 100
    cm = confusion_matrix(all_labels, all_predictions)
    print(f'Resubstitution Accuracy: {accuracy:.2f}%')
    #print('Confusion Matrix:\n', cm)
    #print(dataset.get_dataset_name())
    print_confusion_matrix(cm, class_names, all_labels, all_predictions)

def one_fold_with_bias(model, dataset, num_epochs, lr, class_names):
    # Split the data with bias (random train-test split)
    train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2, stratify=dataset.targets, random_state=42)
    
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=32, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=32, shuffle=False)
    
    # Move model to GPU if available
    model = model.to('cuda')
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)
    
    # Training loop
    print('Starting One-Fold (With Bias) Training...')
    #print(dataset.get_dataset_name())
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            logits, attentions = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}')
    print('Training completed. Starting evaluation...')
    
    # Evaluation
    model.eval()
    all_labels, all_predictions = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            logits, attentions = model(images)
            _, predicted = torch.max(logits, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    # Calculate accuracy and confusion matrix
    accuracy = accuracy_score(all_labels, all_predictions) * 100
    cm = confusion_matrix(all_labels, all_predictions)
    print(f'One-Fold (With Bias) Accuracy: {accuracy:.2f}%')
    #print('Confusion Matrix:\n', cm)
    #print(dataset.get_dataset_name())
    print_confusion_matrix(cm, class_names, all_labels, all_predictions)

def one_fold_without_bias(model, dataset, num_epochs, lr, class_names):
    # Stratified split to reduce bias
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    X = np.arange(len(dataset))
    y = dataset.targets
    for train_idx, test_idx in sss.split(X, y):
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=32, shuffle=True)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=32, shuffle=False)
    
    # Move model to GPU if available
    model = model.to('cuda')
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)
    
    # Training loop
    print('Starting One-Fold (Without Bias) Training...')
#    print(dataset.get_dataset_name())
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            logits, attentions = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}')
    print('Training completed. Starting evaluation...')
    
    # Evaluation
    model.eval()
    all_labels, all_predictions = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            logits, attentions = model(images)
            _, predicted = torch.max(logits, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    # Calculate accuracy and confusion matrix
    accuracy = accuracy_score(all_labels, all_predictions) * 100
    cm = confusion_matrix(all_labels, all_predictions)
    print(f'One-Fold (Without Bias) Accuracy: {accuracy:.2f}%')
    #print('Confusion Matrix:\n', cm) 
    #print(dataset.get_dataset_name())
    print_confusion_matrix(cm, class_names, all_labels, all_predictions)
    
def kfold_cross_validation(model, test_loader, num_epochs, lr, group_by="", class_names=[], n_splits=4):
    """Performs K-Fold Cross-Validation for ViT models with optional grouping on the provided dataset."""
    batch_size = 32
    dataset = test_loader.dataset
    
    # Extract targets directly from the DataLoader
    y = [label for _, label in dataset]

    X = np.arange(len(y))

    # If `group_by` is empty, use regular StratifiedKFold, otherwise use StratifiedGroupKFold
    if group_by:
        groups = grouper(dataset, group_by)
        skf = StratifiedGroupKFold(n_splits=n_splits)
        split = skf.split(X, y, groups)
    else:
        print('Group by: none')
        skf = StratifiedKFold(n_splits=n_splits)
        split = skf.split(X, y)

    # Save initial model state for reinitialization before each fold
    initial_state = copy.deepcopy(model.state_dict())
    fold_metrics = []  # Store metrics for all folds

    print('LR: ', lr)
    print('Starting K-Fold Cross-Validation Loop...')   
    # K-Fold Cross-Validation Loop
    for fold, (train_idx, test_idx) in enumerate(split):
        print(f"\nStarting Fold {fold + 1}")

        # Skip this fold if the test split is empty
        if len(test_idx) == 0 or len(train_idx) == 0:
            print(f"Skipping Fold {fold + 1} as the test set is empty.")
            continue

        # print(f">> Train distribution: -- using grouper logic")    
        # train_distribution = grouper_distribution(dataset, group_by, train_idx, class_names)
        # print(f">> Test distribution: -- using grouper logic") 
        # test_distribution = grouper_distribution(dataset, group_by, test_idx, class_names)

        # print(f"Fold {fold + 1} - Train Distribution: {train_distribution}")
        # print(f"Fold {fold + 1} - Test Distribution: {test_distribution}")

        # Prepare train and test loaders for the current fold
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        #train_loader = create_balanced_dataloader(Subset(dataset, train_idx), batch_size=batch_size)
        
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)

        # Alternatively, analyze distributions from DataLoaders
        train_labels = []
        for _, labels in train_loader:
            train_labels.extend(labels.cpu().numpy())

        train_distribution_from_loader = {class_name: train_labels.count(i) for i, class_name in enumerate(class_names)}
        print(f"Fold {fold + 1} - [debug] Train Distribution from Loader: {train_distribution_from_loader}")

        test_labels = []
        for _, labels in test_loader:
            test_labels.extend(labels.cpu().numpy())

        test_distribution_from_loader = {class_name: test_labels.count(i) for i, class_name in enumerate(class_names)}
        print(f"Fold {fold + 1} - [debug] Test Distribution from Loader: {test_distribution_from_loader}")

        # Reset model to initial state and move to GPU
        #debug print("Initial weight sample before reset:",  model.vit.classifier.weight[0][:5])  # Example layer and slice
        model.load_state_dict(copy.deepcopy(initial_state))
        #debug print("Initial weight sample after reset:",  model.vit.classifier.weight[0][:5])
        
        # Reinitialize the optimizer for each fold
  
        # Layer-Wise Learning Rate Decay (LRD)
        # Layers closer to the output layer receive a higher learning rate than those closer to the input.
        # This can help preserve the pre-trained representations.
        optimizer = AdamW(
            [
                {"params": model.vit.vit.parameters(), "lr": lr * 0.1},  # Transformer encoder layers
                {"params": model.vit.classifier.parameters(), "lr": lr}  # Classification head
            ],
            lr=lr,
            weight_decay=0.01
        )
        
        # Define loss
        criterion = nn.CrossEntropyLoss()

        # Training Loop
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to('cuda'), labels.to('cuda')
                optimizer.zero_grad()
                logits, attentions = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Fold {fold + 1} | Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

        # Evaluation Loop
        print("Evaluating Fold...")

        model.eval()
        all_labels, all_predictions = [], []
        # Track classes already visualized
        visualized_classes = set()

        # Create a dictionary to store one index per class for visualization
        class_sample_indices = {class_name: None for class_name in range(len(class_names))}

        with torch.no_grad():
            for idx, (images, labels) in enumerate(test_loader):
                images, labels = images.to('cuda'), labels.to('cuda')
                logits, attentions = model(images) 
                # Visualize attention for the first 5 samples
                # if idx < 2:
                #     visualize_attention(
                #         dataset=dataset,
                #         model=model,
                #         idx=idx,
                #         attentions=attentions,
                #         head=0,  # Visualize first attention head
                #         layer=-1  # Visualize last attention layer
                #     )
                
                # Store one index per class for visualization
                for i, label in enumerate(labels.cpu().numpy()):
                    if class_sample_indices[label] is None:
                        class_sample_indices[label] = idx * len(labels) + i  # Global index in the dataset
                               
                _, predicted = torch.max(logits, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        # Visualize attention for one sample per class
        for class_id, sample_idx in class_sample_indices.items():
            if sample_idx is not None:
                print(f"Visualizing attention for class '{class_names[class_id]}' (index {sample_idx})...")
                visualize_attention(
                    dataset=dataset,
                    model=model,
                    idx=sample_idx,
                    attentions=None,  # Let visualize_attention compute attentions
                    head=0,  # Visualize first attention head
                    layer=-1  # Visualize last attention layer
                )

        # Check if there are predictions to evaluate
        if all_labels:
            accuracy = accuracy_score(all_labels, all_predictions) * 100
            precision = precision_score(all_labels, all_predictions, average='weighted') * 100
            recall = recall_score(all_labels, all_predictions, average='weighted') * 100
            f1 = f1_score(all_labels, all_predictions, average='weighted') * 100

            cm = confusion_matrix(all_labels, all_predictions, labels=np.arange(len(class_names)))
            
            print(f"Fold {fold + 1} Classification Report:")
            report = classification_report(all_labels, 
                                           all_predictions, 
                                           target_names=class_names, 
                                           labels=np.arange(len(class_names)), 
                                           digits=4, 
                                           zero_division=0)
            print(report)
            
            print(f"Fold {fold + 1} Metrics:")
            print(f"  - Accuracy: {accuracy:.2f}%")
            print(f"  - Precision: {precision:.2f}%")
            print(f"  - Recall: {recall:.2f}%")
            print(f"  - F1-Score: {f1:.2f}%")
                        
            fold_metrics.append({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            })
            
            if isinstance(dataset, ConcatDataset):
                root_dirs = [ds.root for ds in dataset.datasets if hasattr(ds, 'root')]
                print("\nDataset Roots:", root_dirs)
            else:
                if hasattr(dataset, 'root'):
                    print("\nDataset Root:", dataset.root)
                else:
                    print("\nDataset information not available.")            
            
            print_confusion_matrix(cm, class_names, all_labels, all_predictions)
        else:
            print(f"No test data for evaluation in Fold {fold + 1}. Accuracy cannot be computed.")

    # Summary of cross-validation results
    if fold_metrics:
        mean_metrics = {key: np.mean([fold[key] for fold in fold_metrics]) for key in fold_metrics[0]}
        print("\nCross-Validation Metrics Summary:")
        print(f"  - Mean Accuracy: {mean_metrics['accuracy']:.2f}%")
        print(f"  - Mean Precision: {mean_metrics['precision']:.2f}%")
        print(f"  - Mean Recall: {mean_metrics['recall']:.2f}%")
        print(f"  - Mean F1-Score: {mean_metrics['f1']:.2f}%")
    else:
        print("No valid folds with test data to compute cross-validation accuracy.")   
        
    evaluate_full_model(model, test_loader)  

def evaluate_full_model(model, test_loader):
    model.eval()
    all_labels, all_predictions = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            logits, _ = model(images.to('cuda'))
            _, predicted = torch.max(logits, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    report = classification_report(all_labels, all_predictions, zero_division=1)
    print("\nFinal Test Evaluation Report:")
    print(report)

def create_balanced_dataloader(dataset, batch_size):
    # Compute class counts and weights
    class_counts = np.bincount([label for _, label in dataset])
    class_weights = 1.0 / class_counts

    # Assign sample weights
    sample_weights = [class_weights[label] for _, label in dataset]

    # Define the sampler
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    print("Balanced DataLoader created.")
    return dataloader
        
def visualize_attention(dataset, model, idx, attentions, head=0, layer=-1):
    """
    Visualize attention maps for a spectrogram from the dataset.
    Args:
        dataset (Dataset): The spectrogram dataset (SpectrogramImageDataset).
        model (ViTClassifier): The trained Vision Transformer model.
        idx (int): Index of the spectrogram in the dataset.
        attentions: Attention outputs from the model.
        head (int): Attention head to visualize.
        layer (int): Layer to extract attention from (-1 for the last layer).
    """
    # Retrieve the spectrogram and label
    spectrogram, label = dataset[idx]
    
    # Define class mapping (adjust if necessary)
    class_mapping = {0: 'B', 1: 'N', 2: 'O', 3: 'I'}
    
    # Convert the label index to class name
    class_name = class_mapping.get(label, "Unknown")
    
    # Convert the spectrogram to a tensor and preprocess it
    image_tensor = spectrogram.unsqueeze(0).to(model.device)  # Already transformed

    # Forward pass through the model to get attentions
    logits, attentions = model(image_tensor)

    # Extract the attention map for the specified layer and head
    attention_map = attentions[layer][0, head, :, :]  # Shape: [seq_len, seq_len]

    # Average over rows (token queries)
    aggregated_attention = attention_map.mean(dim=0).detach().cpu().numpy()

    # Reshape the attention map to match spectrogram dimensions
    height, width = image_tensor.shape[-2], image_tensor.shape[-1]
    attention_resized = cv2.resize(aggregated_attention, (width, height), interpolation=cv2.INTER_LINEAR)

    # Normalize the attention map for visualization
    attention_resized = (attention_resized - np.percentile(attention_resized, 5)) / (
        np.percentile(attention_resized, 95) - np.percentile(attention_resized, 5)
    )
    # attention_resized = np.clip(attention_resized, 0, 1)
    # attention_resized = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min())

    # Convert spectrogram to numpy for visualization
    spectrogram_np = np.asarray(spectrogram).astype(float)
    if spectrogram_np.shape[0] == 3:  # Check if it's an RGB image
        spectrogram_np = np.transpose(spectrogram_np, (1, 2, 0))  # Convert to HWC format

    vmin = np.percentile(attention_resized, 5)
    vmax = np.percentile(attention_resized, 95)
    
    # Normalize spectrogram data to [0, 1]
    if spectrogram_np.max() > 1.0:
        spectrogram_np = (spectrogram_np - spectrogram_np.min()) / (spectrogram_np.max() - spectrogram_np.min())
    
    # print("Spectrogram min and max:", spectrogram_np.min(), spectrogram_np.max())
    # print("Attention map min and max:", attention_resized.min(), attention_resized.max())
    
    # Plot spectrogram and attention map side-by-side
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f"Attention Visualization for Sample {idx} - Class: {class_name}", fontsize=16)

    # Spectrogram
    spectrogram_im = axs[0].imshow(spectrogram_np, cmap="jet", aspect='auto') # Force auto aspect ratio
    #axs[0].imshow(attention_resized, cmap="plasma", alpha=0.5, aspect="auto") #Overlay attention on spectrogram
    axs[0].set_title("Original Spectrogram", fontsize=14)
    axs[0].set_ylabel("Frequency (Hz)", fontsize=12)
    axs[0].set_xlabel("Time (s)", fontsize=12)
    axs[0].axis("on")  # Ensure axes are shown

    # Add color bar for the spectrogram
    cbar_spec = fig.colorbar(spectrogram_im, ax=axs[0], fraction=0.046, pad=0.04)
    cbar_spec.set_label("Spectrogram Intensity", fontsize=12, rotation=270, labelpad=20)

    # Attention Map
    im = axs[1].imshow(attention_resized, cmap="plasma", aspect='auto', vmin=0, vmax=1)
    threshold = 0.7  # High attention threshold
    binary_mask = attention_resized > threshold
    axs[1].contour(binary_mask, colors=['black', 'gray'], linewidths=0.3, linestyles='dotted') #.contour(attention_resized, levels=5, colors="white", linewidths=0.5)  
    #axs[1].contour(binary_mask, levels=5, colors="white", linewidths=0.5)
    axs[1].set_title("Attention Map", fontsize=14)
    axs[1].set_ylabel("Attention Head Tokens", fontsize=12)
    axs[1].set_xlabel("Sequence Tokens", fontsize=12)

    # Colorbar
    cbar = fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
    cbar.set_label("Attention Weight", fontsize=12, rotation=270, labelpad=20)

    # Layout adjustment
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure to the logs folder
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_folder = "logs"
    os.makedirs(log_folder, exist_ok=True)  # Ensure logs folder exists
    file_name = f"{log_folder}/experiment_log_{timestamp}_{class_name}_{idx}.png"
    fig.savefig(file_name, bbox_inches="tight",dpi=300)  # Save the figure using fig object

    plt.show()
    plt.close(fig)  # Explicitly close the figure to free memory
    print(f"Saved attention visualization to {file_name}")