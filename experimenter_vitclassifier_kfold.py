import torch
import os
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from src.models import CNN2D, ViTClassifier, ResNet18
from src.models.vitclassifier import train_and_save, load_trained_model
from scripts.evaluate_model_vitclassifier import kfold_cross_validation, resubstitution_test, one_fold_with_bias, one_fold_without_bias

import sys
import logging
from utils.logginout import LoggerWriter
from utils.print_info import print_info

def experimenter_vitclassifier_kfold():

    model = ViTClassifier(num_classes=4).to("cuda") 
    # Training parameters 
    num_epochs_vit_train = 10
    lr_vit_train = 0.0001
    batch_size = 32
    
    # Class-to-index mapping
    class_to_idx = {'B': 0, 'I': 1, 'N': 2, 'O': 3}
    
    # Reverse mapping for debugging or display purposes
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # Loads the pre-trained model
    saved_model_path = "saved_models/vit_classifier.pth"

    # Experiment log
    title = "Transfer Learning: Addressing cross Datasets with ViTClasifier"
    print_info("Experiment", [title])
    print(f"Saved model path: {saved_model_path}")
    
    if os.path.exists(saved_model_path):
        print(f"Loading model from {saved_model_path}...")
        model.load_state_dict(torch.load(saved_model_path, weights_only=True, map_location="cuda"))
    else:
        print("Pre-training required. No saved model found.")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
    ])        
    
    # if os.path.exists(saved_model_path):
    #     print(f"There is already an estimator on path '{saved_model_path}'")
    #     model.load_state_dict(torch.load(saved_model_path, weights_only=True))
    #     #return model

    # Datasets that will be used in pre-training
    datasets_name = ["UORED"]
    
    root_dir = "data/spectrograms"
    train_datasets = [ImageFolder(os.path.join(root_dir, ds.lower()), transform) for ds in datasets_name]
    train_concated_dataset = ConcatDataset(train_datasets)
    train_loader = DataLoader(train_concated_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    train_labels = []
    for _, labels in train_loader:
        train_labels.extend(labels.cpu().numpy())

    train_distribution = {idx: train_labels.count(idx) for class_name, idx in class_to_idx.items()}
    print(f"[debug] Train Distribution from Loader (before call the validation): {train_distribution}")

    print("Train dataset:" , datasets_name[0])
    
    # Instantiate the ViTClassifier and train it with dataset2 to narrow the model context
    pretrain_model = False
    if pretrain_model: 
        model = ViTClassifier().to("cuda")
        train_and_save(model, train_loader, num_epochs_vit_train, lr_vit_train, saved_model_path)  # Train and save the model

    # Load the trained model for testing/evaluation
    model = load_trained_model(ViTClassifier, saved_model_path, num_classes=len(class_to_idx)).to("cuda")
    
    # Running the experiment 
    
    # Datasets that will be used in pre-training
    datasets_name = ["CWRU"]
        
    root_dir = "data/spectrograms/"                
    test_dataset = [ImageFolder(os.path.join(root_dir, ds.lower()), transform) for ds in datasets_name]
    test_concated_dataset = ConcatDataset(test_dataset)                
    test_loader = DataLoader(test_concated_dataset, batch_size=32, shuffle=True, num_workers=4)
 
    test_labels = []
    for _, labels in test_loader:
        test_labels.extend(labels.cpu().numpy())

    test_distribution = {idx: test_labels.count(idx) for class_name, idx in class_to_idx.items()}
    print(f"[debug] Test Distribution from Loader (before call the validation): {test_distribution}")

    print("Test dataset:" , datasets_name[0])
   
    num_epochs = 10
    lr = 0.0002
    group_by = "rpm" 
    #group_by = "extent_damage"
    #group_by = "condition_bearing_health"
    #group_by = "damage_method"
    #group_by = ""

    # Evaluating the model
    print(f"Evaluating the model on CWRU dataset.")
    
    ds_test_loader = test_loader.dataset.datasets[0]
    
    # resubstitution_test(
    #     model,
    #     ds_test_loader,
    #     num_epochs,
    #     lr,
    #     class_names = list(class_to_idx.keys())
    # )
    
    # one_fold_with_bias(
    #     model,
    #     ds_test_loader,
    #     num_epochs,
    #     lr,
    #     class_names = list(class_to_idx.keys())
    # )
    
    # one_fold_without_bias(
    #     model,
    #     ds_test_loader,
    #     num_epochs,
    #     lr,
    #     class_names = list(class_to_idx.keys())
    # )
    
    kfold_cross_validation(
        model, 
        test_loader, 
        num_epochs, 
        lr, 
        group_by, 
        class_names = list(class_to_idx.keys()), 
        n_splits=4)

def run_experimenter():     
    experimenter_vitclassifier_kfold()


if __name__ == "__main__":
    #sys.stdout = LoggerWriter(logging.info, "kfold-vitclassifier")
    run_experimenter()