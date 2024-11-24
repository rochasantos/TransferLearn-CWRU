import torch
import os
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from src.models import CNN2D, ViTClassifier, ResNet18
from src.models.vitclassifier import train_and_save, load_trained_model
from scripts.evaluate_model_vitclassifier import kfold_cross_validation

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
    # The dataset1 = CWRU
    class_names = ['N', 'I', 'O', 'B']  # Your dataset class names
    
    # Loads the pre-trained model
    saved_model_path = "saved_models/vit_classifier.pth"
    model.load_state_dict(torch.load(saved_model_path, weights_only=True))
    model.to(device="cuda")

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
    print("Train dataset:" , datasets_name[0])
    
    # Instantiate the ViTClassifier and train it with dataset2 to narrow the model context
    pretrain_model = True
    if pretrain_model: 
        model = ViTClassifier().to("cuda")
        train_and_save(model, train_loader, num_epochs_vit_train, lr_vit_train, saved_model_path)  # Train and save the model

    # Load the trained model for testing/evaluation
    model = load_trained_model(ViTClassifier, saved_model_path, num_classes=len(class_names)).to("cuda")
    # Running the experiment 
    target_dataset = "cwru"    
    target_dir = "data/spectrograms/" + target_dataset                
    test_dataset = ImageFolder(target_dir, transform)                
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4)
   
    num_epochs = 10
    lr = 0.0002
    group_by = "rpm" 
    #group_by = "extent_damage"
    #group_by = "condition_bearing_health"
    #group_by = "damage_method"
    #group_by = ""

    # Evaluating the model
    print(f"Evaluating the model on CWRU dataset.")
    #kfold_cross_validation(model, test_loader, dataset_class, device="cuda")
    kfold_cross_validation(model, test_loader, num_epochs, lr, group_by, class_names, n_splits=4)

def run_experimenter():     
    experimenter_vitclassifier_kfold()


if __name__ == "__main__":
    #sys.stdout = LoggerWriter(logging.info, "kfold-vitclassifier")
    run_experimenter()