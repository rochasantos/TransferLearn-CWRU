import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from src.models.cnn2d import CNN2D
from scripts.evaluate_model import evaluate_model

import sys
import logging
from utils.logginout import LoggerWriter
from utils.print_info import print_info


def experimenter_cross_dataset():

    model = CNN2D()
    
    # Loads the pre-trained model
    saved_model_path = "saved_models/cnn2d-hust-e30-lr0001.pth"
    model.load_state_dict(torch.load(saved_model_path, weights_only=True))
    model.to(device="cuda")

    # Experiment log
    title = "Transfer Learning: Addressing the CWRU Dataset with Hust"
    print_info("Experiment", [title])
    print_info("Model", [str(model)])
    print(f"Saved model path: {saved_model_path}")
    
    # Running the experiment     
    target_dir = "data/spectrograms/cwru"
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
    ])                        
    test_dataset = ImageFolder(target_dir, transform)                
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4)
    dataset_class = test_dataset.classes

    # Evaluating the model
    print(f"Evaluating the model on CWRU dataset.")
    evaluate_model(model, test_loader, dataset_class, device="cuda")

def run_experimenter():     
    experimenter_cross_dataset()


if __name__ == "__main__":
    sys.stdout = LoggerWriter(logging.info, "cross-domain_s-hust_t-cwru")
    run_experimenter()