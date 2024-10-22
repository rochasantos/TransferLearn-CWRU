import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import torch

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, file_info, transform=None):
        """
        Args:
            root_dir (string): Root directory containing the images separated by classes.
            file_info (list): List of dictionaries containing the base names of the files and their labels.
            transform (callable, optional): Transformations to be applied to the images.
        """
        self.root_dir = root_dir
        self.file_info = file_info
        self.transform = transform
        
        
        self.samples = []
        self.targets = []  
        
        label_mapping = {'N': 0, 'I': 1, 'O': 2, 'B': 3}
        
        for item in file_info:
            base_name = item['filename']
            label = label_mapping[item['label']]  
            
            # Find all files belonging to the base name
            for filepath in glob.glob(os.path.join(root_dir, '**', f'{base_name}#*.png'), recursive=True):
                self.samples.append((filepath, label))  
                self.targets.append(label)  
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label
