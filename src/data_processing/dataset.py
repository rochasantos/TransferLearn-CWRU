import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import torch

class SpectrogramImageDataset(Dataset):

    def __init__(self, root_dir, file_info, class_names, transform=None):
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
        
        # Maps class names to integers
        label_mapping = {}
        for i in range(len(class_names)):
            label_mapping[class_names[i]] = i
        
        # Map file labels to corresponding integer values
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
