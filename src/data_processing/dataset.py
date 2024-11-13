import os
from PIL import Image
from torch.utils.data import Dataset
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
        
        # Extract dataset name from the first item in file_info if available
        self.dataset_name = file_info[0].get('dataset_name', 'Unknown') if file_info else 'Unknown'
        
        # Maps class names to integers
        label_mapping = {}
        for i in range(len(class_names)):
            label_mapping[class_names[i]] = i
        
        # Filter out entries in file_info where label is not in class_names
        filtered_file_info = [item for item in file_info if item['label'] in label_mapping]

        # Map file labels to corresponding integer values
        for item in filtered_file_info:
            base_name = item['filename']
            label = label_mapping[item['label']]  
            
            # Find all files belonging to the base name
            for filepath in glob.glob(os.path.join(root_dir, '**', f'{base_name}#*.png'), recursive=True):
                self.samples.append((filepath, label))  
                self.targets.append(label)  
           
        if not self.samples:
            print("Warning: No samples found matching the specified class names.")

    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image_rgb = Image.open(img_path).convert("RGB")
        image = image_rgb.resize((224, 224), Image.Resampling.LANCZOS)
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label
    
    def get_dataset_name(self):
        """Returns the dataset name for informational purposes."""
        return self.dataset_name