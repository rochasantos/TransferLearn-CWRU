import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.get_dataset import get_dataset
from src.data_augmentation import DataAugmentation
from functools import partial 

class PtDataset(Dataset):

    def __init__(self, data_filter, sample_size=120_000, transient_impulse_size = 540,  apply_augmentation=False,
                 label_mapping = {"I": 0, "O": 1, "B": 2}):
        self.apply_augmentation = apply_augmentation
        self.transient_impulse_size = transient_impulse_size
        data = []
        labels = []
        # metainfo
        dataset_name = data_filter["dataset_name"]
        dataset = get_dataset(dataset_name)()
        data_manager = dataset.metainfo
        metainfo = data_manager.filter_data(data_filter)
        # signal
        for info in metainfo:
            basename = info["filename"]        
            filepath = os.path.join('data/raw/', dataset_name.lower(), basename+'.mat')
            # print(f"filepath: {filepath}")          
            signal, label = dataset.load_signal_by_path(filepath)
            if signal.shape[0] < sample_size:
                continue
            m=0
            for idx in range(sample_size, signal.shape[0], sample_size):
                data.append(signal[m*sample_size:idx])
                labels.append(label_mapping[label])
                m+=1
        self.data = torch.tensor(np.array(data), dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(np.array(labels), dtype=torch.long)
    
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        signal = self.data[idx].numpy().squeeze()  # Convert tensor to numpy for augmentation
        label = self.labels[idx]

        if self.apply_augmentation:
            # Randomly select an augmentation strategy
            augmentation_method = np.random.choice([
                partial(DataAugmentation.local_data_reversing, segment_length=self.transient_impulse_size),
                partial(DataAugmentation.local_random_reversing, segment_length=self.transient_impulse_size),
                DataAugmentation.global_data_reversing,
                partial(DataAugmentation.local_data_zooming, zoom_range=(0.8, 1.2), segment_length=self.transient_impulse_size),
                partial(DataAugmentation.global_data_zooming, zoom_range=(0.8, 1.2)),
                partial(DataAugmentation.local_segment_splicing, segment_length=self.transient_impulse_size),
                partial(DataAugmentation.noise_addition, snr_db=20),
            ])
        
            # Apply the selected augmentation
            signal = augmentation_method(signal)
        else:
            signal = signal.copy()
        # Convert back to PyTorch tensor
        signal = torch.tensor(signal.copy(), dtype=torch.float32).unsqueeze(0)  # [1, sample_size]
        
        return signal, label
