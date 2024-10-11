from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

dataset = datasets.ImageFolder(root='test/data/processed/train_spectrograms')

X = np.arange(len(dataset))  # get index from spectrograms
y = dataset.targets  # class tags
groups = [dataset.samples[i][0].split('/')[-2] for i in range(len(dataset))]

skf = StratifiedGroupKFold(n_splits=5)

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y, groups)):
    print(f"Fold {fold + 1}")
    
    train_subset = Subset(dataset, train_idx)
    test_subset = Subset(dataset, test_idx)
    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False, num_workers=4)

