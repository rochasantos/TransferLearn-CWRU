import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN2D(nn.Module):
    def __init__(self):
        super(CNN2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.fc1 = None  
        self.fc2 = nn.Linear(224, 4)  # 4 classes - inner, outer, ball, and normal.

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        if self.fc1 is None:
            num_features = x.shape[1] * x.shape[2] * x.shape[3]
            self.fc1 = nn.Linear(num_features, 224).to(x.device)
        
        x = x.view(x.size(0), -1)  # Flattening
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
