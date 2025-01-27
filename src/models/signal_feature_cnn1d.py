import torch
import torch.nn as nn
import torch.nn.functional as F

# Model definition with modified FeatureExtractor
class SignalFeatureCNN1D(nn.Module):
    def __init__(self):
        super(SignalFeatureCNN1D, self).__init__()

        # Four convolutional blocks
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(4)

        self.conv2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(8)

        self.conv3 = nn.Conv1d(in_channels=8, out_channels=12, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(12)

        self.conv4 = nn.Conv1d(in_channels=12, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(16)

        # Fully connected layer
        self.fc1 = nn.Linear(16 * 469, 128)  # Adjusted for input size
        self.fc2 = nn.Linear(128, 10)  # Example: 10 classes

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=2)

        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=2)

        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.max_pool1d(x, kernel_size=2)

        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = F.max_pool1d(x, kernel_size=2)

        x = x.view(x.size(0), -1)  # Flatten
        features = self.fc1(x)
        outputs = self.fc2(features)
        return features, outputs  # Return both features and classification outputs