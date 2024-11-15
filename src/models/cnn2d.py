import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN2D(nn.Module):
    def __init__(self, input_size=(224, 224)):
        super(CNN2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Compute the output size of the convolutional layers
        conv_output_size = self._get_conv_output_size(input_size)
        
        # Define the fully connected layers based on the computed output size
        self.fc1 = nn.Linear(conv_output_size, 224)
        self.fc2 = nn.Linear(224, 4)  # 4 classes: inner, outer, ball, normal.

    def _get_conv_output_size(self, input_size):
        """Dynamically computes the output size of the convolutional layers."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, *input_size)  # Simulated input
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            return x.numel()  # Total number of elements in the convolutional output

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
