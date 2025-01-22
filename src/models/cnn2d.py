import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN2D(nn.Module):
    def __init__(self, input_size=(224, 224)):
        super(CNN2D, self).__init__()
        
        # First convolutional layer with Batch Normalization
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(16)
        
        # Second convolutional layer with Batch Normalization
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(32)
        
        # Third convolutional layer with Batch Normalization
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(64)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Compute the output size of the convolutional layers
        conv_output_size = self._get_conv_output_size(input_size)
        
        # Fully connected layers with Dropout
        self.fc1 = nn.Linear(conv_output_size, 224)
        self.dropout = nn.Dropout(p=0.5)  # Dropout with 50% probability
        self.fc2 = nn.Linear(224, 4)  # 4 classes: inner, outer, ball, normal.

    def _get_conv_output_size(self, input_size):
        """Dynamically computes the output size of the convolutional layers."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, *input_size)  # Simulated input
            x = self.pool(self.batch_norm1(F.relu(self.conv1(dummy_input))))
            x = self.pool(self.batch_norm2(F.relu(self.conv2(x))))
            x = self.pool(self.batch_norm3(F.relu(self.conv3(x))))
            return x.numel()  # Total number of elements in the convolutional output

    def forward(self, x):
        # Apply convolutional layers with Batch Normalization and pooling
        x = self.pool(self.batch_norm1(F.relu(self.conv1(x))))
        x = self.pool(self.batch_norm2(F.relu(self.conv2(x))))
        x = self.pool(self.batch_norm3(F.relu(self.conv3(x))))
        
        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers with Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
