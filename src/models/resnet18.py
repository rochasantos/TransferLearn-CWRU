import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNet18, self).__init__()
        # Load the pretrained ResNet18 model
        self.resnet18 = models.resnet18(pretrained=True)
        
        # Modify the last fully connected layer to match the number of output classes
        # ResNet18 originally has 512 features before the final classification layer
        self.resnet18.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.resnet18(x)
        return x
