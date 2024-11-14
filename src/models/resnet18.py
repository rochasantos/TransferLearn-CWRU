import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self, num_classes=4, unfreeze_layers=None):
        super(ResNet18, self).__init__()
        
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        if unfreeze_layers:
            for name, param in self.model.named_parameters():
                if any(layer in name for layer in unfreeze_layers):
                    param.requires_grad = True
        
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)
