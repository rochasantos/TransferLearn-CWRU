import torch
import torch.nn as nn
from transformers import ViTForImageClassification

class ViTClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(ViTClassifier, self).__init__()
        # Load the pre-trained ViT model and specify the number of output classes
        self.vit = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224", 
            num_labels=num_classes,  # Setting the number of output classes
            ignore_mismatched_sizes=True  # Ignore mismatched size errors
        )

    def forward(self, x):
        # ViTForImageClassification expects input images in a specific format
        output = self.vit(pixel_values=x)
        return output.logits  # Return only the logits for consistency with your pipeline
