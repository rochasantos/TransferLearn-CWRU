import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig, ViTImageProcessor
from torchvision.transforms import ToPILImage

class ViTClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(ViTClassifier, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the processor for preprocessing images
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.processor.do_rescale = False

        # Load the ViT model with the correct number of classes, replacing the classifier
        self.vit = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=num_classes,
            output_attentions=True, # Enable attention outputs
            ignore_mismatched_sizes=True,  # To handle mismatched classifier layer size
            attn_implementation="eager"  # Explicitly specify the attention implementation
        )
        
        # Manually adjust the classifier layer if necessary
        self.vit.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)


    def forward(self, x):
        # Convert tensors back to PIL images for compatibility with ViTImageProcessor
        to_pil = ToPILImage()
        images = [to_pil(img) for img in x]

        # Preprocess images with the processor for ViT compatibility
        inputs = self.processor(images=images, return_tensors="pt").pixel_values.to(x.device)

        # Forward pass through the model
        output = self.vit(pixel_values=inputs)
        return output.logits, output.attentions  # Return logits and attention weights

    
# To train and save the model after training
def train_and_save(model, train_loader, num_epochs, lr=0.001, save_path="vit_classifier.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    
    print('Starting VitClassifier Training...')
    print("Training dataset >>>" ,train_loader.dataset.get_dataset_name())
    print("LR:" ,lr)
    print("Num Epochs:" ,num_epochs)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(model.device), labels.to(model.device)
            
            optimizer.zero_grad()
            logits, attentions = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")
    
    # Save the trained model weights
    torch.save(model.state_dict(), save_path)
    print("Model saved successfully.")

# To reload the trained model later
def load_trained_model(model_class, save_path, num_classes=4):
    # Instantiate model with specified num_classes
    model = model_class(num_classes=num_classes) 
    model.load_state_dict(torch.load(save_path, weights_only=True))  # Add weights_only=True
    
    return model
