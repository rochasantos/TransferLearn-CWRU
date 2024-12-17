import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig, ViTImageProcessor
from torchvision.transforms import ToPILImage
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


class ViTClassifier(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.6):
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
        #self.vit.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)
        self.vit.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),  # Add Dropout before the final layer
            nn.Linear(self.vit.config.hidden_size, num_classes)
        )


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
def train_and_save(model, train_loader, num_epochs, lr=0.001, save_path="vit_classifier.pth", patience=3, weight_decay=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    class_weights = torch.tensor([1.0, 1.0, 0.25, 1.0]).to(device) 
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, verbose=True)
    
    best_loss = float("inf")  
    patience_counter = 0
    
    results = {
        "epoch": [],
        "loss": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": []
    }
    
    print('Starting VitClassifier Training...')    
    print(f"LR: {lr} | Num Epochs: {num_epochs} | Weight Decay: {weight_decay} | Early Stopping Patience: {patience}")

    
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        all_labels = []
        all_predictions = []
 
        for images, labels in train_loader:
            images, labels = images.to(model.device), labels.to(model.device)
            
            optimizer.zero_grad()
            logits, attentions = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # Collect predictions and labels for metrics calculation
            _, preds = torch.max(logits, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())

            
         # Calculate average loss for the epoch
        avg_loss = running_loss / len(train_loader)
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions) * 100
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0) * 100
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0) * 100
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0) * 100
        
        results["epoch"].append(epoch + 1)
        results["loss"].append(avg_loss)
        results["accuracy"].append(accuracy)
        results["precision"].append(precision)
        results["recall"].append(recall)
        results["f1_score"].append(f1)
        
        print(f"Epoch [{epoch + 1}/{num_epochs}] Metrics: | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}% | Precision: {precision:.2f}% | Recall: {recall:.2f}% | F1-Score: {f1:.2f}%")

        # Step the scheduler with the average loss
        scheduler.step(avg_loss)

        # Early stopping criteria
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            #print(f"Model improved and saved at {save_path}.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered. No improvement in validation loss.")
                break

    if patience_counter < patience:
        print("Training completed successfully.")
    else:
        print("Training stopped early due to lack of improvement.")
     
    # Save the trained model weights
    torch.save(model.state_dict(), save_path)
    print("Model saved successfully:", save_path)
    
    return results

# To reload the trained model later
def load_trained_model(model_class, save_path, num_classes=4):
    # Instantiate model with specified num_classes
    model = model_class(num_classes=num_classes) 
    model.load_state_dict(torch.load(save_path, weights_only=True))  
    
    return model