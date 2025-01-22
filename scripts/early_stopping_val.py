import torch

class EarlyStopping:
    def __init__(self, patience=5, delta=0.01, save_path='best_model.pth', restore_best_weights=True):       
        """
        Early stopping class to monitor validation accuracy during training.

        Args:
            patience (int): Number of epochs to wait before stopping if no improvement.
            delta (float): Minimum change in accuracy to qualify as an improvement.
            save_path (str): Path to save the best model weights.
            restore_best_weights (bool): If True, restores model to the best weights at early stop.
        """
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.restore_best_weights = restore_best_weights
        self.best_accuracy = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_accuracy, model):        
        """
        Monitor validation accuracy and decide whether to stop training.

        Args:
            val_accuracy (float): Current epoch's validation accuracy.
            model (torch.nn.Module): The model being trained.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        if self.best_accuracy is None or val_accuracy > self.best_accuracy + self.delta:
            self.best_accuracy = val_accuracy
            self.counter = 0
            self.best_weights = model.state_dict()
            # Save the best model
            torch.save(self.best_weights, self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    # Restore the best model weights before stopping
                    model.load_state_dict(self.best_weights)
                return True
        return False
