import torch

class EarlyStopping:
    def __init__(self, patience=5, delta=0, save_path="best_model.pth", no_save_model=False):
        """
        :param patience: Number of epochs to wait before stopping if there is no improvement.
        :param delta: Minimum difference to consider that there has been an improvement.
        :param save_path: Path to save the best model.
        """
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.no_save_model = no_save_model

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.no_save_model or self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.no_save_model or self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):       
        torch.save(model.state_dict(), self.save_path)
        print(f"Model saved with validation loss: {val_loss:.4f}")
