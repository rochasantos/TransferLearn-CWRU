import torch

class EarlyStopping:
    def __init__(self, patience=5, delta=0.01, save_path='best_model.pth', restore_best_weights=True, save_model=True):       
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        self.save_model = save_model

    def __call__(self, val_loss, model):        
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = model.state_dict()
            # Salva o melhor modelo
            if self.save_model:
                torch.save(self.best_weights, self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    # Restaura o melhor modelo antes de parar
                    model.load_state_dict(self.best_weights)
                return True
        return False
