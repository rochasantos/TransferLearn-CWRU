import torch.nn as nn

class ModelFactory:
    def __init__(self, model_class, *args, **kwargs):        
        self.model_class = model_class
        self.args = args
        self.kwargs = kwargs

    def create_model(self):
        return self.model_class(*self.args, **self.kwargs)