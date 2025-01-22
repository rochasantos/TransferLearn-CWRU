import torch
import torchvision.transforms.functional as TF

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        # Converte para tensor caso ainda não seja
        if not isinstance(img, torch.Tensor):
            img = TF.to_tensor(img)
        
        # Adiciona ruído gaussiano
        noise = torch.randn(img.size()) * self.std + self.mean
        return torch.clamp(img + noise, 0.0, 1.0)  # Garante que os valores estão entre [0, 1]

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"
