import numpy as np
import torch
from torchvision.datasets import ImageFolder
import librosa

class DatasetAugmented(ImageFolder):
    def __init__(self, root, transforms=None, augmentation=None):
        super().__init__(root, transform=transforms)
        self.augmentation = augmentation

    def __getitem__(self, index):
        # Obtém a imagem e o rótulo usando a classe pai
        image, label = super().__getitem__(index)

        # Converte o tensor PyTorch para NumPy
        if isinstance(image, torch.Tensor):
            # Reordena as dimensões de (C, H, W) para (H, W, C)
            image = image.permute(1, 2, 0).numpy()

        # Aplica a augmentação de dados, se existir
        if self.augmentation:
            augmented = self.augmentation(image=image)
            image = augmented['image']

        # Converte de volta para tensor
        image = torch.from_numpy(image).permute(2, 0, 1)

        return image, label


def apply_pitch_shift(signal, sampling_rate, n_steps):
    
    return librosa.effects.pitch_shift(signal, sr=sampling_rate, n_steps=n_steps)