import torch
import torch.nn as nn
import torch.nn.functional as F

# Model definition with modified FeatureExtractor
class SignalFeatureCNN1D(nn.Module):
    def __init__(self, input_length):
        super().__init__()

        # Four convolutional blocks
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(4)

        self.conv2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(8)

        self.conv3 = nn.Conv1d(in_channels=8, out_channels=12, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(12)

        self.conv4 = nn.Conv1d(in_channels=12, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(16)

        self.feature_size = self._get_conv_output_size(input_length)

        # Fully connected layer
        self.fc1 = nn.Linear(self.feature_size, 128)  # Adjusted for input size
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, 3)  # 3 classes (I, O, B)

    def _get_conv_output_size(self, input_length):
        """ Calcula automaticamente o tamanho da saída antes de `fc1` """
        x = torch.zeros(1, 1, input_length)  # Tensor fictício para cálculo
        x = F.max_pool1d(F.leaky_relu(self.bn1(self.conv1(x))), kernel_size=2)
        x = F.max_pool1d(F.leaky_relu(self.bn2(self.conv2(x))), kernel_size=2)
        x = F.max_pool1d(F.leaky_relu(self.bn3(self.conv3(x))), kernel_size=2)
        x = F.max_pool1d(F.leaky_relu(self.bn4(self.conv4(x))), kernel_size=2)
        return x.numel()  # Retorna o número total de elementos achatados

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=2)

        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=2)

        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.max_pool1d(x, kernel_size=2)

        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = F.max_pool1d(x, kernel_size=2)

        x = x.view(x.size(0), -1)  # Flatten
        features = torch.relu(self.fc1(x))
        features = self.dropout(features)
        outputs = self.fc2(features)
        return features, outputs  # Return both features and classification outputs
    
    def freeze_layers(self, layer_names):
        """
        Congela as camadas especificadas.

        :param layer_names: Lista contendo os nomes das camadas a serem congeladas.
        """
        for layer_name in layer_names:
            layer = getattr(self, layer_name, None)  # Obtém a camada pelo nome
            if layer:
                for param in layer.parameters():
                    param.requires_grad = False
                print(f"✅ Camada {layer_name} congelada.")
            else:
                print(f"⚠️ Aviso: {layer_name} não encontrado no modelo.")

    def unfreeze_layers(self, layer_names):
        """
        Descongela as camadas especificadas.

        :param layer_names: Lista contendo os nomes das camadas a serem descongeladas.
        """
        for layer_name in layer_names:
            layer = getattr(self, layer_name, None)  # Obtém a camada pelo nome
            if layer:
                for param in layer.parameters():
                    param.requires_grad = True
                print(f"🔓 Camada {layer_name} descongelada.")
            else:
                print(f"⚠️ Aviso: {layer_name} não encontrado no modelo.")