import torch
import torch.nn as nn
from torchvision import models


class ResNet50Model(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(self.base_model.fc.weight.shape[1], num_class)
        self.num_class = num_class
    def forward(self, x):
        x = self.base_model(x)
        return x
        
class MobileNetV2(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1, dropout=0.4)
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Linear(in_features, num_class)
        )
        self.feature_maps = []
        self.hook = self.base_model.features[18][0].register_forward_hook(self.hook_fn) # To take filter of conv2
    def forward(self, x):
        x = self.base_model(x)
        return x
    def hook_fn(self, module, input, output):
        self.feature_maps = [output.detach().cpu()] # Reduce memory

class ViT16(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.base_model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        in_features = self.base_model.heads[0].in_features
        self.base_model.heads = nn.Sequential(
            nn.Linear(in_features, num_class)
        )
        self.num_class = num_class
    def forward(self, x):
        x = self.base_model(x)
        return x
