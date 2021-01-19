import torch.nn as nn
from torchvision import models


class Resnet50(nn.Module):
    def __init__(self, num_target_classes: int, pretrained: bool):
        super().__init__()

        self.model = models.resnet50(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_target_classes)

    def forward(self, x):
        y = self.model(x)
        return y
