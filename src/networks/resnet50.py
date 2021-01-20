import torch
import torch.nn as nn
from torchvision import models


class Resnet50(nn.Module):
    def __init__(self, num_target_classes: int, pretrained: bool):
        super().__init__()

        self.model = models.resnet50(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_target_classes)

        self.without_output = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x):
        y = self.model(x)
        return y

    def get_y_and_emb(self, x):
        emb = self.without_output(x)
        y = self.model.fc(torch.flatten(emb, 1))
        return y, emb.view(-1, 2048)
