import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
def get_model():
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2) 
    return model