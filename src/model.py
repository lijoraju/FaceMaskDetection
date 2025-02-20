# PyTorch model architecture

import torch
import torchvision.models as models
import torch.nn as nn
from torchvision.models import VGG16_BN_Weights


def create_vgg16bn_model(num_classes, pretrained=True):
    """Creates a pre-trained VGG16 with Batch Normalization model."""
    
    weights = VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None 
    
    # Load the pre-trained VGG16 with Batch Normalization
    model = models.vgg16_bn(weights=weights)

    # Modify the last layer 
    in_features = model.classifier[6].in_features  # Get input features of last layer
    model.classifier[6] = nn.Linear(in_features, num_classes)  # Replace last layer

    return model