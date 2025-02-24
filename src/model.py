# PyTorch model architecture
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision.models import VGG16_BN_Weights

def create_resnet_model(num_classes=3, pretrained=True, resnet_version="resnet50"):  
    """Creates a ResNet model."""
    if resnet_version == "resnet18":
        model = models.resnet18(pretrained=pretrained)
    elif resnet_version == "resnet34":
        model = models.resnet34(pretrained=pretrained)
    elif resnet_version == "resnet50":
        model = models.resnet50(pretrained=pretrained)
    else:
        raise ValueError("Invalid ResNet version. Choose from 'resnet18', 'resnet34', 'resnet50'.")

    in_features = model.fc.in_features  
    model.fc = nn.Linear(in_features, num_classes)
    return model

def load_resnet_model(model_path, num_classes=3, resnet_version="resnet50", device="cpu"):
    model = create_resnet_model(num_classes, pretrained=False, resnet_version=resnet_version)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model