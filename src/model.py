# PyTorch model architecture
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights

def create_mobilenetv3_model(num_classes=3, pretrained=True, mobilenet_version="mobilenet_v3_large"):
    """Creates a MobileNetV3 model."""
    if mobilenet_version == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(pretrained=pretrained, weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    elif mobilenet_version == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(pretrained=pretrained, weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    else:
        raise ValueError("Invalid MobileNetV3 version. Choose from 'mobilenet_v3_small' or 'mobilenet_v3_large'.")

    in_features = model.classifier[3].in_features 
    model.classifier[3] = nn.Linear(in_features, num_classes)  
    return model

def load_mobilenetv3_model(model_path, num_classes=3, mobilenet_version="mobilenet_v3_large", device="cpu"):
    model = create_mobilenetv3_model(num_classes, pretrained=False, mobilenet_version=mobilenet_version)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False) 
    model.to(device)
    model.eval()
    return model