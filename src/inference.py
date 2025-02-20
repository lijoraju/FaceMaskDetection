# Inference script


import torch
import torchvision.transforms as T
from PIL import Image  


def inference(model, image_path, device="cpu"):
    """Performs inference on a single image.

    Args:
        model: The trained PyTorch model.
        image_path: Path to the image file.
        device: The device to use (CPU or GPU).

    Returns:
        The predicted class label (integer) and the probabilities.
        Or, if there's an error, it can return None, None.
    """
    try:
        # 1. Load and preprocess the image
        img = Image.open(image_path).convert("RGB") 
        transform = T.Compose([
            T.Resize((224, 224)),  # Resize to match training images
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension (batch size 1)

        # 2. Move the input tensor to the device
        img_tensor = img_tensor.to(device)

        # 3. Set the model to evaluation mode
        model.eval()

        # 4. Disable gradients (for inference)
        with torch.no_grad():
            # 5. Perform inference
            outputs = model(img_tensor)

            # 6. Get predictions (probabilities and class label)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)  # Apply softmax to get probabilities
            _, predicted_class = torch.max(outputs, 1) 

            predicted_class = predicted_class.item()
            return predicted_class, probabilities

    except Exception as e:
        print(f"Error during inference: {e}")
        return None, None