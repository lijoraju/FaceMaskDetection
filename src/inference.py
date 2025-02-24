# Inference script
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image  
import mediapipe as mp
from src.utils import load_config

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

    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    try:
        img = Image.open(image_path).convert("RGB")
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) #Convert PIL Image to OpenCV Image.

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # MediaPipe needs RGB
        results = face_detection.process(rgb_frame)

        if results.detections:
            detection = results.detections[0] 
            bbox = detection.location_data.relative_bounding_box
            ih, iw = frame.shape[:2]
            x = int(bbox.xmin * iw)
            y = int(bbox.ymin * ih)
            w = int(bbox.width * iw)
            h = int(bbox.height * ih)

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(iw, x + w)
            y2 = min(ih, y + h)

            face_img = frame[y1:y2, x1:x2]

            if face_img.size > 0: #Check if face_img has valid size
                img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)).convert("RGB") #Convert BGR to RGB and PIL Image
                img_tensor = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(img_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    _, predicted_class = torch.max(outputs, 1)
                    predicted_class = predicted_class.item()
                    return predicted_class, probabilities
            else:
              print("No face detected or face image has invalid size.")
              return None, None
        else:
          print("No face detected.")
          return None, None
    except Exception as e:
        print(f"Error during inference: {e}")
        return None, None