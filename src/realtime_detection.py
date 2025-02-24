# Real time detection
import cv2
import torch
import torchvision.transforms as T
from src.model import create_vgg16bn_model 
from src.utils import load_config
from PIL import Image
import mediapipe as mp

def realtime_detection(device="cpu"):
    config = load_config()
    if not config:
        exit("Failed to load configuration. Exiting.")
    BEST_MODEL_PATH = config['best_model_path']

    # MediaPipe face detection setup
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    # Load the trained model
    model = create_vgg16bn_model(num_classes=3, pretrained=False)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # OpenCV setup
    cap = cv2.VideoCapture(0)  # default camera

    transform = T.Compose([
            T.Resize((224, 224)),  
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    class_names = ['with_mask', 'without_mask', 'mask_weared_incorrect']

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Face detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                ih, iw = frame.shape[:2]
                x = int(bbox.xmin * iw)
                y = int(bbox.ymin * ih)
                w = int(bbox.width * iw)
                h = int(bbox.height * ih)

                # Crop the face for mask classification
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(iw, x + w)
                y2 = min(ih, y + h)

                face_img = frame[y1:y2, x1:x2]

                if face_img.size > 0:
                    # Preprocess the face image for the model
                    img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)).convert("RGB")
                    img_tensor = transform(img).unsqueeze(0).to(device)

                    # Inference
                    with torch.no_grad():
                        outputs = model(img_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        _, predicted_class = torch.max(outputs, 1)
                        predicted_class = predicted_class.item()
                        predicted_class_name = class_names[predicted_class]
                        predicted_probability = probabilities[0][predicted_class].item()

                    # Draw bounding boxes and labels on the frame
                    label = f"{predicted_class_name}: {predicted_probability:.4f}"
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Real-time Face Mask Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()