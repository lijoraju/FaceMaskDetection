import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import mediapipe as mp
from src.model import load_mobilenetv3_model  
from src.utils import load_config

class_colors = {
    'with_mask': (0, 255, 0), 
    'without_mask': (0, 0, 255), 
    'mask_weared_incorrect': (255, 0, 0) 
}

# Load configuration and model
config = load_config()
if config:
    MODEL_PATH = config['best_model_path']
else:
    exit("Failed to load configuration. Exiting.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_mobilenetv3_model(MODEL_PATH, num_classes=3, device=device)

# MediaPipe face detection setup
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Image transforms
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = ['with_mask', 'without_mask', 'mask_weared_incorrect']

def predict(image):
    """Predicts the mask status for a given image."""
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

        if face_img.size > 0:
            img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted_class = torch.max(outputs, 1)
                predicted_class = predicted_class.item()
                predicted_class_name = class_names[predicted_class]
                predicted_probability = probabilities[0][predicted_class].item()
            return predicted_class_name, predicted_probability, (x, y, x + w, y + h)
        else:
            return None, None, None
    else:
        return None, None, None

def main():
    st.title("Face Mask Detection")

    mode = st.radio("Select Mode:", ("Upload Image", "Live Camera"))

    if mode == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image.", use_container_width=True)
            st.write("")
            st.write("Predicting...")

            predicted_class, probability, bbox = predict(image)

            if predicted_class:
                st.write(f"Predicted Class: {predicted_class}")
                st.write(f"Probability: {probability:.4f}")

                if bbox:
                    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    color = class_colors[predicted_class]
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Detected Face", use_container_width=True)

            else:
                st.write("No face detected.")

    elif mode == "Live Camera":
        st.write("Opening camera...")
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            st.error("Could not open camera.")
            return

        frame_placeholder = st.empty()

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            predicted_class, probability, bbox = predict(image)

            if predicted_class:
                if bbox:
                    color = class_colors[predicted_class]
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    cv2.putText(frame, f"{predicted_class}: {probability:.2f}", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()