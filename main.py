
import argparse

from src.train import train_model  
from src.inference import inference
from src.data_preprocessing import FaceMaskDataset, process_annotations, download_dataset
from src.model import create_mobilenetv3_model, load_mobilenetv3_model
from src.evaluate import evaluate_test_set
from src.utils import load_config
from src.realtime_detection import realtime_detection

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Mask Detection")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--inference", type=str, help="Path to image for inference")
    parser.add_argument("--realtime", action="store_true", help="Perform real-time detection")
    args = parser.parse_args()

    config = load_config()
    if config:
        DATASET = config['kaggle_dataset']
        ANNOTATION_PATH, IMAGE_PATH = download_dataset(DATASET)
        MODEL_PATH = config['best_model_path']
        BATCH_SIZE = config['batch_size']
        LR = config['learning_rate']
        NUM_OF_EPOCHS = config['num_epochs']
        PATIENCE = config['patience']
    else:
        exit("Failed to load configuration. Exiting.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train:
        #Load dataframe
        df = process_annotations(ANNOTATION_PATH)

        #Split the data
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['label'])

        #Create datasets
        train_dataset = FaceMaskDataset(train_df, IMAGE_PATH, augment=True)
        val_dataset = FaceMaskDataset(val_df, IMAGE_PATH, augment=False)
        test_dataset = FaceMaskDataset(test_df, IMAGE_PATH, augment=False)

        #Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
 
        model = create_mobilenetv3_model(num_classes=3, pretrained=True)
        model.to(device)

        train_model(model, train_loader, val_loader, num_epochs=NUM_OF_EPOCHS, learning_rate=LR, patience=PATIENCE, device=device)
        print("Training finished!")
        # After training, load the best model and evaluate on the test set:
        model = load_mobilenetv3_model(MODEL_PATH, num_classes=3, device=device)
        test_loss, test_accuracy, precision, recall, f1, auc_pr = evaluate_test_set(model, test_loader, device)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, "
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, AUC-PR: {auc_pr:.4f}")
    elif args.inference:
        model = load_mobilenetv3_model(MODEL_PATH, num_classes=3, device=device)
        predicted_class, probabilities = inference(model, args.inference, device)
        if predicted_class is not None:
            print(f"Predicted Class: {predicted_class}")
            print(f"Probabilities: {probabilities}")

            #Interpret probabilities
            class_names = ['with_mask', 'without_mask', 'mask_weared_incorrect'] 
            predicted_class_name = class_names[predicted_class]
            print(f"Predicted Class Name: {predicted_class_name}")

            #Get probability of the predicted class
            predicted_probability = probabilities[0][predicted_class].item()
            print(f"Probability of Predicted Class: {predicted_probability:.4f}")
        else:
            print("Inference failed.")
    elif args.realtime:
        realtime_detection(device)