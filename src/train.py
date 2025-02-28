# Training loop
import torch
import torch.nn as nn
import torch.optim as optim
from rich.progress import track
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc


def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, patience=3, device="cpu"):
    """Trains a PyTorch model.

    Args:
        model: The PyTorch model to train.
        train_loader: DataLoader for the training data.
        val_loader: DataLoader for the validation data.
        num_epochs: The number of training epochs.
        learning_rate: The learning rate for the optimizer.
        patience: The number of epochs to wait for improvement before stopping.
        device: The device to use (CPU or GPU).

    Returns:
        None.  Prints training and validation loss and accuracy during training.
        Saves the best model checkpoint to 'best_model.pth'.
    """

    all_labels = torch.cat([labels for images, labels in train_loader])
    class_weights = calculate_class_weights(all_labels).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Optimizer

    best_val_loss = float('inf')  # Initialize best validation loss
    train_losses = []
    val_losses = []
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()  # Set model to training mode
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in track(train_loader, description=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()  # Zero the gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            train_loss += loss.item() * images.size(0)  # Accumulate loss
            _, predicted = torch.max(outputs.data, 1)  # Get predictions
            train_total += labels.size(0)  # Accumulate count
            train_correct += (predicted == labels).sum().item() #Accumulate correct predictions
            

        avg_train_loss = train_loss / train_total
        train_accuracy = 100 * train_correct / train_total
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():  # Disable gradients for validation
            for images, labels in track(val_loader, description=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)

                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())


        avg_val_loss = val_loss / val_total
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(avg_val_loss)

        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')

        all_probs = np.array(all_probs)
        all_labels_one_hot = np.eye(3)[all_labels] #One hot encode labels
        precision_curve, recall_curve, _ = precision_recall_curve(all_labels_one_hot.ravel(), all_probs.ravel())
        auc_pr = auc(recall_curve, precision_curve)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, "
              f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, AUC-PR: {auc_pr:.4f}")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'models/checkpoints/best_model.pth')
            print("Best model saved!")
            epochs_no_improve = 0
        else:
            epochs_no_improve +=1

        # Early stopping check
        if epochs_no_improve >= patience:
            print(f"Early stopping after {epoch+1} epochs.")
            break

    # Plotting training and validation loss curves *after* training is complete
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.show()


def calculate_class_weights(labels):
    """Calculates class weights for imbalanced datasets."""
    class_labels = labels.cpu().numpy()
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(class_labels),
        y=class_labels
    )
    return torch.tensor(class_weights, dtype=torch.float)