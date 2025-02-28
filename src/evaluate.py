# Evaluation script

import torch
import torch.nn as nn
from rich.progress import track
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc


def evaluate_test_set(model, test_loader, device="cpu"):
    """Evaluates the model on the test set.

    Args:
        model: The trained PyTorch model.
        test_loader: DataLoader for the test data.
        device: The device to use (CPU or GPU).

    Returns:
        The test loss, test accuracy, precision, recall, f1 score and AUC-PR.
    """

    criterion = nn.CrossEntropyLoss()
    model.eval()  # Set to evaluation mode
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in track(test_loader, description='Testing...'):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_test_loss = test_loss / test_total
    test_accuracy = 100 * test_correct / test_total

    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    all_probs = np.array(all_probs)
    all_labels_one_hot = np.eye(3)[all_labels] 
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels_one_hot.ravel(), all_probs.ravel())
    auc_pr = auc(recall_curve, precision_curve)

    return avg_test_loss, test_accuracy, precision, recall, f1, auc_pr