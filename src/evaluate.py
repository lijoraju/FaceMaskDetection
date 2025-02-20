# Evaluation script

import torch
import torch.nn as nn
from rich.progress import track


def evaluate_test_set(model, test_loader, device="cpu"):
    """Evaluates the model on the test set.

    Args:
        model: The trained PyTorch model.
        test_loader: DataLoader for the test data.
        device: The device to use (CPU or GPU).

    Returns:
        The test loss and test accuracy.
    """

    criterion = nn.CrossEntropyLoss()
    model.eval()  # Set to evaluation mode
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        test_loop = track(test_loader, description="Evaluating Test Set")
        for images, labels in test_loop:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    avg_test_loss = test_loss / test_total
    test_accuracy = 100 * test_correct / test_total

    return avg_test_loss, test_accuracy