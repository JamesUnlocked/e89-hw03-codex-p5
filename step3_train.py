"""Train for 20 epochs and save the final weights and learning history."""

import json
from pathlib import Path

import torch
from torch import nn
from torchmetrics import Accuracy

from step1_data import get_device, load_data
from step2_model import ImageClassifier


def train(model, optimizer, criterion, train_loader, valid_loader, device, n_epochs=20):
    """Track sample-weighted loss and epoch-level multiclass accuracies."""
    metric = Accuracy(task="multiclass", num_classes=10).to(device)
    history = {"train_loss": [], "train_accuracy": [], "valid_accuracy": []}
    for epoch in range(n_epochs):
        model.train()
        metric.reset()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # Clear gradients before each update so batches do not accumulate.
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            metric.update(logits.detach(), labels)
        train_accuracy = metric.compute().item()
        metric.reset()
        model.eval()
        # Validation measures generalization without changing the weights.
        with torch.inference_mode():
            for images, labels in valid_loader:
                metric.update(model(images.to(device)), labels.to(device))
        valid_accuracy = metric.compute().item()
        metric.reset()
        history["train_loss"].append(total_loss / len(train_loader.dataset))
        history["train_accuracy"].append(train_accuracy)
        history["valid_accuracy"].append(valid_accuracy)
        print(f"Epoch {epoch + 1:02d}/{n_epochs}: "
              f"loss={history['train_loss'][-1]:.4f}, "
              f"train={train_accuracy:.4%}, valid={valid_accuracy:.4%}", flush=True)
    return history


def run_training():
    # One CPU thread avoids overhead for this small network and batch size.
    torch.set_num_threads(1)
    train_loader, valid_loader, _, _ = load_data()
    torch.manual_seed(42)
    device = get_device()
    model = ImageClassifier().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    history = train(model, optimizer, nn.CrossEntropyLoss(),
                    train_loader, valid_loader, device, n_epochs=20)
    Path("history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    # A CPU state_dict is portable when loaded on another device.
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, "model.pt")
    return model, history


if __name__ == "__main__":
    run_training()
