"""Evaluate the final checkpoint and inspect three validation predictions."""

import json
from pathlib import Path

import torch
from torchmetrics import Accuracy

from step1_data import get_device, load_data
from step2_model import ImageClassifier


def evaluate(model, valid_loader, test_loader, classes, device):
    model.eval()
    metric = Accuracy(task="multiclass", num_classes=10).to(device)
    metric.reset()
    with torch.inference_mode():
        for images, labels in test_loader:
            metric.update(model(images.to(device)), labels.to(device))
        test_accuracy = metric.compute().item()
        metric.reset()
        # Index the validation subset directly to preserve its first three images.
        examples = [valid_loader.dataset[index] for index in range(3)]
        images = torch.stack([image for image, _ in examples]).to(device)
        probabilities = model(images).softmax(dim=1).cpu()
        top_probabilities, top_indices = probabilities.topk(4, dim=1)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(3))
    print(f"Test accuracy: {test_accuracy:.4%}")
    print(f"Probability column order: {classes}")
    predictions = []
    for index, (_, label) in enumerate(examples):
        predicted = int(probabilities[index].argmax())
        top_four = [{"class": classes[int(class_id)], "probability": float(probability)}
                    for probability, class_id in zip(top_probabilities[index], top_indices[index])]
        print(f"\nValidation image {index + 1}: predicted={classes[predicted]} ({predicted}), "
              f"true={classes[label]} ({label})")
        print("Softmax: [" + ", ".join(f"{p:.3f}" for p in probabilities[index]) + "]")
        print("Top 4: " + ", ".join(f"{item['class']} ({item['probability']:.3f})" for item in top_four))
        predictions.append({"validation_index": index, "predicted_label": predicted,
                            "predicted_class": classes[predicted], "true_label": int(label),
                            "true_class": classes[label], "probabilities": probabilities[index].tolist(),
                            "top_4": top_four})
    # Preserve full precision in JSON; round only the displayed probabilities.
    result = {"test_accuracy": test_accuracy, "classes": classes, "predictions": predictions}
    Path("evaluation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    torch.set_num_threads(1)
    _, valid_loader, test_loader, classes = load_data()
    device = get_device()
    model = ImageClassifier().to(device)
    model.load_state_dict(torch.load("model.pt", map_location=device, weights_only=True))
    evaluate(model, valid_loader, test_loader, classes, device)
