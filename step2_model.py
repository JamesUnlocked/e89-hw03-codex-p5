"""Define the chapter 10 dense image classifier."""

import torch
from torch import nn


class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # CrossEntropyLoss expects raw logits, so omit a final softmax.
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

    def forward(self, images):
        return self.layers(images)


if __name__ == "__main__":
    torch.manual_seed(42)
    model = ImageClassifier()
    parameter_count = sum(p.numel() for p in model.parameters())
    # Check both the parameter total and the ten logits for each image.
    assert parameter_count == 266_610
    assert model(torch.zeros(3, 1, 28, 28)).shape == (3, 10)
    print(model)
    print(f"Parameter count: {parameter_count:,}")
