"""Plot the recorded training and validation accuracies."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_accuracy(history):
    # Show both curves on the same scale to compare fitting and generalization.
    epochs = range(1, len(history["train_accuracy"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_accuracy"], "o-", label="Training accuracy")
    ax.plot(epochs, history["valid_accuracy"], "s-", label="Validation accuracy")
    ax.set(title="FashionMNIST: Training and Validation Accuracy",
           xlabel="Epoch", ylabel="Accuracy", ylim=(0.7, 1.0))
    ax.set_xticks(list(epochs))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("training_accuracy.png", dpi=150)
    return fig


if __name__ == "__main__":
    history = json.loads(Path("history.json").read_text(encoding="utf-8"))
    figure = plot_accuracy(history)
    assert len(figure.axes[0].lines) == 2
    assert Path("training_accuracy.png").stat().st_size > 0
    plt.close(figure)
    print("Saved training_accuracy.png (20 epochs, training and validation).")
