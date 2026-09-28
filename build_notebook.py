"""Build a standalone notebook by embedding the verified script implementations."""

from pathlib import Path
import textwrap

import nbformat


def definitions(filename):
    source = Path(filename).read_text(encoding="utf-8")
    body, main = source.split('if __name__ == "__main__":\n')
    # Embed definitions in the notebook instead of importing project modules.
    body = "\n".join(line for line in body.splitlines() if not line.startswith("from step"))
    return body.strip(), textwrap.dedent(main).strip()


notebook = nbformat.v4.new_notebook()
notebook.metadata.kernelspec = {"display_name": "Python 3 (ipykernel)",
                                "language": "python", "name": "python3"}
notebook.cells = [nbformat.v4.new_markdown_cell(
    "# James Torres\n## CSCI E-89 Assignment 03, Problem 5\n\n"
    "Build and train a PyTorch FashionMNIST image classifier with two hidden layers, "
    "plot its learning progress, and evaluate its predictions. This notebook contains "
    "the full implementation and trains for 20 epochs without importing any project scripts.\n\n"
    "Based on the image-classifier example in chapter 10 of Aurélien Géron's "
    "*Hands-On Machine Learning with Scikit-Learn and PyTorch* (2025). "
    "[Author's reference notebook](https://github.com/ageron/handson-mlp/blob/main/10_neural_nets_with_pytorch.ipynb). "
    "Install dependencies from `requirements.txt` before running; the first run downloads FashionMNIST."
)]


def add(markdown, code):
    notebook.cells.extend([nbformat.v4.new_markdown_cell(markdown),
                           nbformat.v4.new_code_cell(code)])


data, data_run = definitions("step1_data.py")
model, model_run = definitions("step2_model.py")
training, _ = definitions("step3_train.py")
plotting, _ = definitions("step4_plot_accuracy.py")
evaluation, evaluation_run = definitions("step5_evaluate.py")

add("### 1. Prepare FashionMNIST\nDefine the float32 image transform, seed the "
    "55,000/5,000 split, build batch-32 loaders, and select CUDA, MPS, or CPU.", data)
add("Download the data and verify the split sizes, disjoint subsets, image shape, "
    "dtype, and pixel range.", "# Inspect one training batch and check the data contract.\n" + data_run)
add("### 2. Define the classifier\nFlatten each image and apply dense layers of "
    "300, 100, and 10 units, with ReLU between hidden layers. Return raw logits.", model)
add("Count the trainable parameters and verify that three images produce a 3-by-10 "
    "logit tensor.", model_run)
add("### 3. Train and record learning progress\nDefine the training loop with "
    "TorchMetrics multiclass accuracy reset between phases and epochs. Weight loss "
    "by batch size so the final short batch is handled correctly. Save final weights "
    "and history. Training accuracy is measured during updates; validation uses the "
    "fixed weights at the end of each epoch.", training)
add("Train a fresh seeded model for all 20 epochs with SGD (learning rate 0.1) and "
    "cross-entropy. Save `history.json` and `model.pt` in the working directory.",
    "# Reset data and model seeds for the same experiment as the training script.\n"
    "model, history = run_training()\n"
    "print(f\"Final validation accuracy: {history['valid_accuracy'][-1]:.4%}\")")
add("### 4. Plot accuracy\nDefine a labeled plot comparing training and validation "
    "accuracy across epochs; also save it as a PNG.", plotting)
add("Render the accuracy plot inline and save `training_accuracy.png`.",
    "# Use Jupyter's inline backend so the executed notebook embeds the figure.\n"
    "%matplotlib inline\n"
    "figure = plot_accuracy(history)\nplt.show()")
add("### 5. Evaluate the final model\nCompute accuracy on the untouched test set "
    "and inspect the first three validation images. Show true and predicted labels, "
    "all ten softmax probabilities to three decimals, and the four most probable classes.", evaluation)
add("Reload the final checkpoint and evaluate it. Save full-precision results to "
    "`evaluation.json`; only the printed probabilities are rounded.",
    "# Evaluate the saved checkpoint using the same seeded validation subset.\n" + evaluation_run)
add("Display the three validation images alongside their true and predicted classes.",
    "# Visualize the same examples used in the probability report.\n"
    "fig, axes = plt.subplots(1, 3, figsize=(10, 3))\n"
    "with torch.inference_mode():\n"
    "    for index, ax in enumerate(axes):\n"
    "        image, label = valid_loader.dataset[index]\n"
    "        predicted = model(image.unsqueeze(0).to(device)).argmax(dim=1).item()\n"
    "        ax.imshow(image.squeeze().numpy(), cmap='gray')\n"
    "        ax.set_title(f'True: {classes[label]}\\nPredicted: {classes[predicted]}')\n"
    "        ax.axis('off')\n"
    "fig.tight_layout()\nplt.show()")
nbformat.validate(notebook)
nbformat.write(notebook, "e89_Torres_James_HW03_Prob5.ipynb")
print("Created self-contained notebook with explanatory markdown before every code cell.")
