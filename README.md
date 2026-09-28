# FashionMNIST image classifier

James Torres — CSCI E-89 Assignment 03, Problem 5.

An executable reproduction of the image-classifier workflow in chapter 10 of
Aurélien Géron's *Hands-On Machine Learning with Scikit-Learn and PyTorch* (2025).
Reference: [author's chapter 10 notebook](https://github.com/ageron/handson-mlp/blob/main/10_neural_nets_with_pytorch.ipynb).
The implementation uses the requested dense network and a training loop following
the `train2` workflow, with sample-weighted average training loss.

## Setup and run order

Use Python 3.10 or newer from the repository directory:

```bash
python -m pip install -r requirements.txt
python step1_data.py
python step2_model.py
python step3_train.py
python step4_plot_accuracy.py
python step5_evaluate.py
```

The first run downloads FashionMNIST into `datasets/`. The split is seeded with
42 and contains 55,000 training and 5,000 validation examples, with 10,000 separate
test examples. Batches contain 32 images; only training batches are shuffled.
The model has 266,610 trainable parameters and outputs ten raw logits.
Training uses cross-entropy and SGD with learning rate 0.1 for all 20 epochs.
Accuracy uses TorchMetrics multiclass accuracy, reset between phases and epochs.
Training accuracy is measured during updates; validation accuracy is measured
after each epoch with fixed weights. The final epoch is used for test evaluation.

Outputs are `history.json`, `model.pt`, `training_accuracy.png`, and
`evaluation.json`. The evaluation script prints test accuracy, all ten softmax
probabilities to three decimals, and the four most probable classes for each of
the first three validation images. Dataset files and model weights are ignored
by git; regenerate the weights by running training.

## Self-contained notebook

`e89_Torres_James_HW03_Prob5.ipynb` includes the complete implementation without
imports from the step scripts. It trains a fresh model and renders the plot inline.
Every code cell has an explanatory markdown cell and code comments.

```bash
jupyter nbconvert e89_Torres_James_HW03_Prob5.ipynb --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=3600
jupyter nbconvert e89_Torres_James_HW03_Prob5.ipynb --to html
```

The notebook writes the same artifacts as the scripts. CPU training uses one
PyTorch thread to avoid excessive overhead for small matrix operations. CUDA,
then MPS, is selected when available; otherwise the model uses CPU. The split
and initialization are seeded, but results may vary across hardware and library
versions. Dependencies are lower bounds so pip can resolve a compatible
PyTorch/torchvision pair for the selected platform.

## Verified results

The CPU run completed all 20 epochs:

| Measurement | Result |
| --- | ---: |
| Trainable parameters | 266,610 |
| Final training accuracy | 92.86% |
| Final validation accuracy | 87.28% |
| Test accuracy | 87.42% |

The first three validation images are Sneaker, Coat, and Pullover, and all three
are classified correctly. These are final-epoch results, not the best validation
epoch. The learning curves show a growing training/validation gap and fluctuating
validation accuracy; higher training accuracy does not guarantee better test
performance.

Verified environment: Python 3.10.21, torch 2.14.0+cpu, torchvision 0.29.0+cpu,
torchmetrics 1.9.0, matplotlib 3.10.9, nbformat 5.11.1, nbconvert 7.17.1, and
ipykernel 7.3.0. Each step script was executed and checked before the next script
was created, and each has its own git commit. The notebook is then executed
independently using `python3` and exported to HTML.

`build_notebook.py` regenerates the notebook source from the verified scripts.
It embeds their definitions and removes cross-script imports. Running it clears
stored notebook outputs, so follow it with the execution and HTML export commands
above to refresh the deliverables.
