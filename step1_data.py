"""Download FashionMNIST, make the seeded split, and inspect a batch."""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST
from torchvision.transforms import v2 as T


def get_device():
    """Prefer CUDA, then Apple MPS, and otherwise use the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_data():
    # Convert PIL images into float32 tensors with pixels scaled to [0, 1].
    transform = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
    full_train = FashionMNIST(root="datasets", train=True, download=True, transform=transform)
    test_set = FashionMNIST(root="datasets", train=False, download=True, transform=transform)
    torch.manual_seed(42)
    train_set, valid_set = random_split(full_train, [55_000, 5_000])
    # Only training batches are shuffled; validation/test order stays fixed.
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=32)
    test_loader = DataLoader(test_set, batch_size=32)
    return train_loader, valid_loader, test_loader, full_train.classes


if __name__ == "__main__":
    train_loader, valid_loader, test_loader, classes = load_data()
    images, labels = next(iter(train_loader))
    assert [len(loader.dataset) for loader in (train_loader, valid_loader, test_loader)] == [55_000, 5_000, 10_000]
    assert images.shape == (32, 1, 28, 28) and images.dtype == torch.float32
    assert 0 <= images.min() <= images.max() <= 1
    assert set(train_loader.dataset.indices).isdisjoint(valid_loader.dataset.indices)
    print(f"Device: {get_device()}")
    print("Train: 55000; validation: 5000; test: 10000")
    print(f"Batch: {tuple(images.shape)}, {images.dtype}; classes: {classes}")
