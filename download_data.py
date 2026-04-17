"""
Download datasets for loss-of-plasticity experiments.
Datasets: MNIST, CIFAR-10, CIFAR-100, Continual ImageNet, Two Moons.
"""

import os
import pickle
import gdown
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.datasets import make_moons

DATA_DIR = "data"


def download_mnist():
    """
    Download MNIST and save as pickle file matching loss-of-plasticity format.
    Output: data/mnist_ containing [x_train, y_train, x_test, y_test]
    where x is flattened to 784-dim vectors.
    """
    print("Downloading MNIST...")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = torchvision.datasets.MNIST(
        root=DATA_DIR, train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root=DATA_DIR, train=False, transform=transform, download=True
    )
    
    # Load all data at once
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=60000, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=10000, shuffle=False
    )
    
    # Get train data and flatten
    for images, labels in train_loader:
        x_train = images.flatten(start_dim=1)
        y_train = labels
    
    # Get test data and flatten
    for images, labels in test_loader:
        x_test = images.flatten(start_dim=1)
        y_test = labels
    
    # Save as pickle (matching load_mnist.py format)
    mnist_path = os.path.join(DATA_DIR, "mnist_")
    with open(mnist_path, "wb") as f:
        pickle.dump([x_train, y_train, x_test, y_test], f)
    
    print(f"MNIST saved to {mnist_path}")
    print(f"  Train: {x_train.shape}, Test: {x_test.shape}")


def download_cifar100():
    """
    Download CIFAR-100 dataset.
    Used by mlproj_manager.problems.CifarDataSet which expects standard torchvision format.
    """
    print("Downloading CIFAR-100...")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Download train and test sets
    torchvision.datasets.CIFAR100(root=DATA_DIR, train=True, download=True)
    torchvision.datasets.CIFAR100(root=DATA_DIR, train=False, download=True)
    
    print(f"CIFAR-100 saved to {DATA_DIR}/cifar-100-python/")


def download_imagenet_preprocessed():
    """
    Download preprocessed ImageNet data from Google Drive.
    Format: data/classes/<class_id>.npy for each of 1000 classes.
    Each .npy file contains 700 images (600 train + 100 test).
    """
    print("Downloading preprocessed ImageNet...")
    
    imagenet_dir = os.path.join(DATA_DIR, "classes")
    
    if os.path.exists(imagenet_dir) and len(os.listdir(imagenet_dir)) > 0:
        print(f"ImageNet data already exists at {imagenet_dir}")
        return
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Google Drive file ID from the README
    file_id = "1i0ok3LT5_mYmFWaN7wlkpHsitUngGJ8z"
    zip_path = os.path.join(DATA_DIR, "imagenet_classes.zip")
    
    # Download from Google Drive
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading from Google Drive (this may take a while)...")
    gdown.download(url, zip_path, quiet=False)
    
    # Extract
    print("Extracting ImageNet data...")
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)
    
    os.remove(zip_path)
    print(f"ImageNet saved to {imagenet_dir}/")


def download_cifar10():
    """Download CIFAR-10 dataset."""
    print("Downloading CIFAR-10...")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True)
    torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True)
    
    print(f"CIFAR-10 saved to {DATA_DIR}/cifar-10-batches-py/")


def generate_two_moons():
    """
    Generate Two Moons dataset and save as pickle file.
    Output: data/two_moons_ containing [x_train, y_train, x_test, y_test]
    """
    print("Generating Two Moons dataset...")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Generate train set (10000 samples)
    x_train, y_train = make_moons(n_samples=10000, noise=0.1, random_state=42)
    
    # Generate test set (2000 samples)
    x_test, y_test = make_moons(n_samples=2000, noise=0.1, random_state=43)
    
    # Convert to torch tensors
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()
    
    # Save as pickle (same format as MNIST)
    two_moons_path = os.path.join(DATA_DIR, "two_moons_")
    with open(two_moons_path, "wb") as f:
        pickle.dump([x_train, y_train, x_test, y_test], f)
    
    print(f"Two Moons saved to {two_moons_path}")
    print(f"  Train: {x_train.shape}, Test: {x_test.shape}")


def main():
    print("=" * 50)
    print("Downloading datasets for loss-of-plasticity")
    print("=" * 50)
    
    download_mnist()
    print()
    
    download_cifar10()
    print()
    
    download_cifar100()
    print()
    
    download_imagenet_preprocessed()
    print()
    
    generate_two_moons()
    print()
    
    print("=" * 50)
    print("All datasets downloaded successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
