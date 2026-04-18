"""
Download datasets for loss-of-plasticity experiments.
Datasets: MNIST, CIFAR-10, CIFAR-100, Continual ImageNet, Tiny ImageNet, Two Moons.
"""

import os
import pickle
import gdown
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import urllib.request
import zipfile
from PIL import Image
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
    out_dir = os.path.join(DATA_DIR, "mnist")
    os.makedirs(out_dir, exist_ok=True)
    mnist_path = os.path.join(out_dir, "mnist_")
    with open(mnist_path, "wb") as f:
        pickle.dump([x_train, y_train, x_test, y_test], f)
    
    print(f"MNIST saved to {mnist_path}")
    print(f"  Train: {x_train.shape}, Test: {x_test.shape}")


def download_cifar10():
    """
    Download CIFAR-10 dataset and save as pickle file.
    Output: data/cifar10_ containing [x_train, y_train, x_test, y_test]
    Shape stays (N, C, H, W) to match CNN inputs in loss-of-plasticity.
    """
    print("Downloading CIFAR-10...")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, transform=transform, download=True)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=50000, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=10000, shuffle=False)
    
    x_train, y_train = next(iter(train_loader))
    x_test, y_test = next(iter(test_loader))
    
    out_dir = os.path.join(DATA_DIR, "cifar10")
    os.makedirs(out_dir, exist_ok=True)
    cifar10_path = os.path.join(out_dir, "cifar10_")
    with open(cifar10_path, "wb") as f:
        pickle.dump([x_train, y_train, x_test, y_test], f)
    
    print(f"CIFAR-10 saved to {cifar10_path}")
    print(f"  Train: {x_train.shape}, Test: {x_test.shape}")


def download_cifar100():
    """
    Download CIFAR-100 dataset and save as pickle file.
    Output: data/cifar100_ containing [x_train, y_train, x_test, y_test]
    Shape stays (N, C, H, W) to match CNN inputs in loss-of-plasticity.
    """
    print("Downloading CIFAR-100...")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = torchvision.datasets.CIFAR100(root=DATA_DIR, train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.CIFAR100(root=DATA_DIR, train=False, transform=transform, download=True)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=50000, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=10000, shuffle=False)
    
    x_train, y_train = next(iter(train_loader))
    x_test, y_test = next(iter(test_loader))
    
    out_dir = os.path.join(DATA_DIR, "cifar100")
    os.makedirs(out_dir, exist_ok=True)
    cifar100_path = os.path.join(out_dir, "cifar100_")
    with open(cifar100_path, "wb") as f:
        pickle.dump([x_train, y_train, x_test, y_test], f)
    
    print(f"CIFAR-100 saved to {cifar100_path}")
    print(f"  Train: {x_train.shape}, Test: {x_test.shape}")


def download_tiny_imagenet():
    """
    Download Tiny ImageNet dataset and save as pickle file.
    Output: data/tiny_imagenet_ containing [x_train, y_train, x_test, y_test]
    Shape stays (N, C, H, W) to match CNN inputs in loss-of-plasticity.
    """
    print("Downloading Tiny ImageNet...")
    out_dir = os.path.join(DATA_DIR, "tinyImagenet")
    os.makedirs(out_dir, exist_ok=True)
    
    url = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(out_dir, "tiny-imagenet-200.zip")
    tiny_base_dir = os.path.join(out_dir, "tiny-imagenet-200")
    
    if not os.path.exists(tiny_base_dir):
        print(f"Downloading from {url}...")
        # Use wget or curl instead of urllib.request to avoid 28KB error block
        os.system(f"wget -q --show-progress -O {zip_path} {url} || curl -L -o {zip_path} {url}")
        if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 1000000:
            print("Download failed or zip is corrupted. Please try again.")
            return
        print("Extracting Tiny ImageNet...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(out_dir)
        os.remove(zip_path)
    
    print("Processing Tiny ImageNet...")
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Load Training Set
    train_dir = os.path.join(tiny_base_dir, 'train')
    train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    # 100,000 images is safe to load into RAM (takes ~1.2GB)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100000, shuffle=False)
    print("Loading Tiny ImageNet train images into memory...")
    x_train, y_train = next(iter(train_loader))
    
    # Load Validation Set (used as test set)
    val_dir = os.path.join(tiny_base_dir, 'val')
    val_images_dir = os.path.join(val_dir, 'images')
    
    print("Loading Tiny ImageNet test images into memory...")
    with open(os.path.join(val_dir, 'val_annotations.txt'), 'r') as f:
        val_annots = f.readlines()
        
    class_to_idx = train_dataset.class_to_idx
    val_images = []
    val_labels = []
    
    for line in val_annots:
        parts = line.strip().split('\t')
        img_name = parts[0]
        class_name = parts[1]
        
        img_path = os.path.join(val_images_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        val_images.append(transform(img))
        val_labels.append(class_to_idx[class_name])
        
    x_test = torch.stack(val_images)
    y_test = torch.tensor(val_labels)
    
    # Clean up to free memory before converting
    del train_dataset, train_loader, val_images, val_labels
    import gc
    gc.collect()

    # Save as pickle
    tiny_path = os.path.join(out_dir, "tiny_imagenet_")
    print(f"Saving Tiny ImageNet to {tiny_path} (this may take a minute)...")
    with open(tiny_path, "wb") as f:
        # Save large pickles efficiently using protocol=4
        pickle.dump([x_train, y_train, x_test, y_test], f, protocol=4)
        
    print(f"Tiny ImageNet saved to {tiny_path}")
    print(f"  Train: {x_train.shape}, Test: {x_test.shape}")


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
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)
    
    os.remove(zip_path)
    print(f"ImageNet saved to {imagenet_dir}/")


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
    out_dir = os.path.join(DATA_DIR, "two_moons")
    os.makedirs(out_dir, exist_ok=True)
    two_moons_path = os.path.join(out_dir, "two_moons_")
    with open(two_moons_path, "wb") as f:
        pickle.dump([x_train, y_train, x_test, y_test], f)
    
    print(f"Two Moons saved to {two_moons_path}")
    print(f"  Train: {x_train.shape}, Test: {x_test.shape}")


def main():
    print("=" * 50)
    print("Downloading datasets for loss-of-plasticity")
    print("=" * 50)
    
    # download_mnist()
    # print()
    
    # download_cifar10()
    # print()
    
    # download_cifar100()
    # print()
    
    download_tiny_imagenet()
    print()
    
    # download_imagenet_preprocessed()
    # print()
    
    generate_two_moons()
    print()
    
    print("=" * 50)
    print("All datasets downloaded successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
