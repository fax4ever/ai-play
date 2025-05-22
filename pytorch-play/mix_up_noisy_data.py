"""
Copied and adapted from the colab:
https://colab.research.google.com/drive/1qnZBatCPSnjDW4lLm-3wT39Xb0A7ONMX?authuser=3&usp=classroom_web#scrollTo=FOOLJokMJ2kE
by Prof. Fabrizio Silvestri
"""
import torch
from torch.utils.data import Dataset,  DataLoader
import torchvision
from torchvision import datasets
import pytorch_lightning as pl
import torchvision.transforms as transforms
from rich.pretty import pprint
import matplotlib.pyplot as plt
import numpy as np


BATCH_SIZE = 32


class Noisy_Images_CIFAR10(Dataset):
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False,
                 std= 0.1,
                 mean=0):
        self.cifar10 = datasets.CIFAR10(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download)
        self.std = std
        self.mean = mean

    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, index):
        image, label = self.cifar10[index]
        image = image + torch.randn(image.size()) * self.std + self.mean
        return image, label


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.dims = (3, 32, 32)
        self.num_classes = 10

    def prepare_data(self):
        # download
        torchvision.datasets.CIFAR10(self.data_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            cifar_full =  torchvision.datasets.CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = torch.utils.data.random_split(cifar_full, [45000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.cifar_test = torchvision.datasets.CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.cifar_train, batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return  torch.utils.data.DataLoader(self.cifar_val, batch_size=BATCH_SIZE)

    def test_dataloader(self):
        return  torch.utils.data.DataLoader(self.cifar_test, batch_size=BATCH_SIZE)


def base_dataset():
    pl.seed_everything(42)
    transform = transforms.Compose([transforms.ToTensor()])
    custom_dataset = Noisy_Images_CIFAR10(
        root='./data',
        train=True,
        transform=transform,
        download=True)
    pprint(custom_dataset[0][0].shape)
    img = custom_dataset[1][0]
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    # mix up
    # x^=λxi+(1−λ)xj and y^=λyi+(1−λ)yj
    # original paper: https://www.google.com/url?q=https%3A%2F%2Farxiv.org%2Fabs%2F1710.09412


def with_lightning():
    dm = CIFAR10DataModule()
    dm.prepare_data()
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Data shape: {data.shape}, Target shape: {target.shape}")
        if batch_idx >= 2:  # Only check the first 3 batches
            break
    print(f"Number of training samples: {len(dm.cifar_train)}")
    print(f"Number of validation samples: {len(dm.cifar_val)}")
    print(f"Number of test samples: {len(dm.cifar_test)}")


def main():
    base_dataset()
    with_lightning()


if __name__ == "__main__":
    main()