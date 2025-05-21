"""
Copied and adapted from the colab:
https://colab.research.google.com/drive/1qnZBatCPSnjDW4lLm-3wT39Xb0A7ONMX?authuser=3&usp=classroom_web#scrollTo=FOOLJokMJ2kE
by Prof. Fabrizio Silvestri
"""
import torch
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

BATCH_SIZE = 256 if torch.cuda.is_available() else 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class OneLayerNetwork(nn.Module):
    def __init__(self):
        super(OneLayerNetwork, self).__init__()
        self.layer = nn.Sequential(nn.Linear(28 * 28, 10), nn.ReLU())

    def forward(self, x):
        return self.layer(x.view(x.size(0), -1))


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        # Compute prediction error
        prediction = model(X)
        loss = loss_fn(prediction, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        return loss


def main():
    pl.seed_everything(0)

    train_ds = MNIST(".", train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)

    model = OneLayerNetwork().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    epochs = 10
    losses = []

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        loss = train(train_loader, model, loss_fn, optimizer)
        losses.append(loss)  # Assuming train function returns loss value

    print("Done!")

    # Plotting loss per epoch
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), losses, marker='o', linestyle='-', color='b')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()